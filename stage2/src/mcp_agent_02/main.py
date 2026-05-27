"""
Stage 2: Final agent — search plus the browser reader tools
=============================================================
Same multi-server agent as Part 4, but the browser MCP server now
exposes two tools (`fetch` and `extract`). The system prompt is updated
to teach the model when to prefer each.

Prerequisites:
  - SearXNG running locally
  - camofox-browser running locally

Run with:
    mcp-agent-stage2

Try the WHOIS demo:
    > For google.com, extract the registrar, the registration expiration
    > date, and the list of nameservers. Use viewdns.info if you need to
    > look it up.
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime
import logging
from typing import Any

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_browser_config import MODEL_NAME, MODEL_TEMPERATURE, MODEL_THINKING, HISTORY_DIR, MAX_TOOL_RESULT_CHARS

log = logging.getLogger(__name__)

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

SYSTEM_PROMPT = """You are an assistant with access to two MCP servers:

  - search-server provides `search-server_search(query, max_results)`: a
    web search via a local SearXNG instance. Returns URLs with titles
    and snippets.
  - browser-server provides TWO tools:
      * `browser-server_fetch(url)`: opens a URL in a real browser and
        returns the page's accessibility-tree snapshot (the page
        contents in LLM-friendly form).
      * `browser-server_extract(url, schema)`: opens a URL, fetches the
        page, and populates the JSON Schema you provide using structured
        extraction. Returns clean JSON. Supports arrays, nested objects,
        and any schema shape JSON allows.

CRITICAL RULES:

  1. NEVER describe a tool call in words. If you decide to use a tool,
     emit the tool_call. Saying "let me use the tool" without actually
     calling it is wrong.

  2. When you receive search results, your next action MUST be a
     `fetch` or `extract` call on the most promising URL. Do not stop
     after a search. Do not summarize the snippets and call it done.

  3. When the user gives you a fresh question that requires looking
     something up, the very first action is `search-server_search`.

  4. After the fetch/extract returns, THEN answer the user's question
     from the returned content. Cite the URL you used.

When to use which browser tool:

  - Use `extract` when the user asks for specific named fields you can
    enumerate in advance (registrar, price, author, publication date,
    list of nameservers, etc.). Provide a JSON Schema with property
    descriptions that match how those fields would appear on the page.
  - Use `fetch` when the user asks an open-ended question, wants a
    free-form summary, or you cannot predict the field shape ahead of
    time.

For purely timeless questions (math, definitions, syntax, well-established
historical facts), answer directly without using any tool.
"""


class Agent:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.history_file = os.path.join(HISTORY_DIR, f"{self.session_id}.json")
        self.messages: list[dict[str, Any]] = []
        self.mcp_sessions: dict[str, ClientSession] = {}
        self.mcp_tools_by_server: dict[str, list[Any]] = {}
        self._tool_to_server: dict[str, str] = {}
        self.ollama_tools: list[dict] = []
        self._exit_stack = AsyncExitStack()

    async def connect(self, name: str, command: str, args: list[str]):
        params = StdioServerParameters(command=command, args=args)
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools = (await session.list_tools()).tools
        self.mcp_sessions[name] = session
        self.mcp_tools_by_server[name] = tools

        for t in tools:
            prefixed = f"{name}_{t.name}"
            self._tool_to_server[prefixed] = name

        log.info("connected to %r; tools: %s", name, [t.name for t in tools])

    def rebuild_ollama_tools(self):
        out = []
        for server_name, tools in self.mcp_tools_by_server.items():
            for tool in tools:
                out.append({
                    "type": "function",
                    "function": {
                        "name": f"{server_name}_{tool.name}",
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                })
        self.ollama_tools = out

    def build_messages_for_model(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.messages

    async def close(self):
        await self._exit_stack.aclose()

    def save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=4, default=str)

    async def handle_tools(self, tool_calls) -> dict:
        for tool in tool_calls:
            prefixed_name = tool.function.name
            args = tool.function.arguments or {}

            server_name = self._tool_to_server.get(prefixed_name)
            if not server_name:
                text = f"Unknown tool: {prefixed_name}"
            else:
                real_name = prefixed_name[len(server_name) + 1:]
                session = self.mcp_sessions[server_name]
                try:
                    result = await session.call_tool(real_name, args)
                    text = ""
                    for block in result.content:
                        if hasattr(block, "text"):
                            text += block.text
                except Exception as e:
                    text = f"Tool error on {server_name}: {e}"

            if len(text) > MAX_TOOL_RESULT_CHARS:
                head = MAX_TOOL_RESULT_CHARS // 2 - 100
                text = text[:head] + "\n...[TRUNCATED]...\n" + text[-head:]
            log.debug("tool result (%d chars): %s",
                      len(text),
                      text[:300].replace("\n", " ") + ("..." if len(text) > 300 else ""))
            self.messages.append({"role": "tool", "content": text})

        # Ask the model what to do next. It may either continue the chain
        # by calling more tools, or produce a final natural-language answer.
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=self.build_messages_for_model(),
            tools=self.ollama_tools,
            options={"temperature": MODEL_TEMPERATURE},
            think=MODEL_THINKING,
        )
        msg = resp["message"]

        # If the model wants more tool calls, recurse so the chain can
        # continue indefinitely. Without this, the agent could only do
        # one tool call per user turn, which makes search+fetch
        # composition impossible.
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                log.debug("chained tool call: %s(%s)",
                          tc.function.name, tc.function.arguments)
            self.messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
            return await self.handle_tools(msg.tool_calls)

        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        # Smaller models occasionally produce an empty turn after a tool
        # result. Nudge them once to either continue or finalize.
        log.debug("post-tool content preview: %r", content[:200])
        if not content.strip():
            log.debug("empty response after tool; nudging the model")
            self.messages.append({
                "role": "user",
                "content": "Based on the tool result above, either call another tool to continue, or give the user a final answer. Do not respond with empty text.",
            })
            resp = ollama.chat(
                model=MODEL_NAME,
                messages=self.build_messages_for_model(),
                tools=self.ollama_tools,
                options={"temperature": MODEL_TEMPERATURE},
                think=MODEL_THINKING,
            )
            msg = resp["message"]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                self.messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
                return await self.handle_tools(msg.tool_calls)
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

        return {"role": "assistant", "content": content}


async def _main():
    agent = Agent()
    try:
        await agent.connect("search-server", "mcp-search-part3", [])
        await agent.connect("browser-server", "mcp-browser-stage2", [])
        agent.rebuild_ollama_tools()

        log.info("agent session %s; %d tools available across %d servers",
                 agent.session_id, len(agent.ollama_tools), len(agent.mcp_sessions))
        print(f"\n--- Agent session: {agent.session_id} ---")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit"):
                break

            agent.messages.append({"role": "user", "content": user_input})

            resp = ollama.chat(
                model=MODEL_NAME,
                messages=agent.build_messages_for_model(),
                tools=agent.ollama_tools,
                options={"temperature": MODEL_TEMPERATURE},
                think=MODEL_THINKING,
            )
            msg = resp["message"]

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    log.debug("tool call: %s(%s)",
                              tc.function.name, tc.function.arguments)
                agent.messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
                final = await agent.handle_tools(msg.tool_calls)
                print(f"Assistant: {final['content']}\n")
                agent.messages.append(final)
            else:
                print(f"Assistant: {msg.content}\n")
                agent.messages.append({"role": "assistant", "content": msg.content})

            agent.save_history()
    finally:
        await agent.close()


def chat():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(_main())


if __name__ == "__main__":
    chat()

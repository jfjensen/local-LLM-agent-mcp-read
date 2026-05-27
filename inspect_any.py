"""
A tiny CLI MCP inspector. Use it to poke at any of the MCP servers in
this repo without needing the npx-based Inspector.

Examples:
    # List the tools exposed by a server:
    python inspect_any.py mcp_browser_02.main
    python inspect_any.py mcp_browser_02.main
    python inspect_any.py mcp_search_part3.main

    # Call a tool with simple key=value args:
    python inspect_any.py mcp_browser_02.main fetch --kv url=https://example.com

    # Call a tool with inline JSON (for nested args like `schema`):
    python inspect_any.py mcp_browser_02.main extract --args '{"url": "https://viewdns.info/whois/?domain=google.com", "schema": {"type": "object", "properties": {"registrar": {"type": "string"}, "expiration_date": {"type": "string"}}}}'

    # Or from a JSON file:
    python inspect_any.py mcp_browser_02.main extract --args-file extract_args.json
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def parse_kv(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


async def main(module_name: str, tool_name: str | None, tool_args: dict):
    params = StdioServerParameters(command="python", args=["-m", module_name])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print(f"Tools in {module_name}:")
            for t in tools:
                print(f"  - {t.name}: {t.description}")

            if tool_name:
                print(f"\nCalling {tool_name}({tool_args})...")
                result = await session.call_tool(tool_name, tool_args)
                for block in result.content:
                    if hasattr(block, "text"):
                        print(f"\n{block.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="MCP server module, e.g. mcp_browser_02.main")
    parser.add_argument("tool", nargs="?", help="Tool name to call (omit to just list)")
    parser.add_argument("--args", help="JSON string with the tool arguments")
    parser.add_argument("--args-file", help="Path to a JSON file with the tool arguments")
    parser.add_argument("--kv", nargs="*", default=[], help="key=value pairs")
    ns = parser.parse_args()

    if ns.args:
        tool_args = json.loads(ns.args)
    elif ns.args_file:
        with open(ns.args_file, "r", encoding="utf-8") as f:
            tool_args = json.load(f)
    elif ns.kv:
        tool_args = parse_kv(ns.kv)
    else:
        tool_args = {}

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main(ns.module, ns.tool, tool_args))

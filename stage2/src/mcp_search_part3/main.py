"""
Copy of Part 3's SearXNG MCP server, used by the Stage 2 agent.

This file is byte-for-byte the same as Part 3's stage 3 SearXNG MCP
server. We copy it in so the repo is self-contained.

Prerequisites:
    - SearXNG running locally (see Part 3, stage 1, or any SearXNG
      install with the JSON format enabled and the limiter disabled).

Run with:
    mcp-search-part3

Override the SearXNG URL if needed:
    SEARXNG_URL=http://localhost:8089 mcp-search-part3
"""

import logging
import httpx
from mcp.server.fastmcp import FastMCP

from mcp_browser_config import SEARXNG_URL

log = logging.getLogger(__name__)

mcp = FastMCP("search-server")


@mcp.tool()
def search(query: str, max_results: int = 5) -> str:
    """
    Search the web via a local SearXNG instance. You MUST use this tool
    for any question that requires up-to-date or specific information.

    Examples of when to use this tool:
      - "What is the latest version of Python?"
      - "What did <company> announce this week?"
      - "Who won <recent event>?"
      - "What is the current price of <product>?"
      - "Find the WHOIS lookup page for google.com"

    Do NOT use this tool for math, definitions, syntax questions, or
    anything that does not depend on current real-world information.

    Returns the top results, each with a title, URL, and snippet.
    """
    log.info("search %r (max %d results)", query, max_results)
    try:
        response = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("search failed: %s", e)
        return f"Search failed: {e}. Is SearXNG running at {SEARXNG_URL}?"

    data = response.json()
    results = data.get("results", [])[:max_results]
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        snippet = r.get("content", "")
        lines.append(f"[{i}] {title}\n    {url}\n    {snippet}")

    return "\n\n".join(lines)


def chat():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    chat()

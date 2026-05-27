# Local LLM agent with smarter web reading (camofox-browser + MCP)

Code for Part 5 of the *Build Your Own Claude Code* series:
*Reading Whole Web Pages with a Small Local LLM, without Truncation*.

The series so far:
- Part 1: the agent (CLI, tools, skills, history, compaction)
- Part 2: the browser UI (FastAPI + WebSockets)
- Part 3: web search (SearXNG via MCP)
- Part 4: web browsing (camofox-browser via MCP) + composing with Part 3's search
- Part 5 (this repo): reading whole pages without truncation — chunked
  `extract` and `summarize`, plus a family of small single-purpose
  reader tools (`fetch_snippet`, `fetch_urls`, `fetch_structure`)

The problem this part solves: a small local model has a limited context
window, so the obvious move of dumping a whole page snapshot into it
forces truncation, and middle-dropping truncation silently discards the
body content (a Wikipedia infobox, a data table) that you were after.
The fix is to keep the full snapshot inside the MCP server and have each
tool return only a small, purposeful slice: chunk-and-merge for
`extract`, refine for `summarize`, and a head slice / link list / heading
outline for the lightweight reader tools.

## Install

```bash
git clone https://github.com/jfjensen/local-LLM-agent-mcp-read.git
cd local-LLM-agent-mcp-read
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -e .
```

You will also need:

- **Docker** to run camofox-browser and SearXNG (Stage 1 brings up both).
- **Ollama** with a tool-capable model. The default is `qwen3.5:9b`:
  ```bash
  ollama pull qwen3.5:9b
  ```

SearXNG comes pre-configured (the settings file is mounted into the
container), so the JSON API works without editing anything.

## Configuration

All tunable settings live in `config.toml` at the repo root:

- model name, temperature, thinking;
- SearXNG and camofox URLs;
- snapshot, chunking, and reading budgets;
- log verbosity (`logging.level = "DEBUG"` shows each tool call, its
  arguments, and a preview of what came back).

To see what is currently active:

```bash
mcp-config-show
```

The loader looks for `config.toml` in the current working directory
first, then falls back to the one bundled with the repo.

## How to run

| Stage | What it is                                                       | How to run |
|-------|------------------------------------------------------------------|------------|
| 1     | Standing up camofox-browser and SearXNG via Docker               | `cd stage1 && docker compose up -d` (see `stage1/README.md` for the camofox image build step) |
| 2     | The full agent: search plus the browser reader tools             | `mcp-agent-stage2` |

The browser MCP server (`mcp-browser-stage2`) exposes five tools:

- **`fetch_snippet`** — the head of a page, for a quick look.
- **`fetch_urls`** — the page's links as `{text, url}` pairs, absolute.
- **`fetch_structure`** — the heading outline.
- **`extract`** — named fields as JSON, via chunk-and-merge over the
  whole page (no truncation).
- **`summarize`** — a prose summary, built by refining a running summary
  chunk by chunk.

Plus the search server (`mcp-search-part3`), a copy of Part 3's SearXNG
MCP server, so the repo is self-contained.

## Probing MCP servers with `inspect_any.py`

```bash
# List a server's tools:
python inspect_any.py mcp_browser_02.main

# Call a reader tool:
python inspect_any.py mcp_browser_02.main fetch_snippet --kv url=https://example.com
python inspect_any.py mcp_browser_02.main fetch_urls --kv url=https://example.com

# Call extract with a JSON Schema (use --args or --args-file for nested args):
python inspect_any.py mcp_browser_02.main extract --args-file extract_args.json
```

Where `extract_args.json` might look like:

```json
{
  "url": "https://en.wikipedia.org/wiki/Vleteren",
  "schema": {
    "type": "object",
    "properties": {
      "mayor": {"type": "string", "description": "The current mayor, from the infobox"},
      "postal_code": {"type": "string", "description": "The postal code"},
      "population": {"type": "string", "description": "The total population"}
    }
  }
}
```

## Notes

- The agent creates a `history/` folder in the current working directory
  on first run.
- The repo ships a copy of Part 3's SearXNG MCP server as
  `mcp-search-part3`, byte-for-byte the same, so you do not need to
  install Part 3.

## Troubleshooting

- **The Docker build fails with "dist not found".** Use `Dockerfile.ci`
  instead of the default `Dockerfile`. See `stage1/README.md`.
- **`FileNotFoundError: [WinError 2]` when the agent spawns an MCP server.**
  Your venv is not activated. Activate it so console scripts are on `PATH`.
- **Script exits silently on Windows.** The default asyncio event loop on
  Windows cannot spawn subprocesses. The agent sets
  `WindowsProactorEventLoopPolicy` at startup; do the same if you copy
  the code elsewhere.
- **`extract` returns nulls on a page you know has the data.** With the
  chunked extract this should be rare, but a very large page makes many
  model calls. Lower `chunking.chunk_chars` for smaller, more numerous
  chunks, or raise it for fewer, larger ones.
- **`fetch_structure` returns few or no headings.** Some pages (short
  stubs, pages whose content lives in tables or infoboxes) have a thin
  heading outline. Use `summarize` or `extract` for those.
- **camofox returns a small or empty snapshot.** Some pages need more
  than the default 1.5-second settle. Bump `browser.settle_seconds`.

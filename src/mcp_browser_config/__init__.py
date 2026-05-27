"""
Tiny config loader for the Part 5 project. Every stage imports from
here to find its model name, service URLs, and tunable parameters.

The config lives in `config.toml` at the top of the repo. It is
loaded once when this module is imported. So the rule is: edit
config.toml, restart the stage, done.

If `config.toml` is not found in the current working directory, we
fall back to the bundled default that ships with the package. This
means the stages still work when you run them from a fresh folder
(e.g. `cd my-session && mcp-agent-stage2`), but you can also drop
your own `config.toml` next to your work folder to override.

Importing this module also configures the root logger using the
`logging.level` config value. So every stage that does
`import logging; log = logging.getLogger(__name__)` immediately gets
the right verbosity. Logs go to stderr, so MCP stdio servers stay
clean on stdout.
"""

import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

# Defaults, used if neither the cwd nor the package ships a config.
# Kept in sync with config.toml at the repo root.
_DEFAULTS: dict[str, Any] = {
    "model": {
        "name": "qwen3.5:9b",
        "temperature": 0.1,
        "thinking": False,
    },
    "searxng": {
        "url": "http://localhost:8090",
    },
    "camofox": {
        "url": "http://localhost:9500",
    },
    "browser": {
        "max_snapshot_chars": 30000,
        "settle_seconds": 1.5,
    },
    "chunking": {
        "chunk_chars": 6000,
        "overlap_lines": 3,
    },
    "reading": {
        "snippet_chars": 4000,
        "max_link_label_chars": 100,
        "summarize_strategy": "mapreduce",
    },
    "agent": {
        "history_dir": "history",
        "max_tool_result_chars": 30000,
    },
    "logging": {
        "level": "INFO",
    },
}


def _find_config_file() -> Path | None:
    """Look for config.toml. Check cwd first, then the package install dir."""
    cwd_path = Path.cwd() / "config.toml"
    if cwd_path.is_file():
        return cwd_path

    # Bundled with the package: src/mcp_browser_config/../../config.toml
    pkg_path = Path(__file__).resolve().parent.parent.parent / "config.toml"
    if pkg_path.is_file():
        return pkg_path

    return None


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay into base, recursing into nested dicts."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load() -> dict:
    path = _find_config_file()
    if path is None:
        return _DEFAULTS

    with open(path, "rb") as f:
        loaded = tomllib.load(f)
    return _deep_merge(_DEFAULTS, loaded)


# Loaded once at import time.
_CONFIG = _load()


# --- Public surface ---

# Model
MODEL_NAME: str = _CONFIG["model"]["name"]
MODEL_TEMPERATURE: float = _CONFIG["model"]["temperature"]
MODEL_THINKING: bool = _CONFIG["model"]["thinking"]

# Services
SEARXNG_URL: str = _CONFIG["searxng"]["url"]
CAMOFOX_URL: str = _CONFIG["camofox"]["url"]

# Browser MCP server
MAX_SNAPSHOT_CHARS: int = _CONFIG["browser"]["max_snapshot_chars"]
SETTLE_SECONDS: float = _CONFIG["browser"]["settle_seconds"]

# Chunking (used by extract and summarize for large snapshots)
CHUNK_CHARS: int = _CONFIG["chunking"]["chunk_chars"]
OVERLAP_LINES: int = _CONFIG["chunking"]["overlap_lines"]

# Reading (lightweight reader tools)
SNIPPET_CHARS: int = _CONFIG["reading"]["snippet_chars"]
MAX_LINK_LABEL_CHARS: int = _CONFIG["reading"]["max_link_label_chars"]
SUMMARIZE_STRATEGY: str = _CONFIG["reading"]["summarize_strategy"]

# Agent
HISTORY_DIR: str = _CONFIG["agent"]["history_dir"]
MAX_TOOL_RESULT_CHARS: int = _CONFIG["agent"]["max_tool_result_chars"]

# Logging
LOG_LEVEL: str = _CONFIG["logging"]["level"]


def _setup_logging():
    """Configure logging from LOG_LEVEL. Logs go to stderr so MCP stdio
    servers stay clean on stdout.

    Important: we apply LOG_LEVEL only to OUR packages (mcp_agent_*,
    mcp_browser_*, mcp_search_*, mcp_browser_config). Third-party
    libraries (httpcore, httpx, asyncio, anyio, mcp) stay at WARNING
    regardless. Otherwise turning DEBUG on for our agent code also
    floods the terminal with HTTP transport chatter we never want.
    """
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    # The root logger collects everything; we pin it at WARNING so
    # libraries are quiet by default.
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)

    # Our own packages get the configured level.
    for name in (
        "mcp_browser_config",
        "mcp_browser_02",
        "mcp_agent_02",
        "mcp_search_part3",
    ):
        logging.getLogger(name).setLevel(level)


_setup_logging()


def show() -> None:
    """Print the active config. Useful for debugging."""
    path = _find_config_file()
    src = str(path) if path else "(built-in defaults)"
    print(f"Config source: {src}")
    print(f"  model:                   {MODEL_NAME}")
    print(f"  model.temperature:       {MODEL_TEMPERATURE}")
    print(f"  model.thinking:          {MODEL_THINKING}")
    print(f"  searxng.url:             {SEARXNG_URL}")
    print(f"  camofox.url:             {CAMOFOX_URL}")
    print(f"  browser.max_snapshot:    {MAX_SNAPSHOT_CHARS}")
    print(f"  browser.settle_seconds:  {SETTLE_SECONDS}")
    print(f"  chunking.chunk_chars:    {CHUNK_CHARS}")
    print(f"  chunking.overlap_lines:  {OVERLAP_LINES}")
    print(f"  reading.snippet_chars:   {SNIPPET_CHARS}")
    print(f"  reading.max_link_label:  {MAX_LINK_LABEL_CHARS}")
    print(f"  reading.summarize:       {SUMMARIZE_STRATEGY}")
    print(f"  agent.history_dir:       {HISTORY_DIR}")
    print(f"  agent.max_tool_result:   {MAX_TOOL_RESULT_CHARS}")
    print(f"  logging.level:           {LOG_LEVEL}")


if __name__ == "__main__":
    show()

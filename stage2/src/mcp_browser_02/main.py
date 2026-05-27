"""
Stage 2: Browser MCP server (snippet, urls, structure, extract, summarize)
=======================================================
Adds a second tool, `extract`, that uses camofox's POST /tabs/{tabId}/extract
endpoint. The model passes a JSON Schema describing the fields it wants,
camofox runs the extraction server-side, and the model gets back
structured JSON instead of an unstructured snapshot.

This is faster, cheaper, and more reliable than asking the model to
extract fields from a raw snapshot — the work happens once on the
server, not in the model's head.

Prerequisites:
  - camofox-browser running locally (see ../stage1/)

Run with:
    mcp-browser-stage2

Probe with:
    python ../inspect_any.py mcp_browser_02.main
    python ../inspect_any.py mcp_browser_02.main fetch --kv url=https://example.com
"""

import json
import logging
import re
import uuid
import time
from urllib.parse import urljoin
import httpx
import ollama
from mcp.server.fastmcp import FastMCP

from mcp_browser_config import (
    CAMOFOX_URL,
    SETTLE_SECONDS,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    CHUNK_CHARS,
    OVERLAP_LINES,
    SNIPPET_CHARS,
    MAX_LINK_LABEL_CHARS,
    SUMMARIZE_STRATEGY,
)

log = logging.getLogger(__name__)

mcp = FastMCP("browser-server")


def _open_tab(client: httpx.Client, user_id: str, url: str) -> str:
    r = client.post(
        f"{CAMOFOX_URL}/tabs/open",
        json={"userId": user_id, "url": url},
        timeout=60.0,
    )
    r.raise_for_status()
    body = r.json()
    tab_id = body.get("tabId") or body.get("id")
    if not tab_id:
        raise RuntimeError(f"camofox returned no tabId: {body}")
    return tab_id


def _get_snapshot(client: httpx.Client, user_id: str, tab_id: str) -> str:
    r = client.get(
        f"{CAMOFOX_URL}/tabs/{tab_id}/snapshot",
        params={"userId": user_id},
        timeout=30.0,
    )
    r.raise_for_status()
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        data = r.json()
        return (
            data.get("snapshot")
            or data.get("aria")
            or data.get("text")
            or str(data)
        )
    return r.text


def _close_tab(client: httpx.Client, user_id: str, tab_id: str) -> None:
    try:
        client.delete(
            f"{CAMOFOX_URL}/tabs/{tab_id}",
            params={"userId": user_id},
            timeout=10.0,
        )
    except Exception:
        pass


def _snippet(snapshot: str) -> str:
    """Return the head of the snapshot, up to SNIPPET_CHARS. If the page
    was longer, append a marker telling the model the slice is partial
    and which tools to use for the rest."""
    if len(snapshot) <= SNIPPET_CHARS:
        return snapshot
    head = snapshot[:SNIPPET_CHARS]
    marker = (
        "\n\n...[snippet ends here. This is only the top of the page. "
        "Use `summarize` for the full content, or `extract` for specific "
        "named fields.]"
    )
    return head + marker


# A link in the snapshot is a label line followed by an indented /url line:
#     - link "West Flanders" [e67]:
#       - /url: /wiki/West_Flanders
# The label is the quoted string; a label-less link looks like `- link [e40]:`.
_LINK_RE = re.compile(r'-\s+link\s+"(?P<label>.*)"\s+\[e\d+\]:\s*$')
_URL_RE = re.compile(r'-\s+/url:\s*(?P<url>.+?)\s*$')


def _parse_links(snapshot: str, base_url: str) -> list[dict]:
    """Extract {text, url} pairs from a snapshot.

    Rules (confirmed against real camofox output):
      - pair each `- link "label" [eN]:` with the following `- /url:` line
      - strip surrounding quotes from the URL
      - drop links whose URL is a pure fragment ("#...")
      - drop label-less links
      - resolve relative and protocol-relative URLs to absolute
      - dedupe by resolved URL, keeping the first label seen
      - cap label length at MAX_LINK_LABEL_CHARS
    """
    lines = snapshot.splitlines()
    out: list[dict] = []
    seen: set[str] = set()

    for i, line in enumerate(lines):
        m = _LINK_RE.search(line)
        if not m:
            continue
        label = m.group("label").strip()
        if not label:
            continue  # drop label-less links

        # The /url line is normally the next non-empty line.
        url_raw = None
        for j in range(i + 1, min(i + 3, len(lines))):
            um = _URL_RE.search(lines[j])
            if um:
                url_raw = um.group("url").strip()
                break
        if not url_raw:
            continue

        # Strip surrounding quotes.
        if len(url_raw) >= 2 and url_raw[0] == '"' and url_raw[-1] == '"':
            url_raw = url_raw[1:-1]

        if not url_raw or url_raw.startswith("#"):
            continue  # drop pure fragments

        absolute = urljoin(base_url, url_raw)
        if absolute in seen:
            continue
        seen.add(absolute)

        if len(label) > MAX_LINK_LABEL_CHARS:
            label = label[: MAX_LINK_LABEL_CHARS - 1].rstrip() + "\u2026"

        out.append({"text": label, "url": absolute})

    return out


# A heading line carries a level but (per camofox) no element ref:
#     - heading "References" [level=2]
_HEADING_RE = re.compile(r'-\s+heading\s+"(?P<text>.*)"\s+\[level=(?P<level>\d+)\]')

# Heading texts that are page chrome, not article sections. Wikipedia and
# many other sites render a navigation/table-of-contents heading that is
# not part of the document outline.
_CHROME_HEADINGS = {"contents", "navigation menu", "navigation", "menu"}


def _parse_headings(snapshot: str) -> list[dict]:
    """Extract the heading outline as {level, text} entries, in order,
    skipping chrome headings like 'Contents' that are navigation, not
    article structure."""
    out: list[dict] = []
    for line in snapshot.splitlines():
        m = _HEADING_RE.search(line)
        if m:
            text = m.group("text").strip()
            if text.lower() in _CHROME_HEADINGS:
                continue
            out.append({"level": int(m.group("level")), "text": text})
    return out


def _chunk_by_lines(snapshot: str, chunk_chars: int, overlap_lines: int) -> list[str]:
    """Split a snapshot into overlapping chunks on line boundaries.

    The snapshot is line-oriented (each element is its own line), so we
    pack whole lines into a chunk until adding the next would exceed
    chunk_chars, then start a new chunk that repeats the last
    overlap_lines lines of the previous one. The overlap means a fact
    sitting on a chunk boundary is seen whole by at least one chunk,
    instead of being cut in half and missed by both.
    """
    lines = snapshot.splitlines()
    if not lines:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        add_len = len(line) + 1  # +1 for the rejoined newline

        # A single line bigger than the whole budget gets its own chunk;
        # splitting inside a line would break an accessibility element.
        if add_len > chunk_chars and not cur:
            chunks.append(line)
            i += 1
            continue

        if cur_len + add_len > chunk_chars and cur:
            chunks.append("\n".join(cur))
            if overlap_lines > 0:
                cur = cur[-overlap_lines:]
                cur_len = sum(len(x) + 1 for x in cur)
            else:
                cur = []
                cur_len = 0
            continue  # re-test the same line against the fresh chunk

        cur.append(line)
        cur_len += add_len
        i += 1

    if cur:
        chunks.append("\n".join(cur))

    return chunks


def _merge_extractions(schema: dict, partials: list[dict]) -> dict:
    """Combine per-chunk extraction results into one object.

    Scalar fields take the first non-null value found across chunks.
    Array fields are concatenated across chunks, in order, de-duplicated.

    Note: array merging is a UNION across chunks, so it tends to
    OVER-collect: a chunk that mentions a loosely-related item adds it to
    the list. The caller runs `_clean_array_field` afterwards to prune
    the union back to genuine matches.
    """
    props = schema.get("properties", {})
    out: dict = {}
    for field, spec in props.items():
        if spec.get("type") == "array":
            seen = set()
            merged: list = []
            for p in partials:
                val = p.get(field)
                if isinstance(val, list):
                    for item in val:
                        key = item if isinstance(item, (str, int, float, bool)) else str(item)
                        if key not in seen:
                            seen.add(key)
                            merged.append(item)
            out[field] = merged
        else:
            value = None
            for p in partials:
                val = p.get(field)
                if val not in (None, "", "null"):
                    value = val
                    break
            out[field] = value
    return out


def _clean_array_field(field: str, description: str, candidates: list) -> list:
    """Prune a merged array down to the items that genuinely match the
    field. The union from `_merge_extractions` over-collects across
    chunks (e.g. a 'major versions of Llama' field picking up Alpaca and
    Meditron, which are derived models, not versions). One LLM call with
    the field description filters the candidates back to real matches.

    Order and exact strings are preserved; the model only removes items.
    On any failure the original candidates are returned unchanged.
    """
    if len(candidates) <= 1:
        return candidates
    prompt = (
        "You are cleaning a list of candidate values that were collected "
        "for one field of a data-extraction result. Some candidates were "
        "picked up by mistake and do not actually belong. Keep only the "
        "candidates that genuinely match the field, using its description. "
        "Do not add anything. Do not reword kept items; return them "
        "exactly as given. Respond with ONLY a JSON array of the kept "
        "strings.\n\n"
        f"FIELD: {field}\n"
        f"DESCRIPTION: {description}\n"
        f"CANDIDATES: {json.dumps(candidates, ensure_ascii=False)}"
    )
    try:
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": MODEL_TEMPERATURE},
            format="json",
            think=False,
        )
        cleaned = json.loads(resp["message"]["content"])
        # The model may return {"items": [...]} or a bare [...]; handle both.
        if isinstance(cleaned, dict):
            for v in cleaned.values():
                if isinstance(v, list):
                    cleaned = v
                    break
        if isinstance(cleaned, list) and cleaned:
            # Only keep items that were actually in the candidate set, so
            # the model cannot invent new ones.
            allowed = set(map(str, candidates))
            kept = [c for c in cleaned if str(c) in allowed]
            return kept or candidates
        return candidates
    except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
        log.warning("array cleanup failed for %r: %s", field, e)
        return candidates


def _extract_from_chunk(schema: dict, chunk: str, url: str) -> dict:
    """Run the extraction model over a single chunk. Returns a dict
    (possibly with many null fields). On any failure returns {} so the
    caller can keep going with the other chunks."""
    prompt = (
        "You are a precise data extraction tool. Read the page snapshot "
        "below and return a JSON object that matches the schema. Use the "
        "property descriptions to find the right values. If a field is "
        "not present in THIS snapshot, set it to null. Do not invent "
        "values. Do not explain. Respond with ONLY the JSON object.\n\n"
        f"SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
        f"PAGE SNAPSHOT (from {url}):\n{chunk}"
    )
    try:
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": MODEL_TEMPERATURE},
            format="json",
            think=False,
        )
        return json.loads(resp["message"]["content"])
    except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
        log.warning("chunk extraction failed: %s", e)
        return {}


def _open_settle_snapshot(url: str, user_id: str) -> str:
    """Open a tab, wait, snapshot, close. Shared by extract and summarize.
    Returns the raw (untruncated) snapshot, or raises on failure."""
    one_shot = not user_id
    if one_shot:
        user_id = f"oneshot-{uuid.uuid4().hex[:8]}"
    with httpx.Client() as client:
        tab_id = _open_tab(client, user_id, url)
        time.sleep(SETTLE_SECONDS)
        try:
            snapshot = _get_snapshot(client, user_id, tab_id)
        finally:
            if one_shot:
                _close_tab(client, user_id, tab_id)
    return snapshot


@mcp.tool()
def fetch_snippet(url: str, user_id: str = "") -> str:
    """
    Fetch a webpage and return a short snippet from the top of it.

    This is the quick-look tool. It returns the head of the page's
    accessibility-tree snapshot, which is usually enough to tell what
    the page is and whether it is the right one. If the page is longer
    than the snippet, the result ends with a marker telling you to use
    `summarize` for the full content or `extract` for specific fields.

    Use this when you want a fast look at a page, or to confirm a URL is
    what you expect before doing more with it. For a full understanding
    of a long page, prefer `summarize`; for named fields, prefer
    `extract`.

    Args:
        url: The full URL to fetch (must include http:// or https://).
        user_id: Optional. If set, camofox reuses a browser context
            across calls (faster). Default opens a one-shot tab.

    Returns:
        The head of the page snapshot, with a marker if it was longer.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https://; got {url!r}"

    log.info("fetch_snippet %s", url)
    try:
        snapshot = _open_settle_snapshot(url, user_id)
    except Exception as e:
        log.warning("fetch_snippet failed: %s", e)
        return f"Error fetching snapshot: {e}"

    log.debug("snapshot is %d chars; snippet budget %d", len(snapshot), SNIPPET_CHARS)
    return _snippet(snapshot)


@mcp.tool()
def fetch_urls(url: str, user_id: str = "") -> str:
    """
    Fetch a webpage and return the links on it as a list of
    {text, url} pairs, where text is the link's visible label and url is
    the absolute target.

    Use this when you need to navigate from a page: to find which link to
    follow next, or to see what a page links out to. The list is
    deduplicated and the URLs are made absolute, so you can pass any of
    them straight to another tool.

    Args:
        url: The full URL to read links from.
        user_id: Optional, same semantics as for `fetch_snippet`.

    Returns:
        A JSON array of {"text": ..., "url": ...} objects.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https://; got {url!r}"

    log.info("fetch_urls %s", url)
    try:
        snapshot = _open_settle_snapshot(url, user_id)
    except Exception as e:
        log.warning("fetch_urls failed: %s", e)
        return f"Error fetching snapshot: {e}"

    links = _parse_links(snapshot, url)
    log.debug("found %d links on %s", len(links), url)
    if not links:
        return "No links found on the page."
    return json.dumps(links, indent=2, ensure_ascii=False)


@mcp.tool()
def fetch_structure(url: str, user_id: str = "") -> str:
    """
    Fetch a webpage and return its heading outline: the page's headings
    with their levels, in order, like a table of contents.

    Use this to see how a page is organized and whether the section you
    want is on it, before deciding what to read in full. Note that some
    pages (short stubs, pages whose content sits in tables or infoboxes
    rather than under headings) have a thin outline; in that case prefer
    `summarize` or `extract`.

    Args:
        url: The full URL to outline.
        user_id: Optional, same semantics as for `fetch_snippet`.

    Returns:
        A plain-text outline, one heading per line, indented by level.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https://; got {url!r}"

    log.info("fetch_structure %s", url)
    try:
        snapshot = _open_settle_snapshot(url, user_id)
    except Exception as e:
        log.warning("fetch_structure failed: %s", e)
        return f"Error fetching snapshot: {e}"

    headings = _parse_headings(snapshot)
    log.debug("found %d headings on %s", len(headings), url)
    if not headings:
        return "No headings found on the page."

    lines = []
    for h in headings:
        indent = "  " * (h["level"] - 1)
        lines.append(f"{indent}h{h['level']} {h['text']}")
    return "\n".join(lines)


@mcp.tool()
def extract(url: str, schema: dict, user_id: str = "") -> str:
    """
    Fetch a webpage and extract structured data from it according to a
    JSON Schema. The MCP server fetches the page, then asks a local
    Ollama model to populate the schema from the page contents. So the
    caller gets clean JSON back, without having to read or parse the
    snapshot itself.

    Use this tool when the user asks for specific fields that you can
    name in advance, especially on pages with structured content (WHOIS
    lookups, product pages, GitHub repos, recipes, tables of data).
    Arrays and nested objects are supported, since the work is done by
    an LLM and not by a constrained server-side extractor.

    Prefer `fetch` when the user asks an open-ended question or wants a
    free-form summary.

    Args:
        url: The full URL to extract from.
        schema: A JSON Schema describing the fields to extract. Property
            descriptions guide the extraction model in finding the right
            page content. Example:
                {
                    "type": "object",
                    "properties": {
                        "registrar": {"type": "string",
                                      "description": "Domain registrar name"},
                        "expiration_date": {"type": "string",
                                            "description": "Registrar Registration Expiration Date"},
                        "nameservers": {"type": "array",
                                        "items": {"type": "string"},
                                        "description": "Name Server entries"}
                    }
                }
        user_id: Optional, same semantics as for `fetch`.

    Returns:
        A JSON string with the extracted fields. If a field cannot be
        found, it is set to null.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https://; got {url!r}"

    try:
        snapshot = _open_settle_snapshot(url, user_id)
    except Exception as e:
        log.warning("fetch for extract failed: %s", e)
        return f"Error fetching snapshot: {e}"

    # Chunk the snapshot instead of truncating it. A page smaller than
    # one chunk is processed in a single pass; a larger page is split
    # into overlapping chunks, each extracted, then merged. This is what
    # lets us pull a field out of the middle of a long page (a Wikipedia
    # infobox, say) instead of dropping the middle on the floor.
    chunks = _chunk_by_lines(snapshot, CHUNK_CHARS, OVERLAP_LINES)
    log.info("extracting from %s: %d chars -> %d chunk(s), %d schema properties",
             url, len(snapshot), len(chunks), len(schema.get("properties", {})))

    partials: list[dict] = []
    for idx, chunk in enumerate(chunks):
        log.debug("extracting chunk %d/%d (%d chars)", idx + 1, len(chunks), len(chunk))
        partials.append(_extract_from_chunk(schema, chunk, url))

    if not partials:
        return "Extraction failed: no chunks could be processed."

    merged = _merge_extractions(schema, partials)

    # If chunking happened, array fields are a union across chunks and may
    # have over-collected. Run a cleanup pass per array field to prune the
    # union back to genuine matches. Skipped when there was only one chunk
    # (no union to clean) so single-page extracts pay nothing extra.
    if len(chunks) > 1:
        props = schema.get("properties", {})
        for field, spec in props.items():
            if spec.get("type") == "array" and isinstance(merged.get(field), list):
                before = merged[field]
                merged[field] = _clean_array_field(
                    field, spec.get("description", ""), before
                )
                if merged[field] != before:
                    log.debug("cleaned array %r: %d -> %d items",
                              field, len(before), len(merged[field]))

    return json.dumps(merged, indent=2)


def _summarize_chunk(chunk: str, question: str) -> str:
    """Summarize a single chunk, optionally focused by a question."""
    focus = (
        f"Focus on anything relevant to this question: {question}\n\n"
        if question else ""
    )
    prompt = (
        "Summarize the following page content concisely and factually. "
        "Keep concrete facts, names, numbers and dates. Do not add "
        "information that is not present. Ignore site navigation, search "
        "boxes, donation or fundraising banners, cookie notices, login "
        "prompts, and other page chrome; summarize only the article "
        "content itself. Do not preface with 'This page'.\n\n"
        f"{focus}PAGE CONTENT:\n{chunk}"
    )
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": MODEL_TEMPERATURE},
        think=False,
    )
    return resp["message"]["content"].strip()


def _refine_summary(running: str, chunk: str, question: str) -> str:
    """Refine an existing running summary with the content of a new chunk.
    This is the stateful strategy: the running summary IS the memory of
    what has been read so far."""
    focus = (
        f"Keep the summary focused on this question: {question}\n\n"
        if question else ""
    )
    prompt = (
        "You are refining a running summary of a long page as you read it "
        "in pieces. Below is the summary so far, then the next piece of "
        "the page. Produce an updated summary that folds in any new facts "
        "from the next piece. Keep it concise and factual. Ignore site "
        "navigation, donation or fundraising banners, cookie notices, and "
        "other page chrome; summarize only the article content. Do not "
        "drop important facts that were already in the summary. Do not "
        "invent anything.\n\n"
        f"{focus}SUMMARY SO FAR:\n{running}\n\n"
        f"NEXT PIECE OF THE PAGE:\n{chunk}"
    )
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": MODEL_TEMPERATURE},
        think=False,
    )
    return resp["message"]["content"].strip()


def _reduce_summaries(partials: list[str], question: str) -> str:
    """Combine independent per-chunk summaries into one (the reduce step
    of map-reduce). This is a DEDICATED prompt, not the per-chunk
    summarizer reused: because it sees all partials at once, it is the
    right place to discard the ones that only describe page chrome (a
    navigation block, a fundraising banner, a footer of category links),
    which is what keeps one junk chunk from polluting the final result."""
    focus = (
        f"Keep the summary focused on this question: {question}\n\n"
        if question else ""
    )
    joined = "\n\n".join(f"[Part {i + 1}]\n{s}" for i, s in enumerate(partials))
    prompt = (
        "Below are partial summaries of consecutive pieces of one web "
        "page. Combine them into a single concise, factual summary of the "
        "whole page. Some partials may only describe page chrome (site "
        "navigation, fundraising or donation banners, cookie notices, "
        "lists of category links, footers); discard those entirely and "
        "summarize only the actual article content. Keep concrete facts, "
        "names, numbers and dates. Do not invent anything. Do not refer to "
        "'parts' or 'summaries' in your answer; write about the page.\n\n"
        f"{focus}PARTIAL SUMMARIES:\n{joined}"
    )
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": MODEL_TEMPERATURE},
        think=False,
    )
    return resp["message"]["content"].strip()


@mcp.tool()
def summarize(url: str, question: str = "", user_id: str = "") -> str:
    """
    Fetch a webpage and return a concise summary of it. The MCP server
    fetches the page, and if it is large, splits it into overlapping
    chunks and combines per-chunk work into one summary (the strategy,
    map-reduce or refine, is set in config), so the whole page is
    summarized rather than a truncated slice.

    Use this tool when the user wants a free-form summary of a page, or
    an answer to an open question about a long page, rather than a fixed
    set of named fields (use `extract` for named fields).

    Args:
        url: The full URL to summarize.
        question: Optional. If set, the summary is focused on answering
            this question rather than being a general overview.
        user_id: Optional, same semantics as for `fetch`.

    Returns:
        A plain-text summary of the page.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: URL must start with http:// or https://; got {url!r}"

    try:
        snapshot = _open_settle_snapshot(url, user_id)
    except Exception as e:
        log.warning("fetch for summarize failed: %s", e)
        return f"Error fetching snapshot: {e}"

    chunks = _chunk_by_lines(snapshot, CHUNK_CHARS, OVERLAP_LINES)
    log.info("summarizing %s: %d chars -> %d chunk(s)",
             url, len(snapshot), len(chunks))

    if not chunks:
        return "Nothing to summarize: the page snapshot was empty."

    # A single chunk needs no combining.
    if len(chunks) == 1:
        return _summarize_chunk(chunks[0], question)

    if SUMMARIZE_STRATEGY == "refine":
        # Refine: summarize the first chunk, then carry a running summary
        # forward, folding each subsequent chunk into it. More coherent,
        # but sequential.
        #
        # Guard against a known failure: a weak model sometimes responds
        # to a low-content chunk (a page's footer of category links, say)
        # by DESCRIBING that chunk and discarding the running summary it
        # was given. So if a refine step returns dramatically less than it
        # started with, we treat the chunk as adding nothing and keep the
        # prior summary instead of letting the tail of the page erase it.
        running = _summarize_chunk(chunks[0], question)
        for idx, chunk in enumerate(chunks[1:], start=2):
            log.debug("refining summary with chunk %d/%d", idx, len(chunks))
            refined = _refine_summary(running, chunk, question)
            if len(refined) < 0.5 * len(running):
                log.debug("refine step %d shrank summary %d -> %d; keeping prior",
                          idx, len(running), len(refined))
                continue
            running = refined
        return running

    # Default: map-reduce. Summarize each chunk independently (map), then
    # combine with a dedicated reduce step that discards chrome partials.
    # No single chunk is the final word, so a junk chunk cannot erase the
    # result, and the map phase does not depend on order.
    partials: list[str] = []
    for idx, chunk in enumerate(chunks):
        log.debug("map-summarizing chunk %d/%d", idx + 1, len(chunks))
        partials.append(_summarize_chunk(chunk, question))
    log.debug("reducing %d partial summaries", len(partials))
    return _reduce_summaries(partials, question)


def chat():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    chat()

# Stage 1 — camofox-browser and SearXNG in Docker

This stage has no Python that the article tells you to write. It stands
up two services that the agent will talk to in later stages:

- **camofox-browser** on port 9377 (the stealth browser for `fetch`)
- **SearXNG** on port 8888 (the search engine for `search`)

It also gives you two probe scripts to verify camofox before we wrap it
in MCP.

> If you already have SearXNG from Part 3 of this series running on a
> different port, stop it first (`docker stop searxng`) before bringing
> up the compose here. Part 4 is designed to be standalone, with its
> own SearXNG on port 8888.

## Build the camofox image

The upstream camofox-browser repo does not ship a Docker Hub image, and
the default `Dockerfile` requires pre-downloaded binaries in `dist/`.
Use `Dockerfile.ci` instead — it downloads everything at build time.

```bash
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser
docker build -f Dockerfile.ci -t camofox-browser:latest .
cd ..
```

Expect 5-10 minutes for the first build (the Camoufox binary is ~300 MB).

## Run both services

From this directory (`stage1/`):

```bash
docker compose up -d
docker compose logs -f
```

Wait for `server started` from the camofox container and a SearXNG
startup banner. Then Ctrl+C out of the log view.

The compose file mounts `./searxng/settings.yml` into the SearXNG
container, so the JSON API is enabled and the rate limiter is off
from the first run. No manual editing needed.

## Verify

Quick smoke checks:

```bash
curl http://localhost:9377/health
curl "http://localhost:8888/search?q=ollama&format=json" | head -c 500
```

Both should return JSON. If SearXNG returns HTML instead of JSON, the
settings bind-mount did not apply — check that `./searxng/settings.yml`
exists.

Two probe scripts also ship with this stage to exercise camofox end-to-end:

```bash
pip install httpx
python test_camofox.py         # single-URL end-to-end smoke test
python test_camofox_multi.py   # multi-URL test against 8 real sites
```

Both should report all checks passing.

## Tear down

```bash
docker compose down
```

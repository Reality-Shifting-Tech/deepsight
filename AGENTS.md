# AGENTS.md

Guidance for AI agents working in this repository. Human-facing contribution
rules live in [CONTRIBUTING.md](CONTRIBUTING.md); this file is the operational
quick-reference. Where the two overlap, CONTRIBUTING wins.

## What this is

DeepSight is an MIT-licensed vision toolkit that turns any text-only LLM into
a multimodal desktop agent. Works on **macOS** (Apple Vision, zero tokens)
and **Windows** (PIL + optional Tesseract OCR). Provides **16 tools** across
four layers: vision analysis (look, ocr, zoom, count, locate), live screen
capture (capture, watch), web grounding (ground), and desktop automation
(click, type, key, scroll, open, focus, apps, window).

## Toolchain

- Python >= 3.11 (3.12 also supported and tested in CI).
- `uv` for environment and dependency management.
- macOS + Xcode SDK to compile `vision_eyes` from `scripts/vision_eyes.swift`
  (the SDK path is pinned in the Makefile).
- Unit tests must not call external model endpoints — inject fakes.

## Commands

Run from the repo root unless noted.

```bash
uv sync
uv pip install -e '.[dev]'

make lint        # ruff check . (zero-warning policy)
make typecheck   # mypy src
make test        # pytest (testpaths: tests)
make format      # ruff format + ruff check --fix
make build-eyes  # compile scripts/vision_eyes.swift
```

Full pre-push gate: `make all` (lint + typecheck + test).

## Layout

```
src/deepsight/         Core package
  __main__.py          CLI: deepsight describe / doctor
  backends.py          NativeVisionBackend, ReasoningBackend, SearchBackend,
                       ComputerUseBackend, VisionResult, ReasoningResult,
                       SearchResult, ToolCall
  config.py            pydantic-settings with DEEPSIGHT_* prefix
  orchestrator.py      Tool loop, ACTION_TOOL_DEFINITIONS, Orchestrator
  perception.py        Vision tools (TOOL_DEFINITIONS), Perception class,
                       sketch, dhash, cross-capture memory
  cache.py             Content-addressed perception cache
bench/                 Cloud-streaming benchmark harness
docs/                  Architecture docs, images
  images/              Hero, workflow, and demo screenshots for README
tests/                 pytest suite
scripts/              vision_eyes.swift — native Apple Vision binary source
```

## Tools

### Vision tools (defined in `perception.py:TOOL_DEFINITIONS`)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `look` | x, y, w, h | Describe a region of the image |
| `ocr` | x, y, w, h | Transcribe text in a region |
| `zoom` | x, y, w, h | Upscale and inspect a region |
| `count` | what, [x, y, w, h] | Count objects matching a description |
| `locate` | what, [x, y, w, h] | Find object by description, return bbox |
| `capture` | [region] | Screenshot + full vision analysis |
| `watch` | [seconds, interval, until] | Temporal screen monitoring |
| `ground` | what, [query] | Web search + page fetch for verification |

### Action tools (defined in `orchestrator.py:ACTION_TOOL_DEFINITIONS`)

| Tool | Parameters | Description |
|------|-----------|-------------|
| `click` | x, y | Click at %-coordinate |
| `type` | text | Type into focused field |
| `key` | keys | Keyboard shortcut |
| `scroll` | [direction, clicks] | Scroll active window |
| `open` | name | Launch or activate an app |
| `focus` | window | Focus window by title substring |
| `apps` | *(none)* | List running apps + windows |
| `window` | [name, x, y, w, h] | Resize/reposition window |

Routing: action tools are dispatched to `ComputerUseBackend.execute()` in the
orchestrator loop; vision tools go to `Perception.execute()`. The
`_tool_defs()` function merges both into a single tool list.

## Configuration

Runtime configuration is via environment variables (see README). Key ones:

- `DEEPSIGHT_VISION_BIN` — path to compiled vision_eyes binary (required)
- `DEEPSIGHT_REASONING_API_KEY` — for the reasoning loop (optional)
- `DEEPSIGHT_SEARCH_API_KEY` — Brave Search key for grounding (optional)

## Conventions (enforced in review/CI)

- Conventional Commits: `<type>(<scope>): <imperative summary>`; types
  `feat|fix|chore|docs|refactor|test|ci|build|perf`.
- Ruff clean with zero warnings (select `E,F,I,UP,B,SIM`, line length 100).
- mypy clean over `src/` (`disallow_untyped_defs`, `warn_unused_ignores`).
- New behavior ships with tests. Tests must not require live model endpoints;
  inject fakes or use mocked HTTP backends.
- Style bar is "edited, not generated": no narrating comments (comment the
  why or nothing), no dead code, no speculative abstractions beyond what the
  current milestone requires, reuse existing vocabulary.
- One logical change per PR; keep diffs reviewable.
- Device-native vision is a hard requirement: the eyes never call the network
  or require a GPU. Action tools are local-only. Optional features (web
  grounding, reasoning backend) are gated by env vars and degrade gracefully.

## Working agreement for agents

- Never commit or push unless explicitly asked.
- Benchmark runs hit live external endpoints and spend tokens/API credits.
  Do not run `make bench` casually; it requires explicit endpoint/model/API
  key environment variables and is intended for the bench owner.
- API keys are read from the environment only. Never write a secret into a
  file that could be committed, and never log request bodies that contain
  user images or API keys.
- `bench/` and `src/` may be actively owned by other agents during the
  initial build; coordinate through the repo root before restructuring either.
- When adding a new tool: add to the appropriate definition list
  (TOOL_DEFINITIONS in perception.py or ACTION_TOOL_DEFINITIONS in
  orchestrator.py), implement the handler method, route in execute(), add
  tests, update this file and README.

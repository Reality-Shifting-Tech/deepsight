# AGENTS.md

Guidance for AI agents working in this repository. Human-facing contribution
rules live in [CONTRIBUTING.md](CONTRIBUTING.md); this file is the operational
quick-reference. Where the two overlap, CONTRIBUTING wins.

## What this is

DeepSight is an MIT-licensed, device-native vision toolkit. `deepsight
describe` describes any image using the OS's own vision frameworks (Apple
Vision on macOS): OCR, scene classification, saliency, face/human/rectangle
detection. No server, no network, zero tokens. Python 3.11+, setuptools src
layout. The compiled `vision_eyes` binary is the eyes; a small CLI
(`__main__.py`) shells it directly. A vision-session loop (orchestrator +
perception) is available as a library for text-only LLMs, with an optional
reasoning backend (DeepSeek default). Currently at 0.1.0 (initial release).

**Hard rule: vision is device-native. Never introduce a server, an HTTP
endpoint, or a remote VLM for vision.** No ports, no `serve` command, no
Ollama/OpenAI-compatible vision backends. The device's own frameworks are the
product.

## Toolchain

- Python >= 3.11 (3.12 is also supported and tested in CI).
- `uv` for environment and dependency management.
- macOS + Xcode SDK to compile `vision_eyes` from
  `scripts/vision_eyes.swift` (see docs/architecture.md for the command).
- No live infrastructure is required to run the test suite; unit tests must
  not call external model endpoints.

## Commands

Run from the repo root unless noted.

```bash
uv venv
uv pip install -e '.[dev]'

make lint        # ruff check . (zero-warning policy)
make typecheck   # mypy src
make test        # pytest (testpaths: tests)
make format      # ruff format + ruff check --fix
make bench       # benchmark comparison of baseline modes
```

Full pre-push gate: `make all` (lint + typecheck + test).

## Layout

```
src/deepsight/     CLI + vision session loop, tool protocol, backends,
                   perception cache
  __main__.py      CLI entry point (deepsight describe / doctor)
bench/             Cloud-streaming benchmark harness (bench/harness.py) and
                   the BENCHES registry (ChartQA, MathVista, OCRBench-v2)
docs/              architecture.md, benchmarks.md
tests/             pytest suite (testpaths configured in pyproject.toml)
scripts/           vision_eyes.swift - the native Apple Vision binary source
```

The package lives under `src/` (setuptools `packages.find where = ["src"]`);
imports are `deepsight.*`, not `src.deepsight.*`.

## Configuration

Runtime configuration is via environment variables (see README). Key ones for
development:

- `DEEPSIGHT_VISION_BIN` - path to the compiled native vision binary.
- `DEEPSIGHT_REASONING_API_KEY` - optional, only for the vision-session loop.

## Conventions (enforced in review/CI)

- Conventional Commits: `<type>(<scope>): <imperative summary>`; types
  `feat|fix|chore|docs|refactor|test|ci|build|perf`. The changelog is
  maintained from these messages.
- Ruff clean with zero warnings (select `E,F,I,UP,B,SIM`, line length 100).
- mypy clean over `src/` (`disallow_untyped_defs`, `warn_unused_ignores`).
- New behavior ships with tests. Tests must not require live model endpoints;
  inject fakes or use mocked HTTP backends.
- Style bar is "edited, not generated": no narrating comments (comment the
  why or nothing), no dead code, no speculative abstractions beyond what the
  current milestone requires, reuse existing vocabulary.
- One logical change per PR; keep diffs reviewable.
- Never reintroduce the server. If a change adds a port, an HTTP vision
  endpoint, or a remote VLM dependency, it will be rejected in review.

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

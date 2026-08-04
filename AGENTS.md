# AGENTS.md

Guidance for AI agents working in this repository. Human-facing contribution
rules live in [CONTRIBUTING.md](CONTRIBUTING.md); this file is the operational
quick-reference. Where the two overlap, CONTRIBUTING wins.

## What this is

DeepSight is an MIT-licensed, OpenAI-compatible vision proxy that gives
text-only LLMs interactive vision via a vision-session loop. A reasoning model
gets a compact scene sketch, then issues tool calls (`look`, `crop`, `ocr`,
`zoom`, `summarize`) that are answered by targeted vision-model passes, with a
content-addressed perception cache. Python 3.11+, setuptools src layout,
FastAPI server. Default reasoning backend is DeepSeek V4 Flash; default vision
backend is Ollama `minicpm-v`. Currently at 0.1.0 (initial release).

## Toolchain

- Python >= 3.11 (3.12 is also supported and tested in CI).
- `uv` for environment and dependency management.
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
make dev         # uvicorn --reload on :8080
make format      # ruff format + ruff check --fix
make bench       # three-way benchmark comparison
```

Full pre-push gate: `make all` (lint + typecheck + test).

## Layout

```
src/deepsight/     FastAPI app: OpenAI-compatible /v1 server, vision session
                   loop, tool protocol, backends, perception cache
  __main__.py      CLI entry point (deepsight command)
bench/             Cloud-streaming benchmark harness (bench/harness.py) and
                   the BENCHES registry (ChartQA, MathVista, OCRBench-v2)
docs/              architecture.md, benchmarks.md
tests/             pytest suite (testpaths configured in pyproject.toml)
```

The package lives under `src/` (setuptools `packages.find where = ["src"]`);
imports are `deepsight.*`, not `src.deepsight.*`.

## Configuration

Runtime configuration is via environment variables (see README). Key ones for
development:

- `REASONING_API_KEY` - required for real reasoning calls (DeepSeek default).
- `VISION_BASE_URL` - defaults to `http://localhost:11434/v1` (Ollama).
- `PORT` - server port, default 8080.

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

## Streaming status events

`POST /v1/chat/completions` with `"stream": true` emits live progress chunks
before the final answer, so clients can show feedback instead of silence
during the ~1 min vision session.

Each progress chunk carries a `delta.status` string (standard OpenAI SSE
shape, unknown field ignored by strict clients):

- `👁️ viewing image...` - image loaded, sketch in progress
- `✏️ sketching scene...` - vision pass building the scene sketch
- `🔍 looking (<tool>)...` - one `look`/`ocr`/`zoom`/... tool round
- `✅ answering...` - reasoning composing the final answer

The final `delta.content` chunk carries the answer. Non-stream responses
return the same milestones via the `Orchestrator.run(..., on_event=cb)`
callback. Hermes integration: display `delta.status` chunks as transient
status lines and treat `delta.content` as the real answer.

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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (nothing yet; see the 0.1.0 section for the current feature set)

## [0.1.0] - 2026-08-03

### Added

Initial release. DeepSight is an MIT-licensed, OpenAI-compatible vision proxy
that gives text-only LLMs interactive vision through a vision-session loop.

Phase 1 - Core (the vision session loop):

- FastAPI app with OpenAI-compatible `POST /v1/chat/completions` and
  `GET /v1/models`, including streaming pass-through.
- `perception.py`: `sketch()` - a single vision-model pass producing a compact
  JSON scene inventory (objects, text regions, layout, palette).
- Tool protocol: `look`, `crop`, `ocr`, `zoom`, `summarize`, each answered by
  a targeted vision-backend pass.
- Text-fallback protocol (`[LOOK ...]` / `[OCR ...]` markers) for tool-less
  reasoning models.
- Reasoning backends: DeepSeek preset (default) and generic
  OpenAI-compatible base URL.
- Vision backends: Ollama (`minicpm-v` default) and OpenAI-compatible VLM.
- Perception cache: content-addressed by image hash and crop hash, with TTL.
- Concurrent tool calls (all pending looks fire at once).

Phase 2 - Viewer UI (the differentiator):

- React + Vite app at `ui/`, served by FastAPI at `/ui` (static build).
- Chat pane with streaming answers.
- Image input via paste, drag-drop, file picker, and URL.
- Inline thumbnails, lightbox, zoom, pan, and region-select; region-select
  emits a `look` tool call.

Phase 3 - Packaging:

- `pip install .` and `uv tool install` verified on a clean environment.
- Optional Dockerfile and compose example.
- `deepsight doctor` connectivity check for reasoning and vision backends.
- LICENSE (MIT), CONTRIBUTING, AGENTS, and CI (lint + typecheck + tests).

Phase 4 - Benchmarks:

- Cloud-streaming harness (`bench/harness.py`) with zero local dataset
  downloads, streaming rows from the Hugging Face datasets-server API:
  ChartQA, MathVista, OCRBench-v2.
- Efficiency metrics: tokens per correct answer, latency per question,
  cache-hit rate, published in the README and docs/benchmarks.md.
- PyPI publish prep and v0.1.0 tag.

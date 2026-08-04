# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Reasoning-model tool rounds no longer truncate.** `tool_round_max_tokens`
  (orchestrator) and `DEEPSIGHT_REASONING_TOOL_ROUND_MAX_TOKENS` (settings)
  both default to 1024 instead of 128/256. Reasoning models like
  `deepseek-v4-flash` consume thinking tokens inside the same generation
  budget; the old caps truncated the model mid-thought before it could emit a
  tool call, forcing repeated "widening output budget" retries and
  look-rounds that never answered.

### Changed

- **Natural conversational answers.** The vision-session loop no longer forces
  an ultra-terse "caveman" answer style. The orchestrator prompt now asks the
  reasoning model to respond like a natural multimodal assistant: lead with the
  interpretation, hedge honestly when the native vision layer is coarse, and
  describe the image in fluent prose without exposing the sketch/tool pipeline.
  Truncated mid-answer generations are now retried with the final token budget,
  so longer natural-language answers no longer get clipped.

### Removed

- **Server removed.** The FastAPI/uvicorn OpenAI-compatible server adapter
  (`deepsight serve`, `src/deepsight/server.py`) is gone. Vision is
  device-native only: `deepsight describe` shells the Apple Vision binary
  directly. No ports, no HTTP vision endpoint, no Ollama/OpenAI-compatible
  vision backends, no launchd service. The vision-session loop remains as a
  library for text-only LLMs, with the native binary as its only eyes.
- Dropped `fastapi`, `uvicorn`, and `python-multipart` dependencies.
- Removed `make dev` and `make bench-deepsight` targets.

## [0.1.0] - 2026-08-03

### Added

Initial release. DeepSight is an MIT-licensed, device-native vision toolkit:
`deepsight describe` uses the OS's own vision frameworks (Apple Vision on
macOS) for OCR, scene classification, saliency, and face/human/rectangle
detection. No server, zero tokens.

Phase 1 - Core (the native eyes + vision session loop):

- `vision_eyes` - compiled Apple Vision binary (OCR, scene classification,
  saliency, face/human/rectangle detection), on-device and free.
- `deepsight describe` - CLI that shells the binary directly.
- `perception.py`: `sketch()` - a single native vision pass producing a
  compact JSON scene inventory (objects, text regions, layout, palette).
- Tool protocol: `look`, `crop`, `ocr`, `zoom`, `summarize`, each answered by
  a targeted native vision pass.
- Text-fallback protocol (`[LOOK ...]` / `[OCR ...]` markers) for tool-less
  reasoning models.
- Reasoning backend: DeepSeek preset (default) and generic OpenAI-compatible
  base URL, used by the optional loop.
- Perception cache: content-addressed by image hash and crop hash, with TTL.
- Concurrent tool calls (all pending looks fire at once).

Phase 2 - Packaging:

- `pip install .` and `uv tool install` verified on a clean environment.
- `deepsight doctor` connectivity check for the native vision binary and
  reasoning endpoint.
- LICENSE (MIT), CONTRIBUTING, AGENTS, and CI (lint + typecheck + tests).

Phase 3 - Benchmarks:

- Cloud-streaming harness (`bench/harness.py`) with zero local dataset
  downloads, streaming rows from the Hugging Face datasets-server API:
  ChartQA, MathVista, OCRBench-v2.
- Efficiency metrics: tokens per correct answer, latency per question,
  cache-hit rate, published in the README and docs/benchmarks.md.
- PyPI publish prep and v0.1.0 tag.

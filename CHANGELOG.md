# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Eyes regression eval (tier 0).** `eval/` ships a zero-token scoring harness: `run_eval.py` runs the compiled `vision_eyes` binary over a manifest of 19 images (4 user-validated golden, 5 deterministic synthetic, 10 public-domain Commons photos) and scores OCR, scene, sports, and count signals against expected ground truth. `make eval-eyes` runs it; `make eval-synth` and `make eval-fetch` regenerate the synthetic and public fixtures (provenance tracked in `eval/images/public/.sources.json`). Results land in `eval/results/` (gitignored) as timestamped JSON. Baseline: 17/19 images, 43/45 checks (95.6%). Two deliberate failing anchors document real gaps: tennis rackets missed on B&W photos, and soccer stadiums false-positive as `ice hockey` via the arena mapping. 11 new unit tests cover the parser and every check type; the manifest schema is asserted in CI.
- **Reasoning-loop eval (tier 1).** `eval/run_loop_eval.py` runs every manifest image through the full vision-session loop with a blind prompt (the model never sees ground truth or expectations) and scores the final description against per-entry `loop` blocks: `required` tokens must appear, `forbidden` tokens must not (hallucination/contradiction guard). `make eval-loop` runs it; results land in `eval/results/loop/` (gitignored). `eval/nightly.py` runs eyes + loop together and reports a delta report only when the baseline moves (silent when green), wired to a nightly cron.
- **macOS CI eval gate (tier 2).** `.github/workflows/eval.yml` compiles `scripts/vision_eyes.swift` from repo source (the swift source now ships in-repo for reproducible builds), regenerates synthetic + public fixtures, and runs the eyes eval with the local-only golden photos skipped (`--skip-missing`). Gap-aware exit codes keep CI green while known gaps stay tracked as ⚠️.
- **Eyes v3 sports-equipment signal.** `vision_eyes` now classifies the full frame plus salient-region crops (2x upscaled, subdivided into a 2x2 grid) through a sport taxonomy (baseball, basketball, american football, soccer, ice hockey, tennis, golf, volleyball) and emits a `sports:` line. The sketch surfaces it as `Sports: baseball(0.57), ice hockey(0.54), ...` right after the scene label. The orchestrator prompt instructs the model that two or more distinct sports means an athletes' group photo (typically one city's teams), not a single-team crowd, and to trust it over OCR team names. On the 1980 City of Champions benchmark the blind pass now reads the group as athletes from different sports instead of "Sixers fans in Louisville", and the session is cheaper (2 rounds vs 3).

### Fixed

- **Soccer stadium false positive.** Dropped `arena` from the sports taxonomy's ice-hockey triggers (stadiums were mapping to `ice hockey(0.85)`) and added the literal `soccer` classifier label to the soccer mapping. The soccer stadium now emits `soccer(0.17)`; hockey detection is unaffected (still `rink(0.94)`). The reasoning loop keeps a propagation anchor (weak soccer signal + `arena` scene label) tracked as a known gap.
- **Surfing not recognized on the sports line.** The Apple Vision classifier emits `surfboard` for surfer photos, but it was unmapped, so the `sports:` line only reported generic `sports equipment`. Added `surfboard` to the surfing mapping; the surfer photo now emits `sports: sports equipment(0.82), surfing(0.81)` and the manifest expectation moved from the generic fallback to a real `surfing` signal. Closes the public-surfing gap at eyes level.
- **Loop scoring false alarms.** `score_loop` now treats negated ("no baseball equipment is shown") and hedged ("a faint baseball hint") mentions as non-assertions, so forbidden guards only fail on real claims. New `loop_any` check accepts paraphrase alternatives ("surfing" / "surfer" / "surfboard"). Unit tests cover both. This removes the two stochastic flake sources from the nightly watcher: the SGA photo's correct denial of baseball, and surfing answers that say "surfer" instead of "surfing".
- **Gap-aware eval exits.** `run_eval.py` treats `gap: true` entries as accepted known gaps (⚠️, non-blocking) and reports a gap entry passing as a 🎉 GAP CLOSED event; `--skip-missing` marks absent images as skipped instead of failed. `status_for`/`exit_code_for` are pure functions covered by unit tests, so CI can run with known gaps without going red.
- **Reasoning-model tool rounds no longer truncate.** `tool_round_max_tokens`
  (orchestrator) and `DEEPSIGHT_REASONING_TOOL_ROUND_MAX_TOKENS` (settings)
  both default to 1024 instead of 128/256. Reasoning models like
  `deepseek-v4-flash` consume thinking tokens inside the same generation
  budget; the old caps truncated the model mid-thought before it could emit a
  tool call, forcing repeated "widening output budget" retries and
  look-rounds that never answered.

### Changed

- **Production-ready README.** Replaced the draft badges (PyPI links for an unpublished package, a CI badge pointing at a nonexistent workflow) with real ones (license, macOS 14+, Python 3.11+, the actual `eval.yml` workflow). Added a generated terminal-demo image and an architecture diagram (`docs/images/`, produced by `docs/make_images.py`), verified-copy library snippets, a requirements table, full environment-variable reference, regression-eval scoreboard, and a troubleshooting table.

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

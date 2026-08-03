# DeepSight — Implementation Plan (v2: model-agnostic)

Image input + image viewer for **any** text-only LLM, tuned for DeepSeek
V4 Flash. Drop-in OpenAI-compatible vision proxy + first-class viewer UI.
Open source (MIT).

## Context (verified live 2026-08-03)

- `deepseek-v4-flash` / `deepseek-v4-pro` exist and serve.
- Both text-only: chat rejects `image_url`; Responses/Anthropic accept schema
  but model cannot see.
- **v4-flash supports native tool calling** (finish_reason `tool_calls`,
  correct args) and streaming (43 chunks, clean `[DONE]`). Verified by probe
  script `/tmp/probe_deepseek.py`.
- So: interactive vision loop is fully viable on the flagship target.

## Architecture v2 — vision as an API contract

```
Any client (OpenWebUI, LibreChat, curl, app)
        │  POST /v1/chat/completions (image_url content)
        ▼
deepsight server (FastAPI)
        ├─ orchestrator: vision session loop
        │    1. sketch(image) → compact inventory (objects/layout/text/colors)
        │    2. inject sketch + vision tool descriptions into context
        │    3. reasoning model emits look/crop/ocr/zoom tool calls (or text markers)
        │    4. bridge answers each call with a TARGETED vision-model pass
        │    5. loop until model satisfied → final answer
        ├─ vision backend  → Ollama minicpm-v (default, local) or any VLM
        ├─ reasoning backend → DeepSeek (default) or any OpenAI/Anthropic URL
        └─ perception cache → content-addressed, crop-hashed, zero re-asks
        ▼
deepsight UI (React) — /ui route: paste/drop, inline viewer, region-select
```

### Why this beats the one-shot description

| | One-shot desc (visionbridge) | Interactive session (deepsight) |
|---|---|---|
| First pass | 300-500 tokens scene novel | ~60-120 token sketch |
| "Error in bottom-right?" | model guesses / re-reads all | targeted crop → exact |
| 4K UI screenshot | 500 tokens, still vague | sketch + on-demand zooms |
| Cost per question | fixed per image | proportional to looking |
| Quality ceiling | description quality | model's own curiosity |

## Phases

### Phase 1 — Core: vision session loop (build the engine)
- [ ] Scaffold FastAPI app `deepsight/`, OpenAI-compatible `/v1/chat/completions`
      + `/v1/models`, streaming pass-through.
- [ ] `perception.py`: sketch() — one vision-model pass producing compact JSON
      inventory (objects, text regions, layout, palette).
- [ ] Tool protocol: `look(x,y,w,h)`, `crop(x,y,w,h)`, `ocr(region?)`,
      `zoom(x,y,scale)`, `summarize()`. Vision calls go to the vision backend.
- [ ] Text-fallback protocol for tool-less models: `[LOOK ...]` / `[OCR ...]`
      markers parsed from plain content; same handlers.
- [ ] Reasoning backends: DeepSeek preset (default), OpenAI-compatible
      (generic base_url+key), Anthropic-compatible adapter.
- [ ] Vision backends: Ollama (minicpm-v default) + OpenAI-compatible VLM.
- [ ] Perception cache: content-addressed by image hash + crop hash; TTL.
- [ ] Concurrent tool calls (fire all pending looks at once).
- [ ] Verify: curl a UI screenshot → "what's the error message?" → correct,
      and the token spend is < one-shot baseline.

### Phase 2 — Viewer UI (the differentiator)
- [ ] React + Vite app at `ui/`, served by FastAPI at `/ui` (static build).
- [ ] Chat pane: OpenAI-compatible messages, streaming answers.
- [ ] Image input: paste (clipboard), drag-drop, file picker, URL.
- [ ] Viewer: inline thumbnails → lightbox, zoom, pan, region-select box.
- [ ] Region-select → emits a `look` tool call ("ask about this part").
- [ ] Image references persisted in message history (store + resend).

### Phase 3 — Packaging + docs (easy for anyone)
- [ ] `pip install .` / `uv tool install` verified on clean venv.
- [ ] Optional Dockerfile + compose example (not required path).
- [ ] README: 5-minute setup (DeepSeek + Ollama), client configs
      (OpenWebUI, LibreChat, curl, openai/anthropic SDK), env reference.
- [ ] `deepsight doctor` — check reasoning/vision connectivity.
- [ ] LICENSE (MIT), CONTRIBUTING, CI (lint + pytest).

### Phase 4 — Benchmarks (prove "leading in efficiency + capability")
Cloud-streaming harness — NO local dataset downloads (user directive).
Bench rows stream from HF datasets-server `/rows` API; images via
signed cached-asset URLs; scoring local. Verified streamable 2026-08-03:
- [x] **ChartQA** (`HuggingFaceM4/ChartQA` test) — chart reasoning. label list.
- [x] **MathVista** (`AI4Math/MathVista` testmini) — visual math. answer +
      answer_type (free_form/float/multi). Use `decoded_image` src.
- [x] **OCRBench-v2** (`lmms-lab/OCRBench-v2` test) — OCR + UI/screenshot
      QA (rico/APP agent rows = DeepSight's exact sweet spot).
- [ ] MMMU — public but datasets-server parquet-path bug (500); retry via
      `MMMU/MMMU` config=discipline or skip (not required for v0.1).
- [ ] Harness `bench/harness.py`: stream N rows → fetch image → POST to
      target endpoint → score (normalized exact-match; float tolerance for
      MathVista) → record tokens+latency per question.
- [ ] Baselines: (a) direct VLM = sensenova-6.7-flash-lite (free,
      multimodal, verified) = capability ceiling; (b) one-shot description
      bridge (visionbridge-style) = category baseline; (c) DeepSight.
- [ ] Efficiency metrics (the "leading" claim): tokens per CORRECT answer,
      latency per question, cache-hit rate. Publish table in README.
- [ ] PyPI publish prep (`python -m build`, twine check).
- [ ] README comparison table, viewer screenshots, demo GIF.
- [ ] Tag v0.1.0.

## Verification

- Phase 1: pytest green; curl with real PNG → correct answer about a
  specific region; tool calls + text-marker paths both work; stream works.
- Phase 2: /ui serves; paste → visible; region-select → `look` fires (logs);
  answer references the crop.
- Phase 3: clean-venv install; `deepsight doctor` passes; README reproducible
  on a fresh machine/docker.
- Phase 4: PyPI test upload; `pip install deepsight` from index.

## Risks / notes

- v4-flash tool calls: verified working; verify multi-turn tool loop (second
  round after tool result) — most likely failure point.
- Ollama minicpm-v: good for screenshots/OCR/UI; swappable to stronger VLM.
- Reasoning token cost: sketch + tool turns > zero; but << one-shot desc for
  complex images. Perception cache kills repeats.
- Models without tools: text-marker protocol is the safety net; test with a
  small local model (e.g. qwen2.5:3b text-only) in CI.
- Keep upstream ideas attributed: README credits visionbridge + minicpm-mcp.

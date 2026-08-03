# Architecture

DeepSight is an OpenAI-compatible vision proxy that gives text-only LLMs
interactive vision. This document explains the vision-session loop, the API
surface, and why the design beats the one-shot description baseline.

## The problem

Text-only reasoning models (DeepSeek V4 Flash and most cheap/fast models)
reject or ignore `image_url` content. The usual workaround is a one-shot
description bridge: a vision model describes the entire image once, in prose,
and the text model answers from that description. That approach has a hard
quality ceiling: the description is written before the question is known, so
the model guesses what matters. Details in one corner of a 4K screenshot are
vague or missing, and the token cost is paid in full for every image, every
time.

## The vision-session loop

Instead of one description, DeepSight runs a loop. The reasoning model is
given a cheap scene sketch plus tool schemas, and it decides what to look at
next. Every look is a small, targeted vision pass over just the region the
model asked about.

```
+----------------------+
|  Any OpenAI client   |  curl, openai SDK, OpenWebUI, LibreChat
+----------------------+
        |
        |  POST /v1/chat/completions  (image_url content)
        v
+----------------------+
|   deepsight server   |  FastAPI, OpenAI-compatible /v1
+----------------------+
        |
        |  vision session loop
        v
+---------------------------------------------------+
| 1. sketch(image)      one VLM pass -> compact JSON |
|                       scene inventory (objects,    |
|                       text regions, layout,        |
|                       palette)                     |
| 2. inject sketch + tool schemas into the context   |
| 3. reasoning model emits look / crop / ocr / zoom  |
|    tool calls (or [LOOK]/[OCR] text markers)       |
| 4. bridge answers each call with a TARGETED VLM    |
|    pass over just that region                      |
| 5. repeat until the model is satisfied, then       |
|    produce the final answer                        |
+---------------------------------------------------+
        |                           |
        v                           v
+------------------+     +----------------------+
| reasoning backend |     |   vision backend     |
| DeepSeek (default)|     | Ollama minicpm-v     |
| or any OpenAI /   |     | or any VLM           |
| Anthropic URL     |     |                      |
+------------------+     +----------------------+
        |                           |
        +------------+--------------+
                     v
            +------------------+
            | perception cache |
            | content-addressed|
            | by image + crop |
            | hash, TTL       |
            +------------------+
```

### The steps

1. **Sketch.** One vision pass produces a compact JSON inventory of the image:
   objects, text regions, layout, palette. This is 60-120 tokens, not a
   300-500 token prose novel.
2. **Inject.** The sketch and the tool schemas are added to the reasoning
   context. The model now knows what is in the image and what it can ask for.
3. **Reason.** The model emits tool calls (`look`, `crop`, `ocr`, `zoom`,
   `summarize`). Models without tool support use a text-fallback protocol:
   `[LOOK ...]` / `[OCR ...]` markers parsed from plain content and routed to
   the same handlers.
4. **Look.** Each call is answered by a targeted vision pass over exactly the
   requested region. Pending looks fire concurrently.
5. **Repeat.** The loop continues until the model is satisfied, then the final
   answer is produced. `MAX_TOOL_ROUNDS` bounds runaway loops.

### The tool protocol

| Tool | Purpose |
|---|---|
| `look(x, y, w, h)` | Targeted vision pass over a region |
| `crop(x, y, w, h)` | Region crop for closer reading |
| `ocr(region?)` | Text extraction from the image or a region |
| `zoom(x, y, scale)` | Magnify a region before looking |
| `summarize()` | Fresh holistic pass when context is exhausted |

All vision calls go to the vision backend (Ollama `minicpm-v` by default, or
any OpenAI-compatible VLM).

### The perception cache

Every sketch and every crop answer is content-addressed by image hash and crop
hash. Within the TTL (`CACHE_TTL`), asking about the same region twice costs
zero vision tokens. This is what makes follow-up questions on the same image
nearly free.

## API surface

- `POST /v1/chat/completions` - OpenAI-compatible chat completions accepting
  `image_url` content parts, with streaming pass-through.
- `GET /v1/models` - lists the models the server can route to.

Clients do not need changes: any OpenAI-compatible tool works by pointing it
at `http://localhost:8080/v1`.

## Why this beats the one-shot description

| | One-shot description bridge | Interactive session (DeepSight) |
|---|---|---|
| First pass | 300-500 token scene novel | ~60-120 token sketch |
| "Error in bottom-right?" | model guesses or re-reads everything | targeted crop, exact answer |
| 4K UI screenshot | 500 tokens, still vague | sketch + on-demand zooms |
| Cost per question | fixed per image | proportional to looking |
| Quality ceiling | description quality | the model's own curiosity |

The token-efficiency claim is measured, not asserted: see
[docs/benchmarks.md](benchmarks.md) for the harness and metrics.

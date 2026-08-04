# Architecture

DeepSight is a device-native vision toolkit: `deepsight describe` describes
any image using the OS's own vision frameworks. On macOS that is Apple Vision
directly, via the compiled `vision_eyes` binary. No server, no network, no
model downloads, zero tokens.

## The device is the product

The vision capability is the device's own framework. The CLI shells a small
compiled binary (`vision_eyes`, source in `scripts/vision_eyes.swift`) which
runs Apple Vision requests on-device:

- OCR text extraction (`VNRecognizeTextRequest`)
- Scene classification (`VNClassifyImageRequest`, 1000+ classes)
- Attention/object saliency (`VNGenerateAttentionBasedSaliencyImageRequest`)
- Face detection (`VNDetectFaceRectanglesRequest`)
- Human detection (`VNDetectHumanRectanglesRequest`)
- Rectangle detection (`VNDetectRectanglesRequest`)

Everything runs in milliseconds, on the local CPU/Neural Engine, with zero
token spend and zero model downloads. The binary is the only vision backend
the project ships.

```
+----------------------+
|  deepsight describe  |  shells the OS vision binary directly
+----------------------+
        |
        v
+---------------------------------------------------+
| Apple Vision (on-device, zero tokens, zero cost)  |
|  - OCR text extraction                            |
|  - scene classification (1000+ classes)           |
|  - attention + object saliency maps               |
|  - face / human / rectangle detection             |
+---------------------------------------------------+
```

## Building the binary

```bash
env SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk \
    swiftc -target arm64-apple-macos14 vision_eyes.swift -o vision_eyes
```

Point `DEEPSIGHT_VISION_BIN` at the compiled binary (or put it on `PATH`).

## The vision-session loop (library, optional)

The same native eyes power an optional vision-session loop for text-only
LLMs, available as a Python library (no server involved):

1. **Sketch.** One native vision pass produces a compact JSON inventory of
   the image: objects, text regions, layout, palette.
2. **Inject.** The sketch and the tool schemas are added to the reasoning
   context. The model now knows what is in the image and what it can ask for.
3. **Reason.** The model emits tool calls (`look`, `crop`, `ocr`, `zoom`,
   `summarize`). Models without tool support use a text-fallback protocol:
   `[LOOK ...]` / `[OCR ...]` markers parsed from plain content and routed to
   the same handlers.
4. **Look.** Each call is answered by a targeted native vision pass over
   exactly the requested region. Pending looks fire concurrently.
5. **Repeat.** The loop continues until the model is satisfied, then the final
   answer is produced. `DEEPSIGHT_MAX_LOOK_ROUNDS` bounds runaway loops.

### The tool protocol

| Tool | Purpose |
|---|---|
| `look(x, y, w, h)` | Targeted vision pass over a region |
| `crop(x, y, w, h)` | Region crop for closer reading |
| `ocr(region?)` | Text extraction from the image or a region |
| `zoom(x, y, scale)` | Magnify a region before looking |
| `summarize()` | Fresh holistic pass when context is exhausted |

All vision calls go through `NativeVisionBackend` (the compiled binary). The
reasoning backend is an optional OpenAI-compatible chat endpoint (DeepSeek by
default) that the library talks to as a client; it is never a server that
serves vision.

### The perception cache

Every sketch and every crop answer is content-addressed by image hash and crop
hash. Within the TTL (`DEEPSIGHT_CACHE_TTL_SECONDS`), asking about the same
region twice costs zero vision tokens. This is what makes follow-up questions
on the same image nearly free.

## Why native eyes beat the VLM server

| | VLM server (removed) | Native eyes (DeepSight) |
|---|---|---|
| Setup | install + run Ollama/VLM server, manage RAM | compile once, done |
| Cost | tokens per call | zero tokens, zero API key |
| Speed | 5-30s per pass | milliseconds per pass |
| Privacy | image leaves the device | image never leaves the device |
| Dependencies | torch/transformers-sized | just the OS vision framework |

The device-native design has no server to deploy, no port to expose, no
endpoint to point clients at, and no token bill. `deepsight describe` is the
product; the loop is an optional library for text-only LLMs that need
interactive vision. The token-efficiency claim is measured in
[docs/benchmarks.md](benchmarks.md).

# Eyes v3 — Sports-equipment object signal

Status: proposed. Target: next vision_eyes iteration.

## Problem

The 1980 Philadelphia photo (`img_38e2f9b7f1ae.jpg`) is the benchmark. Eyes v2
caught OCR `SIXERS` ×2, `6` (Dr. J), `45` (Tug McGraw), `OUISVILLE`
(`LOUISVILLE` = Slugger bat), four humans, four low-quality faces. But nothing
in the sketch signals *bat*, *goalie pads*, or *helmet*, so the reasoning model
could not infer "four different sports" and blind mode read
"Sixers fans in Louisville".

Gemini-grade answers need a **multi-sport signal**: a line in the sketch that
names the sports present in the image, independent of jersey text.

## Goal

Emit a `sports:` line in the perception sketch listing detected sport concepts
with confidences, using only on-device Apple Vision. Zero model downloads, zero
network — consistent with the repo's device-native hard rule.

## Approach

1. **Reuse objectness saliency.** v2 already computes
   `VNGenerateObjectnessBasedSaliencyImageRequest`; its `salientObjects`
   (`VNRectangleObservation`) give candidate regions. No new detection pass.
2. **Classify each salient region** with `VNClassifyImageRequest` using
   `request.regionOfInterest` (normalized rect from the observation) on a fresh
   handler per region. Fall back to the full frame when no salient objects
   exist.
3. **Map identifiers → sport concepts** through a small taxonomy table kept in
   code. Threshold 0.15, keep top 3 per region, dedupe across regions.
4. **Emit** a single sorted `sports:` line.

## Output format

```
sports: baseball(0.71), basketball(0.55), american football(0.40), ice hockey(0.31)
```

Concept set (normalized identifiers):

| Concept | Classifier identifiers that map to it |
| --- | --- |
| baseball | baseball, baseball bat, baseball glove, ball |
| basketball | basketball, sports ball, ball |
| american football | american football, football helmet, football |
| ice hockey | ice hockey, hockey puck, hockey rink |
| soccer | soccer ball, football (verify live) |
| tennis | tennis ball, tennis racket |
| volleyball | volleyball |
| golf | golf ball, golf club |

> Caveat: the built-in classifier's identifier set is a curated ~1k-concept
> taxonomy, not ImageNet verbatim. **Verify live** what identifiers actually
> come back for a baseball bat / goalie pads / helmet before finalizing the
> table. The table is the piece most likely to need tuning.

## Sketch schema addition

`Perception` gains a `sports: list[tuple[str, float]]` field. The existing
`scene:` parse pattern in `backends.py::_parse_stdout` is the template:
same `name(0.00)` tuple format, new `sports:` prefix. Parser unit test with a
fixture line, no live endpoints (AGENTS.md rule).

## Prompt change

`orchestrator.py` SYSTEM_PROMPT: add one clause to the sketch inventory —
"sports equipment/activity concepts (e.g. `sports: baseball(0.7)`)".

## Acceptance test

1. **1980 photo:** sketch emits `sports:` with ≥ 2 concepts including
   `baseball` and `basketball` (the bat + Dr. J's ball).
2. **Blind pass flips:** re-run `run_loop_real.py` blind mode (no Gemini
   context) and the answer must not read "Sixers fans in Louisville" — it
   should surface multiple sports and land on a group of athletes across
   sports.
3. Existing 50-test suite stays green; new parser tests added.

## Implementation steps

1. `vision_eyes.swift`: add `sportsPass` — iterate salient boxes (fallback:
   full frame), classify each crop via `regionOfInterest`, map through
   taxonomy, emit `sports:` line. Header comment + usage doc updated to v3.
2. Compile:
   `env SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk swiftc -target arm64-apple-macos14 vision_eyes.swift -o vision_eyes`
   (in `~/.hermes/skills/apple/macos-vision-framework/scripts/`).
3. `backends.py`: parse `sports:` → `Perception.sports`.
4. `orchestrator.py`: prompt clause (above).
5. Tests: parser fixture; keep gates green (ruff / mypy / pytest).
6. Live check on the 1980 photo + blind loop re-run.

## Risks

- **Unknown identifier set:** built-in classifier may not emit
  equipment-specific labels (e.g. "bat" as its own concept). Mitigation:
  verify live first, extend taxonomy table, and accept coarse concepts
  ("ball") as weak signals.
- **Vintage photo noise:** 1980 film grain, low face quality (0.17–0.25),
  small objects. Region crops help; thresholds may need per-region tuning.
- **False positives:** fans, banners, logos in crowd scenes could inject
  sports noise. Mitigation: confidence threshold, top-K cap, and the reasoning
  model weighing sports against OCR/pose evidence.
- **Stretch option (rejected for v3):** a bundled CoreML object-detection
  model (e.g. YOLO-variant) would give real boxes + labels, but adds a model
  artifact and repo weight. Revisit only if the classifier-crop path
  underperforms.

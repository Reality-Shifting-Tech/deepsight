# Eyes v3 — Sports-equipment object signal

Status: **implemented (2026-08-04)**. Shipped with the token-budget fix
(`e9d5667` was the budget; this is the v3 commit). Live results below.

## Problem

The 1980 Philadelphia photo (`img_38e2f9b7f1ae.jpg`) is the benchmark. Eyes v2
caught OCR `SIXERS` ×2, `6` (Dr. J), `45` (Tug McGraw), `OUISVILLE`
(`LOUISVILLE` = Slugger bat), four humans, four low-quality faces. But nothing
in the sketch signaled *bat*, *goalie pads*, or *helmet*, so the reasoning model
could not infer "four different sports" and blind mode read
"Sixers fans in Louisville".

Gemini-grade answers need a **multi-sport signal**: a line in the sketch that
names the sports present in the image, independent of jersey text.

## Goal

Emit a `sports:` line in the perception sketch listing detected sport concepts
with confidences, using only on-device Apple Vision. Zero model downloads, zero
network — consistent with the repo's device-native hard rule.

## Approach (as built)

1. **Reuse objectness saliency.** v2 already computes
   `VNGenerateObjectnessBasedSaliencyImageRequest`; its `salientObjects`
   (`VNRectangleObservation`) give candidate regions. No new detection pass.
2. **Classify full frame + salient region, subdivided into a 2x2 grid.**
   On the benchmark the single salient box covers nearly the whole group
   (501x758 of 791x1024), so a single region crop dilutes equipment signals.
   Each crop is **upscaled 2x** (same trick the v2 OCR pass uses) before
   `VNClassifyImageRequest`. 5 classifications: region + 4 cells.
3. **Map identifiers → sport concepts** through the taxonomy table below,
   aggregated with max confidence per sport. Map threshold 0.10.
4. **Emit** a single sorted `sports:` line (or `sports: none`).

## Taxonomy (verified live)

The classifier's curated taxonomy *does* emit sport identifiers, but not the
ones the first draft guessed. Verified on the benchmark photo:

| Concept | Classifier identifiers that map to it (verified) |
| --- | --- |
| baseball | `baseball`, `baseball bat`, `baseball glove`, **`baseball hat` (0.56 on the cap)** |
| basketball | `basketball`, `sports ball`, `basketball hoop` |
| american football | `american football`, `football helmet`, `football`, **`helmet` (0.30)** |
| ice hockey | `ice hockey`, `hockey`, `hockey puck`, `hockey rink`, `hockey stick`, **`rink`/`arena` (0.50), `ice skates`** |
| soccer | `soccer ball` |
| tennis | `tennis ball`, `tennis racket` |
| volleyball | `volleyball` |
| golf | `golf ball`, `golf club` |
| generic gear | `sports equipment` (self-emits; the bat read as generic gear at 0.27) |

Notes from live tuning:

- `skateboard → skateboarding` was removed: the bat misclassified as
  `skateboard(0.28)` and the fake sport poisoned the line's credibility.
- `football` maps to american football (ImageNet convention), `soccer ball`
  is the soccer signal; soccer and american football don't collide in practice.
- The **2x2 grid is what made it work**: `baseball hat` only appears in one
  cell, `arena`/`rink` only in another. Whole-frame passes miss both.
- `sports raw` debug lines are emitted per classification (tag `full` /
  `region` / `cell`) but are not indented, so the parser never surfaces them.

## Output

```
sports: baseball(0.57), ice hockey(0.54), american football(0.35), sports equipment(0.34)
```

The sketch places `Sports:` immediately after `Scene:` (before OCR text) so the
model reads the multi-sport signal before any jersey text.

## Sketch schema addition

No structured field — `_parse_stdout` gains a `sports:` branch (same
`name(0.00)` tuple format as `scene:`) and appends `Sports: ...` after the
scene line. Parser unit tests cover the happy path and the `none` case; no live
endpoints (AGENTS.md rule).

## Prompt change

`orchestrator.py` SYSTEM_PROMPT: one clause naming the sports line, plus an
override rule: *two or more distinct sports ⇒ the photo shows a group of
athletes from different sports, typically the same city's teams posing
together, NOT fans of one team; use OCR team names + city hints to name the
teams.* Earlier, weaker wording ("trust it over OCR") failed: the model
discounted sub-0.35 signals and stayed fan-coded across 3 runs.

## Acceptance test — PASSED

1. **1980 photo:** sketch emits `sports: baseball(0.57), ice hockey(0.54),
   american football(0.35), sports equipment(0.34)` — 4 concepts, ≥ 2 distinct
   sports. ✅
2. **Blind pass flips:** blind run (no Gemini context) now reads: *"With
   baseball, hockey, and football equipment signals alongside the basketball
   jersey text, it reads like a gathering of athletes from different sports
   rather than a single-team crowd."* — no more "Sixers fans in Louisville". ✅
   Bonus: session cheaper — 2 rounds / 5.1K prompt tokens vs 3 rounds / 9K.
3. Suite green: ruff ✓ mypy ✓ pytest 51/51 ✅ (2 new parser tests).

Still not Gemini-grade: the model says "Louisville jersey" rather than
recognizing `OUISVILLE` is printed on the bat, and it never names the specific
teams. The remaining gap is the reasoning model weighing coarse signals — not
the eyes.

## Implementation steps (as executed)

1. `vision_eyes.swift`: `classifySports` + `upscale` helpers; full-frame pass,
   then salient box → 2x upscale → 2x2 grid crops; taxonomy table;
   `sports:` emit. 
2. Compiled:
   `env SDKROOT=.../MacOSX26.5.sdk swiftc -target arm64-apple-macos14 vision_eyes.swift -o vision_eyes`
   (in `~/.hermes/skills/apple/macos-vision-framework/scripts/`).
3. `backends.py`: `sports:` parse branch; `Sports:` appended right after
   `Scene:` (before OCR).
4. `orchestrator.py`: prompt clause + multi-sport override rule.
5. Tests: `test_native_parse_stdout_vlm_signals` extended + new
   `test_native_parse_stdout_sports_none`.
6. Live check + blind loop re-run. Binary lives outside the repo (same as v2).

## Risks (updated)

- **Identifier set:** verified live on the benchmark; the taxonomy table above
  is tuned to what the classifier actually emits. Other photos may surface new
  identifiers — the `sports raw` debug lines make them easy to add.
- **Vintage photo noise:** handled via 2x upscale + grid; `arena`/`rink` (0.50)
  on the museum steps is a background misread that happens to be correct
  (hockey). Expect occasional misattributions; the model weighs confidences.
- **False positives:** confidence floor 0.10 + max-merge across regions keeps
  the line tight; `skateboard` removal shows the value of dropping
  misclass-prone identifiers.
- **Stretch option (still rejected):** a bundled CoreML object-detection model
  (e.g. YOLO-variant) would give real boxes + labels, but adds a model artifact
  and repo weight. Revisit only if the classifier-crop path underperforms on
  new benchmarks.

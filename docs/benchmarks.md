# Benchmarks

This document describes how DeepSight is benchmarked, what the metrics mean,
and how to reproduce the numbers. The headline claim is **leading in
efficiency and capability**: DeepSight should reach comparable accuracy to a
direct vision model while spending far fewer tokens per correct answer.

## Design constraints

- **Zero local dataset downloads.** Benchmark rows stream from the Hugging
  Face datasets-server `/rows` API, and images load from signed cached-asset
  URLs. Nothing is downloaded to the machine running the harness.
- **Real endpoints.** The harness POSTs each question to a real
  OpenAI-compatible endpoint and records tokens and latency from the usage
  fields in the response.
- **Reproducible.** Fixed offsets and limits, temperature 0, and a fixed seed
  order make runs comparable.

## The three modes

Every benchmark run compares three modes against the same rows:

1. **Direct VLM (capability ceiling).** A strong multimodal model answers
   each question directly from the image. This sets the accuracy ceiling for
   the batch. Verified example: `sensenova-6.7-flash-lite` (free tier).
2. **One-shot description bridge (category baseline).** A vision model
   describes the whole image once, then a text model answers from the
   description. This is the visionbridge-style approach DeepSight improves on.
3. **DeepSight.** The vision-session loop: sketch, then targeted tool calls
   until the answer is produced.

The Makefile runs all three with `make bench`; per-mode targets are
`make bench-direct`, `make bench-bridge`, `make bench-deepsight`.

## Benchmarks

The `BENCHES` registry lives in `bench/harness.py`, one entry per benchmark:
`name: (dataset, config, split, image_field, question_field, answer_fields, answer_type_field)`.

| Benchmark | Dataset / config / split | What it tests | Scoring |
|---|---|---|---|
| ChartQA | `HuggingFaceM4/ChartQA`, `default`, `test` | Chart reasoning | Normalized exact match against the label |
| MathVista | `AI4Math/MathVista`, `default`, `testmini` | Visual math | Exact match; float tolerance for `float` answers; all parts for `multi` |
| OCRBench-v2 | `lmms-lab/OCRBench-v2`, `default`, `test` | OCR + screenshot/UI QA | Normalized exact match against the answer list |

OCRBench-v2 includes rico/APP agent rows, which are exactly DeepSight's sweet
spot: dense UI screenshots where targeted crops beat whole-image descriptions.

## Metrics

- **Accuracy.** Fraction of questions answered correctly, per benchmark
  scoring rules above.
- **Tokens per correct answer (headline).** Total tokens (prompt + completion)
  spent across all questions, divided by the number of correct answers. This
  rewards both capability and efficiency: a system that answers correctly with
  fewer tokens wins even at equal accuracy.
- **Average latency per question.** Wall-clock seconds from request to
  response, averaged over the batch.
- **Cache-hit rate.** Fraction of vision passes served from the perception
  cache rather than the vision backend (reported for DeepSight mode).

## How to run

Prerequisites: a running DeepSight server (or any target endpoint), a direct
VLM endpoint, and a description-bridge endpoint. API keys come from the
environment only.

```bash
# DeepSight mode against a local dev server (defaults)
make bench-deepsight

# All three modes (endpoint/model/api-key env vars required for baselines)
make bench-direct \
    DIRECT_ENDPOINT=https://token.sensenova.ai/v1 \
    DIRECT_MODEL=sensenova-6.7-flash-lite \
    DIRECT_API_KEY=...
make bench-bridge \
    BRIDGE_ENDPOINT=... BRIDGE_MODEL=... BRIDGE_API_KEY=...
make bench
```

Tunables: `BENCH_LIMIT` (rows per benchmark, default 20), `BENCH_SLEEP`
(seconds between requests for rate limiting, default 0), `BENCH_OUT` (JSON
output prefix, default `bench/results`).

The harness can also target a single benchmark directly:

```bash
python3 bench/harness.py --bench chartqa --limit 20 \
    --endpoint http://localhost:8080/v1 --model deepsight \
    --api-key local --out bench/results_chartqa.json
```

## Publishing results

When a run is complete, update the results table in the README and note the
harness version, endpoints, models, and `BENCH_LIMIT` used. Results without
that context are not comparable and should not be published.

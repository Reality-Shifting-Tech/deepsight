#!/usr/bin/env python3
"""deepsight 3-mode benchmark compare.

Runs the same rows through three answer pipelines and compares them on
accuracy and tokens-per-correct-answer:

  direct    -- a capable VLM answers the image question directly. This is
               the capability ceiling (what a full vision model costs).
  oneshot   -- a text-only LLM answers from ONE VLM-generated description
               of the whole image. This is the baseline DeepSight must beat
               on token efficiency.
  deepsight -- the DeepSight proxy: text-only reasoning model + compact
               scene sketch + targeted vision tool calls (look/crop/ocr/
               zoom) served by the local proxy endpoint.

The headline claim: DeepSight beats the oneshot bridge on
tokens-per-correct-answer while landing close to `direct` on accuracy.

Usage:
  python3 bench/compare.py --bench all --limit 10 \\
      --vlm-endpoint https://token.sensenova.ai/v1 \\
      --vlm-model sensenova-6.7-flash-lite \\
      --llm-endpoint https://api.deepseek.com/v1 \\
      --llm-model deepseek-v4-flash \\
      --deepsight-endpoint http://127.0.0.1:8080/v1 \\
      --out /tmp/compare.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import (  # noqa: E402
    BENCHES,
    extract_final,
    fetch_image,
    post_chat,
    score,
    stream_rows,
)

DESCRIBE_PROMPT = (
    "Describe this image in exhaustive detail for someone who cannot see it: "
    "transcribe ALL visible text and numbers verbatim, describe chart axes, "
    "labels, data points, spatial layout, colors, and any anomalies. "
    "Write in clear factual prose."
)

ONESHOT_SYSTEM = (
    "You answer visual questions from an image description provided by a "
    "vision model. Use ONLY the description; if it lacks the information, "
    "say so. Reply with the final answer and nothing else."
)

SYSTEMS = {
    "chartqa": (
        "Answer the chart question with just the numeric value or short phrase. No explanation."
    ),
    "ocrbench": (
        "Read the image and answer the question with the exact text shown. No explanation."
    ),
    "mathvista": (
        "Solve the visual problem. Reply with only the final answer "
        "(number or short phrase). No explanation."
    ),
}


def image_src(bench: str, row: dict) -> str:
    """Extract the image URL/src for a row (bench-specific schema)."""
    if bench == "mathvista":
        return row.get("decoded_image", {}).get("src", "")
    return row.get("image", {}).get("src", "")


def data_url_of(image_bytes: bytes, src: str) -> str:
    """Wrap raw image bytes as a data URL, guessing mime from the source."""
    mime = "image/png" if src.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


def vlm_payload(bench: str, question: str, image_bytes: bytes, src: str) -> dict:
    """Direct VLM payload: system prompt + question + image."""
    return {
        "model": "__MODEL__",
        "messages": [
            {"role": "system", "content": SYSTEMS.get(bench, "")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url_of(image_bytes, src)}},
                ],
            },
        ],
        "max_tokens": 256,
        "temperature": 0,
    }


def run_direct(args, bench: str, question: str, image_bytes: bytes, src: str) -> tuple[str, dict]:
    """One direct VLM call; returns (pred, usage)."""
    body, _ = post_chat(
        args.vlm_endpoint,
        args.vlm_key,
        args.vlm_model,
        vlm_payload(bench, question, image_bytes, src),
    )
    msg = body["choices"][0]["message"]
    pred = extract_final(msg.get("content") or msg.get("reasoning") or "")
    return pred, body.get("usage", {})


def run_oneshot(args, bench: str, question: str, image_bytes: bytes, src: str) -> tuple[str, dict]:
    """VLM caption -> text-only LLM answer; returns (pred, usage)."""
    cap_body, _ = post_chat(
        args.vlm_endpoint,
        args.vlm_key,
        args.vlm_model,
        vlm_payload(bench, DESCRIBE_PROMPT, image_bytes, src),
    )
    cap_msg = cap_body["choices"][0]["message"]
    caption = (cap_msg.get("content") or cap_msg.get("reasoning") or "").strip()
    caption_usage = cap_body.get("usage", {})

    llm_payload = {
        "model": "__MODEL__",
        "messages": [
            {"role": "system", "content": ONESHOT_SYSTEM},
            {
                "role": "user",
                "content": (f"Image description:\n{caption}\n\nQuestion: {question}\n\nAnswer:"),
            },
        ],
        "max_tokens": 128,
        "temperature": 0,
    }
    llm_body, _ = post_chat(args.llm_endpoint, args.llm_key, args.llm_model, llm_payload)
    llm_msg = llm_body["choices"][0]["message"]
    pred = extract_final(llm_msg.get("content") or llm_msg.get("reasoning") or "")

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for u in (caption_usage, llm_body.get("usage", {})):
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
    return pred, usage


def run_deepsight(args, question: str, image_bytes: bytes, src: str) -> tuple[str, dict]:
    """One call to the local DeepSight proxy; returns (pred, usage)."""
    payload = {
        "model": "deepsight",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url_of(image_bytes, src)}},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
    }
    body, _ = post_chat(args.deepsight_endpoint, None, "deepsight", payload, timeout=600)
    msg = body["choices"][0]["message"]
    pred = extract_final(msg.get("content") or msg.get("reasoning") or "")
    return pred, body.get("usage", {})


def summarize(per: dict) -> dict:
    """Collapse per-mode counters into a comparable summary."""
    n = max(per["n"], 1)
    acc = per["correct"] / n
    tokens = per["tokens_in"] + per["tokens_out"]
    tok_per_correct = tokens / per["correct"] if per["correct"] else float("inf")
    return {
        "n": per["n"],
        "correct": per["correct"],
        "acc": round(acc, 4),
        "tokens_in": per["tokens_in"],
        "tokens_out": per["tokens_out"],
        "tokens_total": tokens,
        "tok_per_correct": round(tok_per_correct, 1),
        "avg_lat_s": round(per["lat"] / n, 2),
    }


def run_one_mode(
    args, mode: str, bench: str, rows: list[tuple[int, dict]]
) -> tuple[dict, list[dict]]:
    """Run a mode over rows; returns (summary, per-row records)."""
    per = {"n": 0, "correct": 0, "tokens_in": 0, "tokens_out": 0, "lat": 0.0}
    records: list[dict] = []
    q_field = BENCHES[bench][4]
    for i, row in rows:
        question = str(row[q_field])
        gold = row[BENCHES[bench][5]]
        src = image_src(bench, row)
        if not src:
            print(f"  [{bench} #{i}] no image src, skip", file=sys.stderr)
            continue
        image_bytes = fetch_image(src)
        start = time.monotonic()
        if mode == "direct":
            pred, usage = run_direct(args, bench, question, image_bytes, src)
        elif mode == "oneshot":
            pred, usage = run_oneshot(args, bench, question, image_bytes, src)
        else:
            pred, usage = run_deepsight(args, question, image_bytes, src)
        lat = time.monotonic() - start
        correct = score(bench, row, pred)
        ti = usage.get("prompt_tokens", 0)
        to = usage.get("completion_tokens", 0)
        per["n"] += 1
        per["correct"] += int(correct)
        per["tokens_in"] += ti
        per["tokens_out"] += to
        per["lat"] += lat
        records.append(
            {
                "i": i,
                "pred": pred,
                "gold": str(gold),
                "correct": bool(correct),
                "tokens_in": ti,
                "tokens_out": to,
                "lat_s": round(lat, 2),
            }
        )
        print(
            f"    [{i}] {'OK ' if correct else 'XX '} pred={pred[:50]!r} "
            f"gold={str(gold)[:50]!r} {lat:.1f}s",
            flush=True,
        )
        if args.sleep:
            time.sleep(args.sleep)
    return summarize(per), records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="all", help="chartqa|mathvista|ocrbench|all")
    ap.add_argument("--limit", type=int, default=10, help="rows per bench")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.25, help="pacing between calls")
    ap.add_argument("--vlm-endpoint", default=os.environ.get("VLM_ENDPOINT", ""))
    ap.add_argument("--vlm-model", default=os.environ.get("VLM_MODEL", ""))
    ap.add_argument("--vlm-key", default=os.environ.get("VLM_KEY", ""))
    ap.add_argument("--llm-endpoint", default=os.environ.get("LLM_ENDPOINT", ""))
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", ""))
    ap.add_argument("--llm-key", default=os.environ.get("LLM_KEY", ""))
    ap.add_argument("--deepsight-endpoint", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--modes", default="direct,oneshot,deepsight", help="comma-separated modes")
    ap.add_argument("--out", default="", help="write JSON results to this path")
    args = ap.parse_args()

    if not (args.vlm_endpoint and args.vlm_model and args.llm_endpoint and args.llm_model):
        ap.error("--vlm-endpoint/--vlm-model/--llm-endpoint/--llm-model are required")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    benchs = list(BENCHES) if args.bench == "all" else [args.bench]
    if args.bench not in BENCHES and args.bench != "all":
        ap.error(f"unknown bench {args.bench!r}; choose from {list(BENCHES)}")

    summary: dict[str, dict[str, dict]] = {}
    rows_out: dict[str, dict[str, list[dict]]] = {}
    verdicts: list[str] = []

    for bench in benchs:
        print(f"\n== {bench} ==", flush=True)
        summary[bench] = {}
        rows_out[bench] = {}
        rows = list(enumerate(stream_rows(bench, args.offset, args.limit)))
        for mode in modes:
            print(f"  running {mode} ({len(rows)} rows)...", flush=True)
            s, records = run_one_mode(args, mode, bench, rows)
            summary[bench][mode] = s
            rows_out[bench][mode] = records
            print(
                f"    {mode}: n={s['n']} acc={s['acc']:.1%} "
                f"tok/correct={s['tok_per_correct']} avg_lat={s['avg_lat_s']}s",
                flush=True,
            )

        if "deepsight" in summary[bench] and "oneshot" in summary[bench]:
            ds, os_m = summary[bench]["deepsight"], summary[bench]["oneshot"]
            if ds["tok_per_correct"] < os_m["tok_per_correct"]:
                verdicts.append(
                    f"{bench}: DEEPSIGHT WINS tok/correct "
                    f"{ds['tok_per_correct']} < {os_m['tok_per_correct']} "
                    f"(acc {ds['acc']:.1%} vs {os_m['acc']:.1%})"
                )
            else:
                verdicts.append(
                    f"{bench}: oneshot wins tok/correct "
                    f"{os_m['tok_per_correct']} < {ds['tok_per_correct']}"
                )
        if "deepsight" in summary[bench] and "direct" in summary[bench]:
            ds, dr = summary[bench]["deepsight"], summary[bench]["direct"]
            verdicts.append(
                f"{bench}: deepsight acc {ds['acc']:.1%} vs direct "
                f"{dr['acc']:.1%} (gap {dr['acc'] - ds['acc']:+.1%})"
            )

    print("\n==== VERDICT ====")
    for v in verdicts:
        print(f"  {v}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "meta": {
                        "bench": args.bench,
                        "limit": args.limit,
                        "modes": modes,
                        "vlm_model": args.vlm_model,
                        "llm_model": args.llm_model,
                    },
                    "summary": summary,
                    "rows": rows_out,
                    "verdicts": verdicts,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""deepsight bench harness — cloud-streaming eval, zero local dataset downloads.

Streams benchmark rows from HF datasets-server (/rows API), fetches images via
signed cached-asset URLs, POSTs each question to any OpenAI-compatible
endpoint, scores, and records tokens + latency per question.

Usage:
  python3 bench/harness.py --bench chartqa --limit 20 \\
      --endpoint https://token.sensenova.ai/v1 \\
      --model sensenova-6.7-flash-lite --api-key KEY --out /tmp/chartqa.json
  python3 bench/harness.py --bench all --limit 10 \\
      --endpoint http://localhost:8080/v1 --model deepsight
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

DS = "https://datasets-server.huggingface.co"

BENCHES = {
    # name: (dataset, config, split, image_field, question_field, answer_fields, answer_type_field)
    "chartqa": (
        "HuggingFaceM4/ChartQA",
        "default",
        "test",
        "image",
        "query",
        "label",
        None,
    ),
    "mathvista": (
        "AI4Math/MathVista",
        "default",
        "testmini",
        "decoded_image",
        "question",
        "answer",
        "answer_type",
    ),
    "ocrbench": (
        "lmms-lab/OCRBench-v2",
        "default",
        "test",
        "image",
        "question",
        "answers",
        None,
    ),
}


def fetch_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stream_rows(bench: str, offset: int, length: int) -> list[dict]:
    dataset, config, split, *_ = BENCHES[bench]
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{DS}/rows?{params}"
    d = fetch_json(url)
    rows = d.get("rows", [])
    if not rows:
        err = d.get("error") or d.get("message") or "no rows"
        raise RuntimeError(f"{bench}: datasets-server: {err}")
    return [r["row"] for r in rows]


def fetch_image(src: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(src, headers={"Accept": "image/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,;:!?\"'")
    s = re.sub(r"^\$", "", s)
    s = re.sub(r"[,%]", "", s)
    return s


def extract_final(raw: str) -> str:
    """Pull the usable final answer out of a model response.

    Reasoning models return chain-of-thought in the ``reasoning`` field
    with an empty ``content``; the answer is the text after a trailing
    ``answer:`` marker, or the last non-empty line. Plain answers pass
    through unchanged.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.search(r"(?:final\s+)?answer\s*[:：]\s*(.+)$", ln, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return lines[-1]


def to_float(s: str) -> float | None:
    m = re.search(r"-?\d+\.?\d*", s.replace(",", ""))
    return float(m.group()) if m else None


def score(bench: str, row: dict, pred: str) -> bool:
    """Score a prediction against a row's gold answer.

    Exact (normalized) match first; falls back to a word-boundary
    substring match so verbose model answers (DeepSight tool-loop
    sentences, reasoning traces) still score when they contain the
    gold value.
    """
    *_, afields, atype_field = BENCHES[bench]
    if bench == "mathvista":
        gold = str(row["answer"])
        atype = row.get(atype_field, "free_form")
        if atype == "float":
            g, p = to_float(gold), to_float(pred)
            return g is not None and p is not None and abs(g - p) <= max(0.05, abs(g) * 0.01)
        # multi: check all parts
        if atype == "multi":
            parts = [normalize(x) for x in gold.split(";") if x.strip()]
            pn = normalize(pred)
            return all(pp in pn for pp in parts)
        if normalize(pred) == normalize(gold):
            return True
        return _contains_gold(gold, pred)
    golds = [str(a) for a in (row[afields] if isinstance(row[afields], list) else [row[afields]])]
    pn = normalize(pred)
    if any(normalize(g) == pn for g in golds):
        return True
    return any(_contains_gold(g, pred) for g in golds)


def _contains_gold(gold: str, pred: str) -> bool:
    """Token substring check: does pred contain gold as a standalone token?

    Uses negative lookaround on word chars (not ``\\b``) so golds ending
    in non-word characters like ``145°`` still match inside verbose
    answers.
    """
    g = normalize(gold)
    if not g:
        return False
    pn = normalize(pred)
    if not pn:
        return False
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", pn) is not None


def make_payload(bench: str, row: dict, image_b64: str, mime: str) -> dict:
    dataset, *_ = BENCHES[bench]
    question = row[BENCHES[bench][4]]
    system = ""
    if bench == "chartqa":
        system = (
            "Answer the chart question with just the numeric value or short phrase. No explanation."
        )
    elif bench == "ocrbench":
        system = "Read the image and answer the question with the exact text shown. No explanation."
    elif bench == "mathvista":
        system = (
            "Solve the visual problem. Reply with only the final answer "
            "(number or short phrase). No explanation."
        )
    return {
        "model": "__MODEL__",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            },
        ],
        "max_tokens": 256,
        "temperature": 0,
    }


def post_chat(
    endpoint: str, api_key: str | None, model: str, payload: dict, timeout: int = 120
) -> dict:
    payload["model"] = model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    lat = time.monotonic() - t0
    return body, lat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", choices=[*BENCHES, "all"], default="chartqa")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--endpoint", required=True, help="base URL, e.g. https://token.sensenova.ai/v1"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--sleep", type=float, default=0.0, help="seconds between requests (rate limiting)"
    )
    args = ap.parse_args()

    benches = list(BENCHES) if args.bench == "all" else [args.bench]
    results = {}
    grand = defaultdict(
        lambda: {"n": 0, "correct": 0, "tokens_in": 0, "tokens_out": 0, "latency": 0.0}
    )

    for bench in benches:
        rows = stream_rows(bench, args.offset, args.limit)
        per = grand[bench]
        out_rows = []
        for i, row in enumerate(rows):
            if bench == "mathvista":
                isrc = row.get("decoded_image", {}).get("src")
            else:
                isrc = row.get("image", {}).get("src")
            if not isrc:
                print(f"  [{bench} #{i}] no image src, skip", file=sys.stderr)
                continue
            img = fetch_image(isrc)
            mime = "image/png" if isrc.lower().endswith(".png") else "image/jpeg"
            payload = make_payload(bench, row, __import__("base64").b64encode(img).decode(), mime)
            body, lat = post_chat(args.endpoint, args.api_key, args.model, payload)
            msg = body["choices"][0]["message"]
            pred = extract_final(msg.get("content") or msg.get("reasoning") or "")
            usage = body.get("usage", {})
            correct = score(bench, row, pred)
            per["n"] += 1
            per["correct"] += 1 if correct else 0
            per["tokens_in"] += usage.get("prompt_tokens", 0)
            per["tokens_out"] += usage.get("completion_tokens", 0)
            per["latency"] += lat
            out_rows.append(
                {
                    "i": i,
                    "q": row[BENCHES[bench][4]][:120],
                    "pred": pred[:160],
                    "gold": str(row["answer"] if bench != "chartqa" else row["label"]),
                    "correct": correct,
                    "latency_s": round(lat, 2),
                    "tokens": usage,
                }
            )
            print(
                f"  [{bench} #{i}] {'✅' if correct else '❌'} "
                f"pred={pred[:60]!r} gold={out_rows[-1]['gold'][:60]!r} {lat:.1f}s"
            )
            if args.sleep:
                time.sleep(args.sleep)

        results[bench] = {"rows": out_rows}

    print("\n=== SUMMARY ===")
    for bench, per in grand.items():
        acc = per["correct"] / per["n"] if per["n"] else 0
        tok_per_correct = (
            (per["tokens_in"] + per["tokens_out"]) / per["correct"]
            if per["correct"]
            else float("inf")
        )
        print(
            f"{bench:10s} n={per['n']:3d} acc={acc:.2%}  tok/correct={tok_per_correct:8.1f}  "
            f"avg_lat={per['latency'] / max(per['n'], 1):.1f}s"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "meta": {
                        "bench": args.bench,
                        "model": args.model,
                        "endpoint": args.endpoint,
                        "limit": args.limit,
                    },
                    "summary": {b: dict(p) for b, p in grand.items()},
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

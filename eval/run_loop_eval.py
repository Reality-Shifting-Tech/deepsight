"""eval/run_loop_eval.py — reasoning-loop regression eval (tier 1).

Same manifest as tier 0, but every image goes through the full DeepSight
vision-session loop (orchestrator + perception + reasoning backend) with
a BLIND prompt: the model never sees ground truth or the expectations.
The final description is scored against the entry's optional ``loop``
block:

    loop: {"required": [..], "forbidden": [..]}

required   tokens that must appear in the final description (normalized)
forbidden  tokens that must NOT appear (hallucination/contradiction)

Gap semantics match tier 0 (see run_eval.py): ``"gap": true`` entries
that fail are accepted (⚠️) and never gate the exit code; a gap entry
that passes is a gap-closed event (🎉). ``--skip-missing`` marks absent
images as skipped (CI runs without golden photos).

Cost: one reasoning session per image (a few cents on deepseek-v4-flash);
intended for nightly runs, not per-commit.

Usage:
    python eval/run_loop_eval.py [--manifest eval/manifest.json]
                                 [--out eval/results/loop] [--skip-missing]
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from deepsight.backends import NativeVisionBackend, ReasoningBackend
from deepsight.cache import PerceptionCache
from deepsight.config import get_settings
from deepsight.orchestrator import Orchestrator

_RUNNER = Path(__file__).resolve().parent / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval_shared", _RUNNER)
assert _spec and _spec.loader
_run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_eval)

norm = _run_eval.norm
load_manifest = _run_eval.load_manifest
status_for = _run_eval.status_for
exit_code_for = _run_eval.exit_code_for

BLIND_PROMPT = (
    "Describe what is in this image. Cover: the overall scene, any people and what "
    "they appear to be doing, any visible text, and whether any sport or athletic "
    "activity is shown. Be specific; name things when the evidence supports it."
)


def score_loop(entry_id: str, text: str, loop_block: dict) -> list[dict]:
    """Score a final description against required/forbidden tokens.

    Checks:
    - ``required``: every fragment must appear (normalized substring).
    - ``loop_any``: at least one of the fragments must appear (handles
      paraphrase variance: "surfing" / "surfer" / "surfboard").
    - ``forbidden``: a fragment fails only when it is asserted. Mentions
      that are negated ("no baseball", "not hockey") or hedged as weak
      signals ("a faint baseball hint", "maybe tennis") do not fail.
    """
    results: list[dict] = []
    detail = f"answer: {text[:200]}"
    raw_lower = text.lower()
    for frag in loop_block.get("required", []):
        passed = norm(frag) in norm(text)
        results.append({
            "id": entry_id,
            "check": f"loop_required '{frag}'",
            "passed": passed,
            "detail": detail,
        })
    for frags in [loop_block.get("loop_any", [])]:
        if not frags:
            continue
        passed = any(norm(f) in norm(text) for f in frags)
        results.append({
            "id": entry_id,
            "check": f"loop_any '{frags}'",
            "passed": passed,
            "detail": detail,
        })
    for frag in loop_block.get("forbidden", []):
        passed = not _asserts_fragment(raw_lower, frag.lower())
        results.append({
            "id": entry_id,
            "check": f"loop_forbidden '{frag}'",
            "passed": passed,
            "detail": detail,
        })
    return results


_NEGATION = re.compile(r"\b(no|not|nor|without|unlikely|rather than)\b[^.!?;\n]{0,30}$")
_HEDGE = re.compile(r"\b(faint|weak|slight|maybe|perhaps|possibly|hint|trace|supposed)\b[^.!?;\n]{0,20}$")  # noqa: E501


def _asserts_fragment(raw_lower: str, frag: str) -> bool:
    """True when the fragment appears in an assertive (non-negated) context.

    A mention is non-assertive when a negation or hedging word sits within
    ~30 chars before it: "no baseball equipment", "a faint baseball hint".
    The fragment fails the guard only if at least one assertive mention
    exists (e.g. "this is baseball, not hockey" asserts baseball).
    """
    unassertive = 0
    total = 0
    for m in re.finditer(re.escape(frag), raw_lower):
        total += 1
        prefix = raw_lower[max(0, m.start() - 40):m.start()]
        if _NEGATION.search(prefix) or _HEDGE.search(prefix):
            unassertive += 1
    return total > 0 and unassertive < total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.json")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "loop")
    ap.add_argument("--skip-missing", action="store_true",
                    help="treat missing images as skipped (CI runs without golden photos)")
    ap.add_argument("--max-images", type=int, default=0, help="limit to first N images (debug)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    settings = get_settings()
    if not settings.reasoning_key:
        print("ERROR: DEEPSIGHT_REASONING_API_KEY not set (run from repo root with .env present)")
        return 2
    if not Path(settings.vision_bin).exists():
        print(f"ERROR: eyes binary not found at {settings.vision_bin} (set DEEPSIGHT_VISION_BIN)")
        return 2

    vision = NativeVisionBackend(bin_path=settings.vision_bin)
    reasoning = ReasoningBackend(
        base_url=settings.reasoning_base_url,
        api_key=settings.reasoning_api_key,
        model=settings.reasoning_model,
        temperature=settings.reasoning_temperature,
        max_tokens=settings.reasoning_max_tokens,
    )
    cache = (
        PerceptionCache(ttl_seconds=settings.cache_ttl_seconds)
        if settings.cache_enabled else None
    )
    orch = Orchestrator(
        reasoning=reasoning,
        vision=vision,
        cache=cache,
        max_look_rounds=settings.max_look_rounds,
        sketch_enabled=settings.sketch_enabled,
        tool_round_max_tokens=settings.reasoning_tool_round_max_tokens,
    )

    results: list[dict] = []
    for i, entry in enumerate(manifest["images"]):
        if args.max_images and i >= args.max_images:
            break
        eid = entry["id"]
        src = entry.get("source", "?")
        path = entry["_resolved"]
        loop_block = entry.get("loop")
        print(f"  {eid} ({src})...", flush=True)
        if not Path(path).exists():
            if args.skip_missing:
                results.append({"id": eid, "source": src, "path": path, "status": "skipped"})
            else:
                results.append({"id": eid, "source": src, "path": path,
                                "status": "error", "error": "image missing"})
            continue
        if not loop_block:
            results.append({"id": eid, "source": src, "path": path,
                            "status": "skip", "error": "no loop expectations in manifest"})
            continue
        data_url = "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()
        try:
            result = orch.run(data_url, BLIND_PROMPT)
        except Exception as exc:  # network/API hiccups must not kill the run
            results.append({"id": eid, "source": src, "path": path,
                            "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        text = result.content or ""
        checks = score_loop(eid, text, loop_block)
        passed = sum(1 for c in checks if c["passed"])
        entry_ok = passed == len(checks) and len(checks) > 0
        results.append({
            "id": eid,
            "source": src,
            "path": path,
            "note": entry.get("note", ""),
            "gap": bool(entry.get("gap")),
            "status": status_for(entry, entry_ok),
            "ok": entry_ok,
            "checks_passed": passed,
            "checks_total": len(checks),
            "checks": checks,
            "content": text,
            "rounds": result.rounds,
            "tool_calls": result.tool_calls,
            "cache_hits": result.cache_hits,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        })
        tok = result.prompt_tokens + result.completion_tokens
        print(f"      -> {passed}/{len(checks)} ({tok} tok)", flush=True)

    # ---- scoreboard ----
    print("\n== loop eval scoreboard ==")
    for r in results:
        status = r.get("status")
        if status == "skipped":
            print(f"  ⏭️  {r['id']} ({r['source']}) — skipped (image missing)")
            continue
        if r.get("error"):
            print(f"  ❌ {r['id']} ({r['source']}) — ERROR {r['error']}")
            continue
        if status == "pass" and r.get("gap"):
            print(f"  🎉 {r['id']} ({r['source']}) — GAP CLOSED — "
                  f"{r['checks_passed']}/{r['checks_total']}")
        elif status == "pass":
            print(f"  ✅ {r['id']} ({r['source']}) — {r['checks_passed']}/{r['checks_total']}")
        elif status == "gap":
            print(f"  ⚠️ {r['id']} ({r['source']}) — known gap (not counted) — "
                  f"{r['checks_passed']}/{r['checks_total']}")
        else:
            print(f"  ❌ {r['id']} ({r['source']}) — {r['checks_passed']}/{r['checks_total']}")
        for c in r["checks"]:
            if not c["passed"]:
                print(f"      ✗ {c['check']} — {c['detail']}")

    total_checks = sum(r.get("checks_total", 0) for r in results)
    passed_checks = sum(r.get("checks_passed", 0) for r in results)
    ok_images = sum(1 for r in results if r.get("status") == "pass")
    gap_images = sum(1 for r in results if r.get("status") == "gap")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    total_tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in results)
    headline = f"images: {ok_images}/{len(results)} pass"
    if gap_images:
        headline += f" · {gap_images} known gap"
    if skipped:
        headline += f" · {skipped} skipped"
    if total_checks:
        headline += (f"  ·  checks: {passed_checks}/{total_checks} "
                     f"({100.0 * passed_checks / total_checks:.1f}%)")
    else:
        headline += "  ·  no checks ran"
    print(f"\n  {headline}")
    print(f"  tokens: {total_tokens} (~${total_tokens / 1_000_000 * 0.30:.4f} at flash rates)")

    # ---- persist ----
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    out_file = args.out / f"{stamp}.json"
    out_file.write_text(json.dumps({
        "model": settings.reasoning_model,
        "manifest": str(args.manifest),
        "images_total": len(results),
        "images_ok": ok_images,
        "gaps_open": gap_images,
        "skipped": skipped,
        "checks_total": total_checks,
        "checks_passed": passed_checks,
        "total_tokens": total_tokens,
        "exit_code": exit_code_for(results),
        "results": results,
    }, indent=2))
    print(f"\n  saved: {out_file}")

    return exit_code_for(results)


if __name__ == "__main__":
    sys.exit(main())

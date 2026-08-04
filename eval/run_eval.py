"""eval/run_eval.py — eyes regression eval (tier 0).

Runs the compiled ``vision_eyes`` binary over every image in the manifest and
scores the raw signals (OCR, scene, sports, counts) against expected ground
truth. Zero tokens, zero network, zero model downloads: the binary is the only
dependency, so this runs in seconds on any macOS box and is CI-safe.

Checks supported in a manifest entry's ``expected`` block:

    ocr_contains    [str]    each string appears in the OCR text (normalized)
    ocr_empty       bool     no OCR lines at all
    scene_top1      str      first scene label equals this (case-insensitive)
    scene_contains  [{label, rank}]  label appears within the top-``rank`` scene labels
    sports_present  [str]    label appears in the merged sports lines
    sports_min      int      distinct sport labels (excl. "sports equipment") >= n
    sports_absent   bool     no sports entries at all
    humans_zero     bool     humans count == 0
    faces_min       int      faces count >= n
    rectangles_min  int      rectangles count >= n
    animals_none    bool     no animal labels
    forbidden       [str]    none of these strings appear in the OCR text

Gap semantics: an entry with ``"gap": true`` that fails is reported as a
known gap (⚠️) and does NOT affect the exit code; a gap entry that passes
is a gap-closed event (🎉). ``--skip-missing`` marks absent images as
skipped instead of failed (CI runs where the golden photos are
local-only).

Usage:
    python eval/run_eval.py [--manifest eval/manifest.json] [--bin PATH]
                            [--out eval/results] [--skip-missing]

The binary path can also come from $DEEPSIGHT_VISION_BIN (that is what a CI
macOS job will set).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = os.environ.get("DEEPSIGHT_VISION_BIN") or str(
    Path.home() / ".hermes/skills/apple/macos-vision-framework/scripts/vision_eyes"
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(text: str) -> str:
    """Lowercase, collapse non-alphanumerics to single spaces."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def parse_conf_list(segment: str) -> list[tuple[str, float]]:
    """Parse ``label(0.97), other(0.30)`` into [(label, conf), ...]."""
    out: list[tuple[str, float]] = []
    for item in segment.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"^(.*)\(([0-9.]+)\)$", item)
        if m:
            out.append((m.group(1).strip(), float(m.group(2))))
        else:
            out.append((item, 0.0))
    return out


def parse_signals(raw: str) -> dict:
    """Extract structured signals from vision_eyes stdout.

    Mirrors NativeVisionBackend._parse_stdout: OCR lines are exactly
    2-space-indented; scene/sports/counts are section markers.
    """
    ocr: list[str] = []
    scene: list[tuple[str, float]] = []
    sports: list[tuple[str, float]] = []
    counts: dict[str, int] = {}
    animals: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if line.startswith("  "):
            ocr.append(stripped)
        elif stripped.startswith("scene:"):
            scene = parse_conf_list(stripped.removeprefix("scene:").strip())
        elif stripped.startswith("sports:"):
            segment = stripped.removeprefix("sports:").strip()
            if segment and segment != "none":
                sports.extend(parse_conf_list(segment))
        elif stripped.startswith("animals:"):
            segment = stripped.removeprefix("animals:").strip()
            if segment and segment != "none":
                animals.append(segment)
    for key in ("faces", "humans", "rectangles"):
        m = re.search(rf"{key}:\s*(\d+)", raw)
        if m:
            counts[key] = int(m.group(1))
    return {"ocr": ocr, "scene": scene, "sports": sports, "counts": counts, "animals": animals}


def run_eyes(bin_path: str, image_path: str, timeout: float = 120.0) -> tuple[str, str]:
    """Run the binary; return (stdout, stderr)."""
    proc = subprocess.run(
        [bin_path, image_path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.stdout, proc.stderr


def check(entry_id: str, name: str, passed: bool, detail: str = "") -> dict:
    return {"id": entry_id, "check": name, "passed": bool(passed), "detail": detail}


def status_for(entry: dict, entry_ok: bool) -> str:
    """Classify an entry outcome: 'pass', 'gap' (accepted known failure), or 'fail'."""
    if entry_ok:
        return "pass"
    return "gap" if entry.get("gap") else "fail"


def exit_code_for(results: list[dict]) -> int:
    """0 unless any result needs attention: hard errors or non-gap failures.

    Skipped images and accepted gaps never gate the run, so CI and the
    nightly watch stay green while known limitations are tracked as ⚠️.
    """
    for r in results:
        if r.get("status") in ("fail", "error"):
            return 1
    return 0


def score(entry_id: str, signals: dict, expected: dict) -> list[dict]:
    """Evaluate one image's signals against its expected block."""
    ocr_joined = " ".join(signals["ocr"])
    ocr_norm = norm(ocr_joined)
    scene_labels = [label for label, _ in signals["scene"]]
    sport_labels = [label for label, _ in signals["sports"]]
    distinct_sports = [s for s in set(sport_labels) if s != "sports equipment"]
    counts = signals["counts"]

    results: list[dict] = []

    for frag in expected.get("ocr_contains", []):
        passed = norm(frag) in ocr_norm
        results.append(check(entry_id, f"ocr_contains '{frag}'", passed,
                             f"got OCR: {ocr_joined[:120] or '(none)'}"))

    if "ocr_empty" in expected:
        passed = not signals["ocr"]
        ocr_snip = f"got OCR: {ocr_joined[:120] or '(none)'}"
        results.append(check(entry_id, "ocr_empty", passed, ocr_snip))

    if "scene_top1" in expected:
        got = scene_labels[0] if scene_labels else None
        passed = got is not None and norm(got) == norm(expected["scene_top1"])
        results.append(check(entry_id, f"scene_top1 '{expected['scene_top1']}'", passed,
                             f"got scene: {scene_labels[:3] or '(none)'}"))

    for spec in expected.get("scene_contains", []):
        rank = int(spec.get("rank", 3))
        pool = [norm(label) for label in scene_labels[:rank]]
        passed = any(pool) and norm(spec["label"]) in pool
        results.append(check(entry_id, f"scene_contains '{spec['label']}' top-{rank}", passed,
                             f"got scene: {scene_labels[:rank] or '(none)'}"))

    for sport in expected.get("sports_present", []):
        passed = any(norm(sport) == norm(s) for s in sport_labels)
        results.append(check(entry_id, f"sports_present '{sport}'", passed,
                             f"got sports: {sport_labels or '(none)'}"))

    if "sports_min" in expected:
        n = int(expected["sports_min"])
        passed = len(distinct_sports) >= n
        results.append(check(entry_id, f"sports_min >= {n}", passed,
                             f"got distinct: {distinct_sports or '(none)'}"))

    if "sports_absent" in expected:
        passed = not sport_labels
        results.append(check(entry_id, "sports_absent", passed,
                             f"got sports: {sport_labels or '(none)'}"))

    if "humans_zero" in expected:
        passed = counts.get("humans", 0) == 0
        results.append(check(entry_id, "humans_zero", passed,
                             f"got humans: {counts.get('humans', 'n/a')}"))

    if "faces_min" in expected:
        passed = counts.get("faces", 0) >= int(expected["faces_min"])
        results.append(check(entry_id, f"faces_min >= {expected['faces_min']}", passed,
                             f"got faces: {counts.get('faces', 'n/a')}"))

    if "rectangles_min" in expected:
        passed = counts.get("rectangles", 0) >= int(expected["rectangles_min"])
        results.append(check(entry_id, f"rectangles_min >= {expected['rectangles_min']}", passed,
                             f"got rectangles: {counts.get('rectangles', 'n/a')}"))

    if "animals_none" in expected:
        passed = not signals["animals"]
        results.append(check(entry_id, "animals_none", passed,
                             f"got animals: {signals['animals'] or '(none)'}"))

    for animal in expected.get("animals_present", []):
        joined = " ".join(signals["animals"]).lower()
        passed = any(norm(animal) in joined for animal in [animal])
        results.append(check(entry_id, f"animals_present '{animal}'", passed,
                             f"got animals: {signals['animals'] or '(none)'}"))

    for frag in expected.get("forbidden", []):
        passed = norm(frag) not in ocr_norm
        results.append(check(entry_id, f"forbidden '{frag}'", passed,
                             f"OCR had: {ocr_joined[:120] or '(none)'}"))

    return results


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    base = path.resolve().parent
    for entry in data["images"]:
        p = Path(entry["path"])
        if not p.is_absolute():
            p = base / p
        entry["_resolved"] = str(p)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.json")
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    ap.add_argument("--skip-missing", action="store_true",
                    help="treat missing images as skipped (CI runs without golden photos)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    bin_path = args.bin
    if not Path(bin_path).exists():
        print(f"ERROR: eyes binary not found at {bin_path} (set DEEPSIGHT_VISION_BIN or --bin)")
        return 2

    results: list[dict] = []
    per_signal: dict[str, list[bool]] = {}

    for entry in manifest["images"]:
        eid = entry["id"]
        src = entry.get("source", "?")
        path = entry["_resolved"]
        is_gap = bool(entry.get("gap"))
        if not Path(path).exists():
            if args.skip_missing:
                results.append({"id": eid, "source": src, "path": path, "status": "skipped"})
                continue
            results.append({"id": eid, "source": src, "path": path,
                            "status": "error", "error": "image missing"})
            continue
        try:
            stdout, stderr = run_eyes(bin_path, path)
        except subprocess.TimeoutExpired:
            results.append({"id": eid, "source": src, "path": path,
                            "status": "error", "error": "timeout"})
            continue
        if "loaded" not in stdout and stderr:
            results.append({"id": eid, "source": src, "path": path,
                            "status": "error", "error": f"binary error: {stderr.strip()[:200]}"})
            continue
        signals = parse_signals(stdout)
        checks = score(eid, signals, entry.get("expected", {}))
        passed = sum(1 for c in checks if c["passed"])
        entry_ok = passed == len(checks) and len(checks) > 0
        for c in checks:
            per_signal.setdefault(c["check"].split(" ")[0], []).append(c["passed"])
        results.append({
            "id": eid,
            "source": src,
            "path": path,
            "note": entry.get("note", ""),
            "gap": is_gap,
            "status": status_for(entry, entry_ok),
            "ok": entry_ok,
            "checks_passed": passed,
            "checks_total": len(checks),
            "checks": checks,
            "signals": signals,
        })

    # ---- scoreboard ----
    print("\n== eval scoreboard ==")
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
    if per_signal:
        parts = [f"{k}: {sum(v)}/{len(v)}" for k, v in sorted(per_signal.items())]
        print("  per-signal: " + " · ".join(parts))

    # ---- confidence stats ----
    confs: dict[str, list[float]] = {"scene": [], "sports": []}
    for r in results:
        for _label, conf in r.get("signals", {}).get("scene", []):
            confs["scene"].append(conf)
        for _label, conf in r.get("signals", {}).get("sports", []):
            confs["sports"].append(conf)
    if confs["scene"]:
        print(f"  scene conf: avg {sum(confs['scene'])/len(confs['scene']):.2f} "
              f"over {len(confs['scene'])} labels")
    if confs["sports"]:
        print(f"  sports conf: avg {sum(confs['sports'])/len(confs['sports']):.2f} "
              f"over {len(confs['sports'])} labels")

    # ---- persist ----
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    out_file = args.out / f"{stamp}.json"
    out_file.write_text(json.dumps({
        "bin": bin_path,
        "manifest": str(args.manifest),
        "images_total": len(results),
        "images_ok": ok_images,
        "gaps_open": gap_images,
        "skipped": skipped,
        "checks_total": total_checks,
        "checks_passed": passed_checks,
        "exit_code": exit_code_for(results),
        "results": results,
    }, indent=2))
    print(f"\n  saved: {out_file}")

    return exit_code_for(results)


if __name__ == "__main__":
    sys.exit(main())

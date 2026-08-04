"""eval/nightly.py — nightly regression watch (eyes + reasoning loop).

Runs both evals over the manifest, compares against the previous nightly
results, and prints a compact delta report. Prints NOTHING when the
baseline is unchanged (cron watchdog pattern: empty stdout = silent).

Exit code: 0 after a normal run (even with known gaps); 1 when a runner
itself broke (missing binary/key) so the cron error alert fires.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EYES_DIR = ROOT / "eval" / "results"
LOOP_DIR = EYES_DIR / "loop"


def _uv_cmd() -> list[str]:
    """Locate uv (repo-managed env for run_loop_eval's deps) across common paths."""
    uv = shutil.which("uv")
    if not uv:
        for cand in (Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv"):
            if cand.exists():
                uv = str(cand)
                break
    if not uv:
        raise RuntimeError("uv not found; cannot run the loop eval")
    return [uv, "run", "python"]


def _latest(directory: Path) -> dict | None:
    files = sorted(directory.glob("*.json"))
    return json.loads(files[-1].read_text()) if files else None


def _summary(data: dict) -> dict:
    return {
        "checks_passed": data.get("checks_passed", 0),
        "checks_total": data.get("checks_total", 0),
        "images_ok": data.get("images_ok", 0),
        "gaps_open": data.get("gaps_open", 0),
        "skipped": data.get("skipped", 0),
        "failed": sorted(r["id"] for r in data.get("results", [])
                         if r.get("status") in ("fail", "error")),
        "gap_closed": sorted(r["id"] for r in data.get("results", [])
                             if r.get("status") == "pass" and r.get("gap")),
        "gaps": sorted(r["id"] for r in data.get("results", [])
                       if r.get("status") == "gap"),
    }


def _fmt(s: dict) -> str:
    return f"{s['images_ok']} img · {s['checks_passed']}/{s['checks_total']} chk"


def _prev(directory: Path) -> dict | None:
    latest = _latest(directory)
    return _summary(latest) if latest else None


def main() -> int:
    prev_eyes = _prev(EYES_DIR)
    prev_loop = _prev(LOOP_DIR)

    for name, cmd in (
        ("eyes", [sys.executable, "eval/run_eval.py"]),
        ("loop", [*_uv_cmd(), "eval/run_loop_eval.py"]),
    ):
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
        if proc.returncode not in (0, 1):
            print(f"🔴 deepsight nightly: {name} runner broke (exit {proc.returncode})")
            print((proc.stdout or proc.stderr)[-800:])
            return 1

    cur_eyes = _summary(_latest(EYES_DIR) or {})
    cur_loop = _summary(_latest(LOOP_DIR) or {})

    changed = cur_eyes != prev_eyes or cur_loop != prev_loop
    if not changed:
        return 0  # silent: baseline unchanged

    lines = [f"🧠 deepsight nightly · {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"]
    eyes_line = f"eyes: {_fmt(cur_eyes)}"
    eyes_line += f" (prev {_fmt(prev_eyes)})" if prev_eyes else " (first run)"
    lines.append(eyes_line)
    loop_line = f"loop: {_fmt(cur_loop)}"
    loop_line += f" (prev {_fmt(prev_loop)})" if prev_loop else " (first run)"
    lines.append(loop_line)
    for name, cur, prev in (("eyes", cur_eyes, prev_eyes), ("loop", cur_loop, prev_loop)):
        for eid in cur["gap_closed"]:
            lines.append(f"🎉 {name}: {eid} GAP CLOSED")
        for eid in cur["failed"]:
            if not prev or eid not in prev["failed"]:
                lines.append(f"🔴 {name}: NEW FAIL {eid}")
        if prev:
            for eid in prev["failed"]:
                if eid not in cur["failed"]:
                    lines.append(f"✅ {name}: {eid} now passing")
    open_gaps = sorted(set(cur_eyes["gaps"]) | set(cur_loop["gaps"]))
    if open_gaps:
        lines.append("⚠️ open gaps: " + ", ".join(open_gaps))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

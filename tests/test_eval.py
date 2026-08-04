"""Unit tests for the eyes eval runner's pure logic.

These run anywhere (no macOS, no vision_eyes binary): they exercise the
parser and the scoring rules against fabricated stdout, which is exactly
what a CI job on any OS can do. The image side of the eval is covered by
``eval/run_eval.py`` on macOS.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"
spec = importlib.util.spec_from_file_location("run_eval", RUNNER)
assert spec and spec.loader
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

SAMPLE_STDOUT = """\
loaded 791x1024
  SIXERS
  THUNDE
scene: people(0.76), basketball(0.40), crowd(0.30)
sports: baseball(0.56), ice hockey(0.50), american football(0.30), sports equipment(0.27)
faces: 4, humans: 4
rectangles: 0
animals: none
"""


def test_norm() -> None:
    assert run_eval.norm("SIXERS") == "sixers"
    assert run_eval.norm("SALE 50% OFF!") == "sale 50 off"
    assert run_eval.norm("  Deep-Sight ") == "deep sight"


def test_parse_conf_list() -> None:
    assert run_eval.parse_conf_list("a(0.97), b(0.30)") == [("a", 0.97), ("b", 0.30)]
    assert run_eval.parse_conf_list("ice hockey(0.50), sports equipment(0.27)") == [
        ("ice hockey", 0.50),
        ("sports equipment", 0.27),
    ]
    assert run_eval.parse_conf_list("") == []


def test_parse_signals() -> None:
    sig = run_eval.parse_signals(SAMPLE_STDOUT)
    assert sig["ocr"] == ["SIXERS", "THUNDE"]
    assert sig["scene"] == [("people", 0.76), ("basketball", 0.40), ("crowd", 0.30)]
    assert sig["sports"] == [
        ("baseball", 0.56),
        ("ice hockey", 0.50),
        ("american football", 0.30),
        ("sports equipment", 0.27),
    ]
    assert sig["counts"] == {"faces": 4, "humans": 4, "rectangles": 0}
    assert sig["animals"] == []


def test_parse_signals_sports_none() -> None:
    sig = run_eval.parse_signals("loaded 10x10\nscene: none(0.10)\nsports: none\n")
    assert sig["sports"] == []


def test_score_ocr_contains() -> None:
    sig = run_eval.parse_signals(SAMPLE_STDOUT)
    checks = run_eval.score("t", sig, {"ocr_contains": ["SIXERS", "THUNDE"]})
    assert all(c["passed"] for c in checks)
    checks = run_eval.score("t", sig, {"ocr_contains": ["PHILADELPHIA"]})
    assert not checks[0]["passed"]


def test_score_ocr_empty() -> None:
    sig = run_eval.parse_signals(SAMPLE_STDOUT)
    assert not run_eval.score("t", sig, {"ocr_empty": True})[0]["passed"]
    bare = run_eval.parse_signals("loaded 10x10\nscene: none(0.10)\n")
    assert run_eval.score("t", bare, {"ocr_empty": True})[0]["passed"]


def test_score_scene_top1_and_contains() -> None:
    sig = run_eval.parse_signals(SAMPLE_STDOUT)
    assert run_eval.score("t", sig, {"scene_top1": "people"})[0]["passed"]
    assert not run_eval.score("t", sig, {"scene_top1": "basketball"})[0]["passed"]
    checks = run_eval.score("t", sig, {"scene_contains": [{"label": "crowd", "rank": 3}]})
    assert checks[0]["passed"]
    checks = run_eval.score("t", sig, {"scene_contains": [{"label": "crowd", "rank": 2}]})
    assert not checks[0]["passed"]


def test_score_sports() -> None:
    sig = run_eval.parse_signals(SAMPLE_STDOUT)
    assert all(run_eval.score("t", sig, {"sports_present": ["baseball", "ice hockey"]}))
    assert not run_eval.score("t", sig, {"sports_present": ["tennis"]})[0]["passed"]
    # sports_min excludes the "sports equipment" catch-all
    assert run_eval.score("t", sig, {"sports_min": 3})[0]["passed"]
    assert not run_eval.score("t", sig, {"sports_min": 4})[0]["passed"]
    assert not run_eval.score("t", sig, {"sports_absent": True})[0]["passed"]


def test_score_counts_and_animals() -> None:
    sig = run_eval.parse_signals(SAMPLE_STDOUT)
    assert run_eval.score("t", sig, {"humans_zero": True})[0]["passed"] is False
    assert run_eval.score("t", sig, {"faces_min": 4})[0]["passed"]
    assert run_eval.score("t", sig, {"faces_min": 5})[0]["passed"] is False
    assert run_eval.score("t", sig, {"rectangles_min": 1})[0]["passed"] is False
    assert run_eval.score("t", sig, {"animals_none": True})[0]["passed"]
    assert not run_eval.score("t", sig, {"animals_present": ["cat"]})[0]["passed"]

    with_animal = run_eval.parse_signals(
        "loaded 10x10\nscene: animal(0.9)\nanimals: cat(0.9), dog(0.7)\n"
    )
    assert run_eval.score("t", with_animal, {"animals_present": ["cat"]})[0]["passed"]
    assert run_eval.score("t", with_animal, {"animals_none": True})[0]["passed"] is False


def test_score_forbidden() -> None:
    sig = run_eval.parse_signals(SAMPLE_STDOUT)
    assert run_eval.score("t", sig, {"forbidden": ["LOUISVILLE"]})[0]["passed"]
    assert not run_eval.score("t", sig, {"forbidden": ["SIXERS"]})[0]["passed"]


def test_manifest_schema_well_formed() -> None:
    """Every manifest entry must resolve to an existing relative/absolute path
    and carry at least one expectation, so the eval can never silently skip."""
    manifest_path = Path(__file__).resolve().parent.parent / "eval" / "manifest.json"
    manifest = run_eval.load_manifest(manifest_path)
    assert len(manifest["images"]) >= 10
    for entry in manifest["images"]:
        assert entry["id"], "every entry needs an id"
        assert entry["_resolved"], f"{entry['id']}: path did not resolve"
        assert entry.get("expected"), f"{entry['id']}: no expected block"


# ---------------------------------------------------------------------------
# Gap semantics (tier 0 runner)
# ---------------------------------------------------------------------------


def test_status_for() -> None:
    assert run_eval.status_for({}, True) == "pass"
    assert run_eval.status_for({}, False) == "fail"
    assert run_eval.status_for({"gap": True}, False) == "gap"
    assert run_eval.status_for({"gap": True}, True) == "pass"


def test_exit_code_for() -> None:
    assert run_eval.exit_code_for([]) == 0
    assert run_eval.exit_code_for(
        [{"status": "pass"}, {"status": "gap"}, {"status": "skipped"}]
    ) == 0
    assert run_eval.exit_code_for([{"status": "fail"}]) == 1
    assert run_eval.exit_code_for([{"status": "error", "error": "timeout"}]) == 1


# ---------------------------------------------------------------------------
# Tier 1 reasoning-loop scorer
# ---------------------------------------------------------------------------

LOOP_RUNNER = Path(__file__).resolve().parent.parent / "eval" / "run_loop_eval.py"


def load_run_loop_eval() -> Any:
    spec = importlib.util.spec_from_file_location("run_loop_eval_shared", LOOP_RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loop_score_required_forbidden() -> None:
    loop_mod = load_run_loop_eval()
    ok = loop_mod.score_loop("t", "A soccer stadium on a sunny afternoon",
                             {"required": ["soccer"], "forbidden": ["hockey"]})
    assert ok[0]["passed"] and ok[1]["passed"]
    bad = loop_mod.score_loop("t", "An ice hockey arena full of fans",
                              {"required": ["soccer"], "forbidden": ["hockey"]})
    assert not bad[0]["passed"] and not bad[1]["passed"]


def test_loop_score_normalization() -> None:
    loop_mod = load_run_loop_eval()
    normed = loop_mod.score_loop("t", "STOP! Deep-Sight 50% OFF",
                                 {"required": ["deep sight", "stop"], "forbidden": ["soccer"]})
    assert all(c["passed"] for c in normed)
    empty = loop_mod.score_loop("t", "nothing here", {"required": []})
    assert empty == []


def test_loop_score_any() -> None:
    loop_mod = load_run_loop_eval()
    ok = loop_mod.score_loop("t", "A lone surfer standing on his board",
                             {"loop_any": ["surfing", "surfer", "surfboard"]})
    assert ok[0]["passed"]
    bad = loop_mod.score_loop("t", "A dark moody ocean at night",
                              {"loop_any": ["surfing", "surfer", "surfboard"]})
    assert not bad[0]["passed"]


def test_loop_forbidden_negation_and_hedge() -> None:
    loop_mod = load_run_loop_eval()
    # negation: "no baseball" is a denial, not a claim
    neg = loop_mod.score_loop("t", "Basketball game; no baseball equipment is shown",
                              {"forbidden": ["baseball"]})
    assert neg[0]["passed"]
    # hedging: "a faint baseball hint" is a weak signal report, not a claim
    hedge = loop_mod.score_loop("t", "Court action, despite a faint baseball hint in the scene",
                                {"forbidden": ["baseball"]})
    assert hedge[0]["passed"]
    # assertion: plain mention fails the guard
    assert_ = loop_mod.score_loop("t", "The players switch to baseball after warmups",
                                  {"forbidden": ["baseball"]})
    assert not assert_[0]["passed"]
    # mixed: negated + assertive mentions still fail (assertion exists)
    mixed = loop_mod.score_loop("t", "No hockey today; this is baseball season",
                                {"forbidden": ["hockey", "baseball"]})
    assert mixed[0]["passed"]
    assert not mixed[1]["passed"]

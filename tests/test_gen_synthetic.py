"""Unit tests for the synthetic image generator's pure logic.

CI-safe (no macOS, no vision_eyes binary): exercises the generator
registry, the variant matrix, and the conservative expectation rules.
Determinism is proven via the --smoke probe: the same seeded input must
produce byte-identical images and identical manifest entries.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

GEN = Path(__file__).resolve().parent.parent / "eval" / "gen_synthetic.py"
spec = importlib.util.spec_from_file_location("gen_synthetic", GEN)
assert spec and spec.loader
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def expected_for(gen_name: str, variant: str) -> dict:
    return gen.build_expected(gen_name, variant)


def test_smoke_is_deterministic() -> None:
    first = json.dumps(gen.smoke(), sort_keys=True)
    second = json.dumps(gen.smoke(), sort_keys=True)
    assert first == second


def test_smoke_matches_subprocess_output() -> None:
    proc = subprocess.run(
        [sys.executable, str(GEN), "--smoke"],
        capture_output=True,
        text=True,
        check=True,
    )
    sub = json.loads(proc.stdout)
    assert sub == gen.smoke()


def test_smoke_entry_schema() -> None:
    for eid, info in gen.smoke().items():
        assert eid.startswith("synth-")
        assert set(info) == {"sha256", "expected"}
        for key, value in info["expected"].items():
            if key == "ocr_contains":
                assert isinstance(value, list) and value
            elif key in ("ocr_empty", "humans_zero", "sports_absent"):
                assert value is True
            elif key == "rectangles_min":
                assert isinstance(value, int) and value >= 1
            else:
                raise AssertionError(f"unexpected expectation key: {key}")


def test_full_matrix_size() -> None:
    count = sum(
        1
        for g in gen.GENERATORS
        for v in gen.VARIANTS
        if not (g == "noise" and v == "noise")
    )
    assert count == 87  # 11 generators x 8 variants, minus noise/noise
    assert count + 10 == 97  # plus 10 fuzz smoke images


def test_textcard_ocr_rules() -> None:
    truth = gen.TEXT_TRUTH["textcard"]
    for variant in gen.SAFE_OCR:
        exp = expected_for("textcard", variant)
        assert exp["ocr_contains"] == truth
        assert "ocr_empty" not in exp
    for variant in ("fliph", "flipv"):
        exp = expected_for("textcard", variant)
        assert "ocr_contains" not in exp
        assert "ocr_empty" not in exp  # flips produce garbage OCR, not none
        assert exp["humans_zero"] is True
        assert exp["sports_absent"] is True


def test_geometry_rect_rules() -> None:
    for g in gen.RECT_GENS:
        for variant in gen.VARIANTS:
            exp = expected_for(g, variant)
            assert exp["rectangles_min"] == 1
            assert exp["ocr_empty"] is True
            assert exp["humans_zero"] is True
            assert exp["sports_absent"] is True


def test_plain_negative_rules() -> None:
    plain = set(gen.GENERATORS) - gen.TEXT_GENS - gen.RECT_GENS
    for g in plain:
        for variant in gen.VARIANTS:
            exp = expected_for(g, variant)
            assert set(exp) == {"ocr_empty", "humans_zero", "sports_absent"}


def test_fuzz_entries_are_smoke_tests() -> None:
    for i in range(10):
        entry = gen.fuzz_entry(i)
        assert entry["id"] == f"fuzz-{i:02d}"
        assert entry["source"] == "synthetic"
        assert entry["expected"] == {}
        assert entry["path"] == f"images/synthetic/fuzz-{i:02d}.png"

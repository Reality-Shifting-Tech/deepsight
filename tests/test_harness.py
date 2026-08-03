"""Tests for the bench harness scoring + extraction logic (no network)."""

import importlib.util
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "bench"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("harness", BENCH_DIR / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize(harness):
    assert harness.normalize("  14 ") == "14"
    assert harness.normalize("$1,234") == "1234"
    assert harness.normalize("  42% ") == "42"
    assert harness.normalize("Answer: 7.") == "answer: 7"


def test_extract_final_plain(harness):
    assert harness.extract_final("14") == "14"
    assert harness.extract_final("  ") == ""


def test_extract_final_reasoning_field(harness):
    cot = "I need to look at the chart.\nLet me count the bars.\nanswer: 14"
    assert harness.extract_final(cot) == "14"


def test_extract_final_last_line(harness):
    assert harness.extract_final("first line\nsecond line") == "second line"


def test_to_float(harness):
    assert harness.to_float("3.5") == 3.5
    assert harness.to_float("$12.99") == 12.99
    assert harness.to_float("about 7") == 7.0
    assert harness.to_float("n/a") is None


def test_score_chartqa_list_label(harness):
    row = {"label": ["14"], "query": "how many?"}
    assert harness.score("chartqa", row, "14") is True
    assert harness.score("chartqa", row, "15") is False
    assert harness.score("chartqa", row, " 14 ") is True


def test_score_ocrbench(harness):
    row = {"answers": ["Hello World"], "question": "what text?"}
    assert harness.score("ocrbench", row, "Hello World") is True
    assert harness.score("ocrbench", row, "hello world") is True
    assert harness.score("ocrbench", row, "Goodbye") is False


def test_score_mathvista_float(harness):
    row = {"answer": "3.5", "answer_type": "float"}
    assert harness.score("mathvista", row, "3.51") is True
    assert harness.score("mathvista", row, "3.6") is False


def test_score_mathvista_multi(harness):
    row = {"answer": "A; B", "answer_type": "multi"}
    assert harness.score("mathvista", row, "A and B") is True
    assert harness.score("mathvista", row, "A only") is False


def test_score_mathvista_free_form(harness):
    row = {"answer": "Triangle", "answer_type": "free_form"}
    assert harness.score("mathvista", row, "triangle") is True


def test_benches_registry(harness):
    assert set(harness.BENCHES) == {"chartqa", "mathvista", "ocrbench"}
    for _, meta in harness.BENCHES.items():
        assert len(meta) == 7
        assert meta[0]  # dataset
        assert meta[4]  # question field
        assert meta[5]  # answer field(s)

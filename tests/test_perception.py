"""Tests for perception: sketch generation and the tool protocol."""

import pytest
from PIL import Image

from deepsight.cache import PerceptionCache
from deepsight.perception import TOOL_DEFINITIONS, Perception


@pytest.fixture
def image(png_bytes):
    return Image.open(__import__("io").BytesIO(png_bytes))


def test_sketch_disabled_returns_empty(fake_vision, image):
    vision = fake_vision(text="{}")
    perc = Perception(vision, sketch_enabled=False)
    assert perc.sketch(image) == ""
    assert vision.calls == 0


def test_sketch_calls_vision_and_counts_tokens(fake_vision, image):
    vision = fake_vision(text='{"objects": ["x"]}', prompt=7, completion=3)
    perc = Perception(vision, sketch_enabled=True)
    text = perc.sketch(image)
    assert text == '{"objects": ["x"]}'
    assert vision.calls == 1
    assert perc.total_prompt_tokens == 7
    assert perc.total_completion_tokens == 3


def test_sketch_cached_same_image(fake_vision, image):
    vision = fake_vision(text="sketch", prompt=7, completion=3)
    perc = Perception(vision, cache=PerceptionCache(ttl_seconds=3600))
    assert perc.sketch(image) == "sketch"
    assert perc.sketch(image) == "sketch"
    assert vision.calls == 1
    assert perc.cache_hits == 1


def test_look_tool(fake_vision, image):
    vision = fake_vision(text="a chart bar", prompt=5, completion=2)
    perc = Perception(vision)
    out = perc.execute("look", {"x": 0, "y": 0, "w": 100, "h": 100}, image)
    assert "a chart bar" in out
    assert out.startswith("look region")
    assert vision.calls == 1


def test_ocr_tool(fake_vision, image):
    vision = fake_vision(text="Total: 42", prompt=5, completion=2)
    perc = Perception(vision)
    out = perc.execute("ocr", {"x": 10, "y": 10, "w": 80, "h": 80}, image)
    assert out.startswith("OCR of region")
    assert "Total: 42" in out


def test_zoom_tool(fake_vision, image):
    vision = fake_vision(text="tiny detail", prompt=5, completion=2)
    perc = Perception(vision)
    out = perc.execute("zoom", {"x": 0, "y": 0, "w": 50, "h": 50}, image)
    assert "tiny detail" in out
    assert out.startswith("zoom region")


def test_unknown_tool(fake_vision, image):
    vision = fake_vision()
    perc = Perception(vision)
    assert perc.execute("teleport", {}, image) == "unknown tool: teleport"
    assert vision.calls == 0


def test_tool_definitions_shape():
    names = {t.name for t in TOOL_DEFINITIONS}
    assert names == {"look", "ocr", "zoom", "count"}
    for t in TOOL_DEFINITIONS:
        assert t.parameters["type"] == "object"
        assert "required" in t.parameters


def test_parse_text_markers(fake_vision, image):
    perc = Perception(fake_vision())
    calls = perc.parse_text_markers("Look at [LOOK 10,20,30,40] and [LOOK 0,0,5,5] now")
    assert calls == [
        ("look", {"x": 10, "y": 20, "w": 30, "h": 40}),
        ("look", {"x": 0, "y": 0, "w": 5, "h": 5}),
    ]

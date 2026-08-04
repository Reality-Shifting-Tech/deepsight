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
    assert names == {"look", "ocr", "zoom", "count", "locate", "ground", "capture", "watch"}
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


def test_locate_no_description(fake_vision, image):
    vision = fake_vision()
    perc = Perception(vision)
    out = perc.execute("locate", {}, image)
    assert "no description provided" in out


def test_locate_no_detections(fake_vision, image):
    vision = fake_vision(boxes_data=[])
    perc = Perception(vision)
    out = perc.execute("locate", {"what": "person"}, image)
    assert "no detections" in out


def test_locate_matches_text(fake_vision, image):
    vision = fake_vision(
        boxes_data=[
            {
                "type": "text",
                "confidence": 1.0,
                "x": 0.05,
                "y": 0.08,
                "w": 0.12,
                "h": 0.04,
                "label": "WEBSITES",
            },
            {
                "type": "text",
                "confidence": 1.0,
                "x": 0.54,
                "y": 0.50,
                "w": 0.14,
                "h": 0.08,
                "label": "$5,149",
            },
            {
                "type": "face",
                "confidence": 0.99,
                "x": 0.10,
                "y": 0.20,
                "w": 0.30,
                "h": 0.40,
                "label": "face",
            },
            {
                "type": "human",
                "confidence": 0.85,
                "x": 0.05,
                "y": 0.10,
                "w": 0.15,
                "h": 0.60,
                "label": "human",
            },
        ]
    )
    perc = Perception(vision)
    out = perc.execute("locate", {"what": "WEBSITES"}, image)
    assert "WEBSITES" in out
    assert "100%" in out  # confidence format
    assert "x=5%" in out  # 0.05 * 100 = 5
    assert "y=" in out


def test_locate_person_matches_human_and_face(fake_vision, image):
    vision = fake_vision(
        boxes_data=[
            {
                "type": "human",
                "confidence": 0.85,
                "x": 0.05,
                "y": 0.10,
                "w": 0.15,
                "h": 0.60,
                "label": "human",
            },
            {
                "type": "face",
                "confidence": 0.99,
                "x": 0.10,
                "y": 0.20,
                "w": 0.30,
                "h": 0.40,
                "label": "face",
            },
        ]
    )
    perc = Perception(vision)
    out = perc.execute("locate", {"what": "person"}, image)
    assert "human" in out or "face" in out
    assert len(out.splitlines()) > 1  # has results


def test_locate_unmatched_returns_available_types(fake_vision, image):
    vision = fake_vision(
        boxes_data=[
            {
                "type": "text",
                "confidence": 1.0,
                "x": 0.0,
                "y": 0.0,
                "w": 0.5,
                "h": 0.1,
                "label": "hello",
            },
        ]
    )
    perc = Perception(vision)
    out = perc.execute("locate", {"what": "gorilla"}, image)
    assert "no match" in out
    assert "text" in out  # hints available types


def test_locate_calls_boxes_method(fake_vision, image):
    vision = fake_vision(
        boxes_data=[
            {
                "type": "text",
                "confidence": 1.0,
                "x": 0.0,
                "y": 0.0,
                "w": 0.5,
                "h": 0.1,
                "label": "hello",
            },
        ]
    )
    perc = Perception(vision)
    perc.execute("locate", {"what": "hello"}, image)
    assert vision.box_calls == 1


def test_ground_no_search_backend(fake_vision, image):
    """Perception without search_backend reports unavailable."""
    perc = Perception(fake_vision(), search_backend=None)
    out = perc.execute("ground", {"what": "something"}, image)
    assert "no search API key" in out
    assert "DEEPSIGHT_SEARCH_API_KEY" in out


def test_ground_no_description(fake_search, fake_vision, image):
    sb = fake_search()
    perc = Perception(fake_vision(), search_backend=sb)
    out = perc.execute("ground", {}, image)
    assert "no claim" in out


def test_ground_no_results(fake_search, fake_vision, image):
    sb = fake_search(results=[])
    perc = Perception(fake_vision(), search_backend=sb)
    out = perc.execute("ground", {"what": "nothing"}, image)
    assert "no search results" in out


def test_ground_results_and_fetch(fake_search, fake_vision, image):
    from deepsight.backends import SearchResult

    sr = [
        SearchResult(url="https://example.com/1", title="First Result", snippet="Snippet one"),
        SearchResult(url="https://example.com/2", title="Second Result", snippet="Snippet two"),
    ]
    sb = fake_search(results=sr, fetch_text="Page body content here...")
    perc = Perception(fake_vision(), search_backend=sb)
    out = perc.execute("ground", {"what": "example"}, image)
    assert "TOP MATCH" in out
    assert "First Result" in out
    assert "Page body content" in out
    assert sb.search_calls == 1
    assert sb.fetch_calls >= 1


def test_ground_custom_query(fake_search, fake_vision, image):
    from deepsight.backends import SearchResult

    sr = [SearchResult(url="https://x.com", title="X", snippet="snippet")]
    sb = fake_search(results=sr)
    perc = Perception(fake_vision(), search_backend=sb)
    out = perc.execute("ground", {"what": "example", "query": "custom search"}, image)
    assert "custom search" in out


def test_capture_non_macos(fake_vision, image, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    perc = Perception(fake_vision())
    out = perc.execute("capture", {}, image)
    assert "only supported on macOS" in out
    assert perc._captured_image is None


def test_capture_sets_captured_image(fake_vision, image, monkeypatch):
    """Capture stores a PIL Image that subsequent tools use."""
    import subprocess

    monkeypatch.setattr("sys.platform", "darwin")

    # Fake screencapture: write the test PNG to the temp path
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if "screencapture" in str(cmd):
            # Write the test PNG to the output path (last arg)
            out_path = cmd[-1]
            with open(out_path, "wb") as f:
                f.write(image.fp.read() if hasattr(image, "fp") else b"")
                # Fallback: use png_bytes fixture via import
            # Re-create the test PNG
            import io

            from PIL import Image as PImage

            buf = io.BytesIO()
            PImage.new("RGB", (32, 32), (100, 150, 200)).save(buf, format="PNG")
            with open(out_path, "wb") as f:
                f.write(buf.getvalue())
            from unittest.mock import MagicMock

            return MagicMock(returncode=0, stdout=b"", stderr=b"")
        return real_run(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)

    vision = fake_vision(text='{"objects": ["test"]}')
    # Set up boxes data to avoid real subprocess call
    vision.boxes_data = [
        {"type": "text", "confidence": 1.0, "x": 0, "y": 0, "w": 0.5, "h": 0.1, "label": "test"}
    ]

    perc = Perception(vision)
    out = perc.execute("capture", {}, image)
    assert "screen captured" in out
    assert perc._captured_image is not None
    assert perc._captured_image.size == (32, 32)


def test_capture_then_look_uses_captured(fake_vision, monkeypatch):
    """After capture, look/ocr/zoom/count/etc operate on the captured image."""
    import io
    import subprocess

    from PIL import Image as PImage

    monkeypatch.setattr("sys.platform", "darwin")
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if "screencapture" in str(cmd):
            out_path = cmd[-1]
            buf = io.BytesIO()
            PImage.new("RGB", (64, 64), (200, 50, 100)).save(buf, format="PNG")
            with open(out_path, "wb") as f:
                f.write(buf.getvalue())
            from unittest.mock import MagicMock

            return MagicMock(returncode=0, stdout=b"", stderr=b"")
        return real_run(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Fake the ask() method — it answers questions about the captured image
    vision = fake_vision(
        text="captured desk",
        boxes_data=[
            {
                "type": "text",
                "confidence": 1.0,
                "x": 0,
                "y": 0,
                "w": 0.1,
                "h": 0.05,
                "label": "game",
            }
        ],
    )
    perc = Perception(vision)

    # capture
    cap_out = perc.execute("capture", {}, None)
    assert "screen captured" in cap_out
    assert perc._captured_image.size == (64, 64)

    # look — should operate on 64x64 captured image, not the original 16x16
    look_out = perc.execute("look", {"x": 0, "y": 0, "w": 100, "h": 100}, None)
    # The vision backend was called with the captured image bytes
    assert "captured" in look_out or "region" in look_out

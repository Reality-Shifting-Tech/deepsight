"""Tests for the backend clients: reasoning (mocked HTTP) + native eyes (subprocess)."""

import pytest

from deepsight import backends
from deepsight.backends import NativeVisionBackend, ReasoningBackend


class FakeResponse:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


class FakeClient:
    """Records the last POST for assertions, returns a canned response."""

    last_url: str | None = None
    last_json: dict | None = None
    last_headers: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        self.timeout = kwargs.get("timeout")

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *args) -> None:
        pass

    def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> FakeResponse:
        type(self).last_url = url
        type(self).last_headers = headers
        type(self).last_json = json
        return FakeResponse(self._response_for(url, json))

    def _response_for(self, url: str, json: dict | None) -> dict:
        if "chat/completions" in url:
            if json and json.get("tools"):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {"name": "look", "arguments": '{"x": 1}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            return {
                "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5},
            }
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    FakeClient.last_url = None
    FakeClient.last_json = None
    FakeClient.last_headers = None
    monkeypatch.setattr(backends.httpx, "Client", FakeClient)


# ---------------------------------------------------------------------------
# Reasoning backend (OpenAI-compatible chat)
# ---------------------------------------------------------------------------


def test_reasoning_backend_max_tokens_in_payload():
    b = ReasoningBackend("https://api.example.com/v1", api_key="sk-x")
    b.chat([{"role": "user", "content": "hi"}], max_tokens=64)
    assert FakeClient.last_json["max_tokens"] == 64


def test_reasoning_backend_parses_cache_usage():
    class CacheResponse(FakeResponse):
        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 5,
                    "prompt_cache_hit_tokens": 490,
                    "prompt_cache_miss_tokens": 10,
                },
            }

    class CacheClient(FakeClient):
        def _response_for(self, url, json):
            return CacheResponse({}).json()

    backends.httpx.Client = CacheClient
    b = ReasoningBackend("https://api.example.com/v1")
    res = b.chat([{"role": "user", "content": "hi"}])
    assert res.prompt_tokens == 500
    assert res.cache_hit_tokens == 490
    assert res.cache_miss_tokens == 10


def test_reasoning_backend_plain_chat():
    b = ReasoningBackend("https://api.example.com/v1", api_key="sk-x")
    res = b.chat([{"role": "user", "content": "hi"}])
    assert res.content == "42"
    assert res.prompt_tokens == 9
    assert res.completion_tokens == 5
    assert not res.wants_tools
    assert FakeClient.last_url == "https://api.example.com/v1/chat/completions"
    assert FakeClient.last_headers["Authorization"] == "Bearer sk-x"


def test_reasoning_backend_no_key_sends_no_auth():
    b = ReasoningBackend("https://api.example.com/v1")
    b.chat([{"role": "user", "content": "hi"}])
    assert "Authorization" not in FakeClient.last_headers


def test_reasoning_backend_tool_calls():
    b = ReasoningBackend("https://api.example.com/v1", api_key="sk-x")
    res = b.chat([{"role": "user", "content": "look"}], tools=[{"type": "function"}])
    assert res.wants_tools
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "look"
    assert res.tool_calls[0].arguments == {"x": 1}


def test_reasoning_backend_malformed_tool_args():
    class MalformedResponse(FakeResponse):
        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": "look", "arguments": "not json"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            }

    class MalformedClient(FakeClient):
        def _response_for(self, url, json):
            return MalformedResponse({}).json()

    backends.httpx.Client = MalformedClient
    b = ReasoningBackend("https://api.example.com/v1")
    res = b.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert res.wants_tools
    assert res.tool_calls[0].arguments == {}


def test_safe_json():
    assert backends._safe_json('{"a": 1}') == {"a": 1}
    assert backends._safe_json("garbage") == {}
    assert backends._safe_json("[1,2]") == {}


# ---------------------------------------------------------------------------
# Native vision backend (Apple Vision binary via subprocess)
# ---------------------------------------------------------------------------


def test_native_parse_stdout_sections():
    raw = (
        "  MIKE\n"
        "  MYERS\n"
        "  crypto.com\n"
        "scene: adult(0.91), people(0.91)\n"
        "faces: 1\n"
        "humans: 1\n"
        "rectangles: 2\n"
    )
    out = NativeVisionBackend._parse_stdout(raw)
    assert "OCR text:" in out
    assert "MIKE" in out
    assert "Scene: adult(0.91), people(0.91)" in out
    assert "faces: 1" in out
    assert "humans: 1" in out
    assert "rectangles: 2" in out


def test_native_parse_stdout_vlm_signals():
    raw = (
        "  THUNDER\n"
        "scene: recreation(0.97), sport(0.97), basketball(0.97)\n"
        "faces: 1\n"
        "face attr: 0: roll=0° yaw=45° pitch=?\n"
        "face quality: 0.39\n"
        "humans: 4\n"
        "pose: 5 human(s)\n"
        "pose 0: joints=18 arms_up=false\n"
        "animals: none\n"
        "colors: #C0B0A0(9%), #000000(4%)\n"
        "sports: baseball(0.71), basketball(0.55)\n"
    )
    out = NativeVisionBackend._parse_stdout(raw)
    assert "OCR text:" in out
    assert "THUNDER" in out
    assert "Scene: recreation(0.97), sport(0.97), basketball(0.97)" in out
    assert "Pose: 5 human(s); 0: joints=18 arms_up=false" in out
    assert "Face attrs: 0: roll=0° yaw=45° pitch=?" in out
    assert "Face quality: 0.39" in out
    assert "Animals: none" in out
    assert "Colors: #C0B0A0(9%), #000000(4%)" in out
    assert "Sports: baseball(0.71), basketball(0.55)" in out


def test_native_parse_stdout_sports_none():
    out = NativeVisionBackend._parse_stdout("scene: people(0.8)\nsports: none\n")
    assert "Sports: none" in out


def test_native_parse_stdout_empty():
    out = NativeVisionBackend._parse_stdout("")
    assert out == "(no text or scene detected)"


def test_native_ask_success(monkeypatch, tmp_path):
    import subprocess

    img = tmp_path / "img.png"
    img.write_bytes(b"pngdata")

    class FakeProc:
        returncode = 0
        stdout = "  HELLO\nscene: text(0.9)\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    b = NativeVisionBackend(bin_path="/fake/vision_eyes")
    res = b.ask("what is this", b"pngdata")
    assert res.text == "OCR text:\nHELLO\nScene: text(0.9)"
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0


def test_native_ask_error(monkeypatch, tmp_path):
    import subprocess

    img = tmp_path / "img.png"
    img.write_bytes(b"pngdata")

    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    b = NativeVisionBackend(bin_path="/fake/vision_eyes")
    res = b.ask("what is this", b"pngdata")
    assert res.text.startswith("vision_eyes error:")
    assert res.prompt_tokens == 0


def test_parse_boxes_text_and_faces():
    """_parse_boxes extracts typed boxes with Y-flip."""
    raw = (
        "scene: people(0.9)\n"
        "  HELLO\n"
        "box:text:1.0000:0.0521:0.8816:0.1196:0.0403:WEBSITES\n"
        "box:text:1.0000:0.2166:0.8947:0.0901:0.0240:15-20 Days\n"
        "box:face:0.9900:0.1000:0.2000:0.3000:0.4000:face\n"
        "box:human:0.8500:0.0500:0.1000:0.1500:0.6000:human\n"
        "box:animal:0.9500:0.5000:0.5000:0.1000:0.1000:Cat\n"
        "box:rect:0.7000:0.6000:0.6000:0.2000:0.3000:rectangle\n"
        "box:salient:1.0000:0.0000:0.0000:0.5000:0.5000:salient_object\n"
        "ALL DONE\n"
    )
    boxes = NativeVisionBackend._parse_boxes(raw)
    assert len(boxes) == 7

    # text box: apple-vision y=0.8816 h=0.0403 -> top-left y = 1 - 0.8816 - 0.0403 = 0.0781
    t = boxes[0]
    assert t["type"] == "text"
    assert t["label"] == "WEBSITES"
    assert abs(t["y"] - 0.0781) < 0.001
    assert abs(t["x"] - 0.0521) < 0.001

    # face box (index 2)
    f = boxes[2]
    assert f["type"] == "face"
    assert f["confidence"] == 0.99
    assert abs(f["y"] - 0.4) < 0.001  # 1 - 0.2 - 0.4 = 0.4

    # animal box (index 4)
    a = boxes[4]
    assert a["label"] == "Cat"
    assert abs(a["y"] - 0.4) < 0.001  # 1 - 0.5 - 0.1 = 0.4

    # rect box (index 5)
    r = boxes[5]
    assert r["type"] == "rect"
    assert abs(r["y"] - 0.1) < 0.001  # 1 - 0.6 - 0.3


def test_parse_boxes_malformed():
    """Malformed box lines are silently skipped."""
    raw = (
        "box:text:1.0000:0.1:0.2:0.3:0.4:hello\n"
        "box:truncated:bad\n"
        "not_a_box_line\n"
        "box:float_error:not_a_float:0:0:0:0:label\n"
    )
    boxes = NativeVisionBackend._parse_boxes(raw)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "hello"


def test_parse_boxes_empty():
    """No box lines returns empty list."""
    boxes = NativeVisionBackend._parse_boxes("scene: empty\nsports: none\n")
    assert boxes == []


def test_native_boxes_success(monkeypatch, tmp_path):
    """boxes() shells the binary and returns parsed results."""
    import subprocess

    class FakeProc:
        returncode = 0
        stdout = "box:text:1.0:0.0:0.0:1.0:1.0:full frame\nALL DONE\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    b = NativeVisionBackend(bin_path="/fake/vision_eyes")
    boxes = b.boxes(b"pngdata")
    assert len(boxes) == 1
    assert boxes[0]["type"] == "text"


def test_native_boxes_error(monkeypatch, tmp_path):
    """boxes() returns empty on binary error."""
    import subprocess

    class FakeProc:
        returncode = 1
        stdout = ""
        stderr = "crash"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    b = NativeVisionBackend(bin_path="/fake/vision_eyes")
    assert b.boxes(b"pngdata") == []


def test_windows_vision_ask_no_tesseract():
    """WindowsVisionBackend.ask() works without pytesseract (PIL only)."""
    from deepsight.backends import WindowsVisionBackend

    b = WindowsVisionBackend()
    assert b._tesseract is None
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (100, 150, 200)).save(buf, format="PNG")
    result = b.ask("what is this", buf.getvalue())
    assert "image: 32x32" in result.text
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_windows_vision_boxes_without_tesseract():
    from deepsight.backends import WindowsVisionBackend
    b = WindowsVisionBackend()
    assert b.boxes(b"pngdata") == []


def test_windows_vision_ask_with_dark_image():
    import io

    from PIL import Image

    from deepsight.backends import WindowsVisionBackend
    b = WindowsVisionBackend()
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 10, 10)).save(buf, format="PNG")
    result = b.ask("", buf.getvalue())
    assert "dark" in result.text
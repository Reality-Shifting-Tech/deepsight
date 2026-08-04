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

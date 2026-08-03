"""Tests for the HTTP backend clients, all network calls mocked."""

import pytest

from deepsight import backends
from deepsight.backends import (
    OllamaVisionBackend,
    OpenAICompatibleVisionBackend,
    ReasoningBackend,
    build_vision_backend,
)


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
        if "/api/chat" in url:
            return {
                "message": {"content": " vision reply "},
                "prompt_eval_count": 11,
                "eval_count": 4,
            }
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


def test_ollama_vision_backend():
    b = OllamaVisionBackend("http://127.0.0.1:11434", model="minicpm-v:latest")
    res = b.ask("what is this", b"imagedata")
    assert res.text == "vision reply"
    assert res.prompt_tokens == 11
    assert res.completion_tokens == 4
    assert FakeClient.last_url == "http://127.0.0.1:11434/api/chat"
    payload = FakeClient.last_json
    assert payload["model"] == "minicpm-v:latest"
    assert payload["messages"][0]["images"][0]  # base64 present
    assert payload["stream"] is False


def test_openai_vision_backend():
    b = OpenAICompatibleVisionBackend("https://vlm.example.com/v1", api_key="vk", model="gpt-4o")
    res = b.ask("what is this", b"imagedata")
    assert res.text == "42"
    assert res.prompt_tokens == 9
    assert FakeClient.last_headers["Authorization"] == "Bearer vk"
    payload = FakeClient.last_json
    part = payload["messages"][0]["content"][1]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


class DummySettings:
    vision_base_url = "http://127.0.0.1:11434"
    vision_model = "minicpm-v:latest"
    vision_temperature = 0.0
    vision_key = None


class DummySettingsOpenAI:
    vision_base_url = "https://vlm.example.com/v1"
    vision_model = "gpt-4o"
    vision_temperature = 0.0
    vision_key = "vk"


def test_build_vision_backend_ollama():
    assert isinstance(build_vision_backend(DummySettings()), OllamaVisionBackend)


def test_build_vision_backend_openai():
    assert isinstance(build_vision_backend(DummySettingsOpenAI()), OpenAICompatibleVisionBackend)

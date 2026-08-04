"""Shared fixtures: fake backends + a tiny real PNG for session tests."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from deepsight.backends import ReasoningResult, SearchResult, ToolCall, VisionResult


def tiny_png_bytes(size: int = 16) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (120, 40, 200)).save(buf, format="PNG")
    return buf.getvalue()


def tiny_png_data_url(size: int = 16) -> str:
    return "data:image/png;base64," + base64.b64encode(tiny_png_bytes(size)).decode()


@pytest.fixture
def png_bytes() -> bytes:
    return tiny_png_bytes()


@pytest.fixture
def png_data_url() -> str:
    return tiny_png_data_url()


class FakeReasoning:
    """Scripted reasoning backend: returns canned results per call."""

    def __init__(self, script: list[ReasoningResult]) -> None:
        self.script = list(script)
        self.calls = 0
        self.last_messages: list[dict] | None = None
        self.last_tools: list[dict] | None = None
        self.last_max_tokens: int | None = None

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ReasoningResult:
        if self.calls >= len(self.script):
            raise AssertionError("fake reasoning script exhausted")
        self.last_messages = messages
        self.last_tools = tools
        self.last_max_tokens = max_tokens
        result = self.script[self.calls]
        self.calls += 1
        return result


@pytest.fixture
def fake_reasoning():
    def _make(script: list[ReasoningResult]) -> FakeReasoning:
        return FakeReasoning(script)

    return _make


class FakeVision:
    """Vision backend returning canned text + fixed usage."""

    def __init__(self, text: str = "answer", prompt: int = 5, completion: int = 2,
                 boxes_data: list[dict] | None = None) -> None:
        self.text = text
        self.prompt = prompt
        self.completion = completion
        self.calls = 0
        self.box_calls = 0
        self.last_prompt: str | None = None
        self.last_bytes: bytes | None = None
        self.last_max_output_tokens: int | None = None
        self.boxes_data = boxes_data if boxes_data is not None else []

    def ask(
        self, prompt: str, image_bytes: bytes, max_output_tokens: int | None = None
    ) -> VisionResult:
        self.calls += 1
        self.last_prompt = prompt
        self.last_bytes = image_bytes
        self.last_max_output_tokens = max_output_tokens
        return VisionResult(self.text, self.prompt, self.completion)

    def boxes(self, image_bytes: bytes) -> list[dict]:
        self.box_calls += 1
        self.last_bytes = image_bytes
        return list(self.boxes_data)


@pytest.fixture
def fake_vision():
    def _make(text: str = "answer", prompt: int = 5, completion: int = 2,
              boxes_data: list[dict] | None = None) -> FakeVision:
        return FakeVision(text=text, prompt=prompt, completion=completion,
                         boxes_data=boxes_data)

    return _make


def simple_result(
    content: str,
    prompt: int = 10,
    completion: int = 3,
    tool_calls: list[ToolCall] | None = None,
) -> ReasoningResult:
    return ReasoningResult(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        prompt_tokens=prompt,
        completion_tokens=completion,
    )


class FakeSearchBackend:
    """Canned search backend for test."""

    def __init__(self, results: list[SearchResult] | None = None,
                 fetch_text: str | None = None) -> None:
        self.results = results or []
        self.fetch_text = fetch_text
        self.search_calls = 0
        self.fetch_calls = 0

    def search(self, query: str, count: int = 3) -> list[SearchResult]:
        self.search_calls += 1
        return self.results[:count]

    def fetch(self, url: str) -> str | None:
        self.fetch_calls += 1
        return self.fetch_text


@pytest.fixture
def fake_search():
    def _make(results: list[SearchResult] | None = None,
              fetch_text: str | None = None) -> FakeSearchBackend:
        return FakeSearchBackend(results=results, fetch_text=fetch_text)
    return _make

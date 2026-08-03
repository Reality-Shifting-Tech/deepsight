"""Shared fixtures: fake backends + a tiny real PNG for session tests."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from deepsight.backends import ReasoningResult, ToolCall, VisionResult


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

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ReasoningResult:
        if self.calls >= len(self.script):
            raise AssertionError("fake reasoning script exhausted")
        self.last_messages = messages
        self.last_tools = tools
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

    def __init__(self, text: str = "answer", prompt: int = 5, completion: int = 2) -> None:
        self.text = text
        self.prompt = prompt
        self.completion = completion
        self.calls = 0
        self.last_prompt: str | None = None
        self.last_bytes: bytes | None = None

    def ask(self, prompt: str, image_bytes: bytes) -> VisionResult:
        self.calls += 1
        self.last_prompt = prompt
        self.last_bytes = image_bytes
        return VisionResult(self.text, self.prompt, self.completion)


@pytest.fixture
def fake_vision():
    def _make(text: str = "answer", prompt: int = 5, completion: int = 2) -> FakeVision:
        return FakeVision(text=text, prompt=prompt, completion=completion)

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

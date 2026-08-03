"""Backend clients: the reasoning model (text-only LLM) and the vision
model (VLM that answers targeted looks at image regions).

Both backends are plain HTTP clients with zero heavy dependencies
(no torch, no transformers), so the proxy installs in seconds.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx

# ---------------------------------------------------------------------------
# Reasoning backend — any OpenAI-compatible chat-completions endpoint
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    """A tool invocation emitted by the reasoning model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ReasoningResult:
    """One reasoning-model turn: either a final answer or tool calls."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class ReasoningBackend:
    """OpenAI-compatible chat backend (DeepSeek, OpenAI, local, anything)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ReasoningResult:
        """Run one chat turn; return content + any tool calls."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        tool_calls_raw = message.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=tc.get("id", "call_" + str(i)),
                name=tc["function"]["name"],
                arguments=_safe_json(tc["function"].get("arguments", "{}")),
            )
            for i, tc in enumerate(tool_calls_raw)
        ]
        return ReasoningResult(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


def _safe_json(raw: str) -> dict[str, Any]:
    """Parse tool-call arguments, tolerating malformed JSON."""
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Vision backend — Ollama VLM (minicpm-v default) or any OpenAI-compatible VLM
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VisionResult:
    """One vision-model answer plus its token usage."""

    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class OllamaVisionBackend:
    """Vision via Ollama's native API (minicpm-v, llava, qwen-vl...)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "minicpm-v:latest",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def ask(self, prompt: str, image_bytes: bytes) -> VisionResult:
        """Ask the VLM a question about an image, returning text + usage."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode()],
                }
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        msg = data.get("message", {})
        return VisionResult(
            text=msg.get("content", "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
        )


class OpenAICompatibleVisionBackend:
    """Vision via any OpenAI-compatible VLM (sensenova, gpt-4o, qwen-vl...)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "sensenova-6.7-flash-lite",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def ask(self, prompt: str, image_bytes: bytes) -> VisionResult:
        """Ask the VLM a question about an image, returning text + usage."""
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        usage = data.get("usage", {})
        return VisionResult(
            text=data["choices"][0]["message"].get("content", "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


VisionBackend = Literal["ollama", "openai"]


def build_vision_backend(settings: Any) -> OllamaVisionBackend | OpenAICompatibleVisionBackend:
    """Construct the configured vision backend.

    The backend kind is chosen by ``DEEPSIGHT_VISION_BASE_URL``: an
    ``http://...:11434`` address is treated as Ollama; anything else is
    treated as an OpenAI-compatible endpoint.
    """
    base = settings.vision_base_url
    if "11434" in base or "ollama" in base:
        return OllamaVisionBackend(
            base_url=base,
            model=settings.vision_model,
            temperature=settings.vision_temperature,
        )
    return OpenAICompatibleVisionBackend(
        base_url=base,
        api_key=settings.vision_key,
        model=settings.vision_model,
        temperature=settings.vision_temperature,
    )

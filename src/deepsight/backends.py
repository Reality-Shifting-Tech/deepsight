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
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_cache_hit_tokens": self.cache_hit_tokens,
            "prompt_cache_miss_tokens": self.cache_miss_tokens,
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
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> ReasoningResult:
        """Run one chat turn; return content + any tool calls.

        ``max_tokens`` overrides the constructor default per call (used by
        the orchestrator to budget tool rounds).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
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
            cache_hit_tokens=int(usage.get("prompt_cache_hit_tokens", 0)),
            cache_miss_tokens=int(usage.get("prompt_cache_miss_tokens", 0)),
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
        max_output_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def ask(
        self,
        prompt: str,
        image_bytes: bytes,
        max_output_tokens: int | None = None,
    ) -> VisionResult:
        """Ask the VLM a question about an image, returning text + usage.

        ``max_output_tokens`` overrides the constructor default per call
        (the sketch gets no cap; tool observations get a small one).
        """
        options: dict[str, Any] = {"temperature": self.temperature}
        cap = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        if cap is not None:
            options["num_predict"] = cap
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
            "options": options,
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
        max_output_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def ask(
        self,
        prompt: str,
        image_bytes: bytes,
        max_output_tokens: int | None = None,
    ) -> VisionResult:
        """Ask the VLM a question about an image, returning text + usage."""
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        payload: dict[str, Any] = {
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
        cap = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        if cap is not None:
            payload["max_tokens"] = cap
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


class NativeVisionBackend:
    """Apple Vision framework eyes via the compiled ``vision_eyes`` binary.

    Zero model downloads, zero tokens, zero GPU: OCR + saliency only, via
    macOS Vision.framework. Best for text-bearing images (charts, docs,
    UI, screenshots). The prompt is advisory; the eyes always emit OCR.
    """

    def __init__(self, bin_path: str, timeout: float = 60.0) -> None:
        self.bin_path = bin_path
        self.timeout = timeout

    @staticmethod
    def _parse_stdout(raw: str) -> str:
        ocr: list[str] = []
        saliency: list[str] = []
        for line in raw.splitlines():
            if line.startswith("  "):
                ocr.append(line.strip())
            elif "salient objects" in line:
                saliency.append(line.strip())
        parts: list[str] = []
        if ocr:
            parts.append("OCR text:\n" + "\n".join(ocr))
        if saliency:
            parts.append("\n".join(saliency))
        return "\n".join(parts).strip() or "(no text detected)"

    def ask(
        self,
        prompt: str,
        image_bytes: bytes,
        max_output_tokens: int | None = None,
    ) -> VisionResult:
        import os
        import subprocess
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".png")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(image_bytes)
            proc = subprocess.run(
                [self.bin_path, tmp],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                return VisionResult(
                    f"vision_eyes error: {proc.stderr.strip()[:200]}", 0, 0
                )
            text = self._parse_stdout(proc.stdout)
        finally:
            os.unlink(tmp)
        return VisionResult(text=text, prompt_tokens=0, completion_tokens=0)


VisionBackend = Literal["ollama", "openai", "native"]


def build_vision_backend(
    settings: Any,
) -> OllamaVisionBackend | OpenAICompatibleVisionBackend | NativeVisionBackend:
    """Construct the configured vision backend.

    The backend kind is chosen by ``DEEPSIGHT_VISION_BACKEND``: ``native``
    uses the Apple Vision framework binary (zero tokens, zero downloads);
    otherwise an ``http://...:11434`` address is treated as Ollama and
    anything else as an OpenAI-compatible endpoint.
    """
    if getattr(settings, "vision_backend", "ollama") == "native":
        return NativeVisionBackend(bin_path=settings.vision_bin)
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

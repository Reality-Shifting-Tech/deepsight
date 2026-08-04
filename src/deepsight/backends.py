"""Backend clients: the reasoning model (text-only LLM) and the eyes.

The eyes are the device's own vision framework (Apple Vision via the compiled
``vision_eyes`` binary): zero tokens, zero model downloads, zero GPU. The
reasoning backend is an optional OpenAI-compatible chat endpoint used by the
vision-session loop. Both are plain HTTP/subprocess clients with zero heavy
dependencies (no torch, no transformers).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

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
# Vision backend — the device's own eyes (Apple Vision framework)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VisionResult:
    """One vision answer. Native eyes burn zero tokens."""

    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class NativeVisionBackend:
    """Apple Vision framework eyes via the compiled ``vision_eyes`` binary.

    Zero model downloads, zero tokens, zero GPU: OCR, scene classification,
    saliency, face/human/rectangle detection, all on-device. The prompt is
    advisory; the eyes always emit OCR + scene + counts.
    """

    def __init__(self, bin_path: str, timeout: float = 60.0) -> None:
        self.bin_path = bin_path
        self.timeout = timeout

    @staticmethod
    def _parse_stdout(raw: str) -> str:
        ocr: list[str] = []
        scene: list[str] = []
        saliency: list[str] = []
        counts: list[str] = []
        pose: list[str] = []
        face_attrs: list[str] = []
        face_quality: list[str] = []
        animals: list[str] = []
        colors: list[str] = []
        sports: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if line.startswith("  "):
                ocr.append(stripped)
            elif stripped.startswith("scene:"):
                scene.append(stripped.removeprefix("scene:").strip())
            elif "salient objects" in stripped:
                saliency.append(stripped)
            elif stripped.startswith("face attr:"):
                face_attrs.append(stripped.removeprefix("face attr:").strip())
            elif stripped.startswith("face quality:"):
                face_quality.append(stripped.removeprefix("face quality:").strip())
            elif stripped.startswith("pose:"):
                pose.append(stripped.removeprefix("pose:").strip())
            elif stripped.startswith("pose "):
                pose.append(stripped.removeprefix("pose ").strip())
            elif stripped.startswith("animals:"):
                animals.append(stripped.removeprefix("animals:").strip())
            elif stripped.startswith("sports:"):
                sports.append(stripped.removeprefix("sports:").strip())
            elif stripped.startswith("colors:"):
                colors.append(stripped.removeprefix("colors:").strip())
            elif stripped.startswith(("faces:", "humans:", "rectangles:")):
                counts.append(stripped)
        parts: list[str] = []
        if ocr:
            parts.append("OCR text:\n" + "\n".join(ocr))
        if scene:
            parts.append("Scene: " + scene[0])
        if sports:
            parts.append("Sports: " + "; ".join(sports))
        if saliency:
            parts.append("\n".join(saliency))
        if counts:
            parts.append("\n".join(counts))
        if pose:
            parts.append("Pose: " + "; ".join(pose))
        if face_attrs:
            parts.append("Face attrs: " + "; ".join(face_attrs))
        if face_quality:
            parts.append("Face quality: " + "; ".join(face_quality))
        if animals:
            parts.append("Animals: " + "; ".join(animals))
        if colors:
            parts.append("Colors: " + "; ".join(colors))
        return "\n".join(parts).strip() or "(no text or scene detected)"

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

"""Vision-session orchestrator.

The orchestrator runs the interactive loop that gives a text-only
reasoning model vision:

1. Load the image from the request (base64 data URL or remote URL).
2. Produce a compact scene sketch via the vision model.
3. Inject the sketch + vision tool definitions into the context.
4. Let the reasoning model answer or issue ``look``/``ocr``/``zoom``
   tool calls; each call is answered with a targeted vision pass.
5. Loop until the model produces a final answer (bounded by
   ``max_look_rounds``), then return the answer with summed token usage.

Models without native tool support can emit ``[LOOK x,y,w,h]`` markers
in plain text; the orchestrator parses and executes those identically.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image

from .backends import (
    OllamaVisionBackend,
    OpenAICompatibleVisionBackend,
    ReasoningBackend,
)
from .cache import PerceptionCache
from .perception import TOOL_DEFINITIONS, Perception

VisionBackendType = OllamaVisionBackend | OpenAICompatibleVisionBackend

SYSTEM_PROMPT = (
    "You are DeepSight, an AI with interactive vision. You can inspect images by calling tools.\n\n"
    "An image is attached to this conversation. A scene sketch (JSON) describing it was generated "
    "first. Use the sketch as your primary source: answer directly from it whenever it contains "
    "the information you need. Use the tools to LOOK at specific regions when you need detail: "
    "read text with `ocr`, zoom into small areas with `zoom`, inspect regions with `look`. "
    "Be surgical: ask only for what you actually need, then answer concisely. You may issue "
    "MULTIPLE tool calls in a single turn (for example, look at several regions at once); batch "
    "them rather than inspecting one region per turn.\n\n"
    "Answer the user's question as soon as you have enough information. Counting questions: "
    "prefer the sketch's object list; only look at individual items to disambiguate.\n\n"
    "When you have enough information, answer the user's question directly and stop calling tools."
)

MARKER_RE = re.compile(r"\[LOOK\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", re.IGNORECASE)


@dataclass(slots=True)
class SessionResult:
    """Outcome of one vision session: the answer plus full token usage."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    rounds: int
    tool_calls: int
    cache_hits: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _tool_defs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in TOOL_DEFINITIONS
    ]


def load_image(image_url: str) -> Image.Image:
    """Load an image from a data URL or an http(s) URL."""
    if image_url.startswith("data:"):
        header, _, b64 = image_url.partition(",")
        raw = base64.b64decode(b64)
    elif image_url.startswith(("http://", "https://")):
        with httpx.Client(timeout=60) as client:
            resp = client.get(image_url)
            resp.raise_for_status()
            raw = resp.content
    else:
        raw = base64.b64decode(image_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _usage_sum(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


class Orchestrator:
    """Runs the vision-session loop for a single request."""

    def __init__(
        self,
        reasoning: ReasoningBackend,
        vision: VisionBackendType,
        cache: PerceptionCache | None = None,
        max_look_rounds: int = 5,
        sketch_enabled: bool = True,
    ) -> None:
        self.reasoning = reasoning
        self.perception = Perception(vision, cache, sketch_enabled=sketch_enabled)
        self.max_look_rounds = max_look_rounds

    # -- main entry --------------------------------------------------------------

    def run(self, image_url: str, user_text: str) -> SessionResult:
        """Execute a full vision session; returns the answer + usage."""
        image = load_image(image_url)

        # 1. sketch
        sketch = self.perception.sketch(image)
        sketch_block = f"\nScene sketch:\n{sketch}\n" if sketch else "\n(no sketch)\n"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + sketch_block},
            {"role": "user", "content": user_text},
        ]

        prompt_tokens = self.perception.total_prompt_tokens
        completion_tokens = self.perception.total_completion_tokens
        rounds = 0
        tool_calls_total = 0
        last_content = ""

        while rounds < self.max_look_rounds:
            rounds += 1
            result = self.reasoning.chat(messages, tools=_tool_defs())
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            if result.content and result.content.strip():
                last_content = result.content

            if result.wants_tools:
                # execute all pending tool calls, then feed results back
                tool_messages: list[dict[str, Any]] = []
                for call in result.tool_calls:
                    tool_calls_total += 1
                    observation = self.perception.execute(call.name, call.arguments, image)
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": observation,
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": _json_dumps(call.arguments),
                                },
                            }
                            for call in result.tool_calls
                        ],
                    }
                )
                messages.extend(tool_messages)
                continue

            # plain text: check for [LOOK ...] text-marker fallback
            markers = MARKER_RE.findall(result.content)
            if markers:
                tool_calls_total += len(markers)
                observations: list[str] = []
                for x, y, w, h in markers:
                    obs = self.perception.execute(
                        "look",
                        {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                        image,
                    )
                    observations.append(obs)
                messages.append({"role": "assistant", "content": result.content})
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool results:\n" + "\n".join(observations),
                    }
                )
                continue

            return SessionResult(
                content=result.content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                rounds=rounds,
                tool_calls=tool_calls_total,
                cache_hits=self.perception.cache_hits,
            )

        # hit the round cap: return the model's last real content if any
        final = last_content or "[deepsight] reached max look rounds without a final answer."
        return SessionResult(
            content=final,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            rounds=rounds,
            tool_calls=tool_calls_total,
            cache_hits=self.perception.cache_hits,
        )


def _json_dumps(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args)

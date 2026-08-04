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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image

from .backends import NativeVisionBackend, ReasoningBackend
from .cache import PerceptionCache
from .perception import TOOL_DEFINITIONS, Perception

VisionBackendType = NativeVisionBackend

SYSTEM_PROMPT = (
    "You are DeepSight, an AI with interactive vision. You can inspect images by calling tools.\n\n"
    "An image is attached to this conversation. A scene sketch describing it was generated "
    "first. Use the sketch as your primary source: answer directly from it whenever it contains "
    'the information you need. If the sketch has an "answer" field, respond with that value '
    "immediately and stop: do not re-derive it, do not call tools. "
    "The sketch may include OCR text (jersey/logo/UI text), scene labels, body-pose counts, "
    "face attributes, a color palette, and object counts. "
    "Use the tools only when the sketch is missing or ambiguous: "
    "read text with `ocr`, zoom into small areas with `zoom`, inspect regions with `look`, and "
    "count objects with `count` (one call counts them all; do not count one by one). "
    "Be surgical: ask only for what you actually need, then answer. You may issue "
    "MULTIPLE tool calls in a single turn (for example, look at several regions at once); batch "
    "them rather than inspecting one region per turn.\n\n"
    "Answer the user's question as soon as you have enough information. Counting questions: "
    'prefer the sketch\'s object list or "answer" field; only call `count` '
    "when the sketch is ambiguous.\n\n"
    "When you have enough information, answer the user's question directly and stop calling "
    "tools.\n\n"
    "ANSWER STYLE (mandatory): respond like a confident multimodal assistant, in fluent "
    "conversational prose, as if you were telling a person what is in the image. Lead with "
    "your direct answer or best interpretation, then fill in what else you noticed. "
    "Interpret, don't just inventory: name the person, team, brand, product, landmark, or "
    "object when the evidence supports it (jersey text + team colors + sport = name the player "
    "and team; a logo + product type = name the brand). Use the sketch's signals plus your own "
    "world knowledge to draw that conclusion, and state it with confidence rather than hedging "
    "when the signals are consistent. Only hedge ('looks like', 'probably') when the evidence "
    "is genuinely ambiguous or contradictory. No markdown, no bullet lists, "
    "no JSON, no robotic dumps; vary sentence length like a human. Keep it reasonably "
    "concise: a short paragraph or two for casual questions, a bit more when asked. Never "
    "mention the sketch, tools, OCR, or the vision pipeline; just talk about the image. "
    'If the sketch has an "answer" field, use that value as ground truth (it may be an '
    "exact count or value) and state it naturally inside your response."
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
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

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


def _usage_sum(
    prompt: int, completion: int, cache_hit: int = 0, cache_miss: int = 0
) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
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
        tool_round_max_tokens: int = 1024,
        final_max_tokens: int | None = None,
        vision_tool_max_tokens: int = 64,
    ) -> None:
        self.reasoning = reasoning
        self.perception = Perception(
            vision,
            cache,
            sketch_enabled=sketch_enabled,
            tool_max_output_tokens=vision_tool_max_tokens,
        )
        self.max_look_rounds = max_look_rounds
        self.tool_round_max_tokens = tool_round_max_tokens
        self.final_max_tokens = final_max_tokens

    # -- main entry --------------------------------------------------------------

    def run(
        self,
        image_url: str,
        user_text: str,
        on_event: Callable[[str], None] | None = None,
    ) -> SessionResult:
        """Execute a full vision session; returns the answer + usage.

        ``on_event`` (optional) receives a short human-readable status string
        at each milestone ("👁️ viewing image...", "✏️ sketching...",
        "🔍 looking...", "✅ answering...") so clients can show live
        progress while the session runs.
        """
        image = load_image(image_url)
        _emit(on_event, "👁️ viewing image...")

        # 1. sketch
        _emit(on_event, "✏️ sketching scene...")
        sketch = self.perception.sketch(image, question=user_text)
        sketch_block = f"\nScene sketch:\n{sketch}\n" if sketch else "\n(no sketch)\n"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + sketch_block},
            {"role": "user", "content": user_text},
        ]

        prompt_tokens = self.perception.total_prompt_tokens
        completion_tokens = self.perception.total_completion_tokens
        cache_hit_tokens = 0
        cache_miss_tokens = 0
        rounds = 0
        tool_calls_total = 0
        last_content = ""

        while rounds < self.max_look_rounds:
            rounds += 1
            result = self.reasoning.chat(
                messages,
                tools=_tool_defs(),
                max_tokens=self.tool_round_max_tokens,
            )
            if not result.tool_calls and result.finish_reason == "length":
                # the output budget truncated the model mid-thought; retry
                # this round once with the final budget so no turn is wasted
                _emit(on_event, "🔁 widening output budget...")
                result = self.reasoning.chat(
                    messages,
                    tools=_tool_defs(),
                    max_tokens=self.final_max_tokens,
                )
            prompt_tokens += result.prompt_tokens
            completion_tokens += result.completion_tokens
            cache_hit_tokens += result.cache_hit_tokens
            cache_miss_tokens += result.cache_miss_tokens
            if result.content and result.content.strip():
                last_content = result.content

            if result.wants_tools:
                # execute all pending tool calls, then feed results back
                tool_messages: list[dict[str, Any]] = []
                for call in result.tool_calls:
                    tool_calls_total += 1
                    _emit(on_event, f"🔍 looking ({call.name})...")
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

            if not result.content.strip():
                # still nothing usable (e.g. truncated again); loop again
                # rather than answering with an empty string
                continue

            _emit(on_event, "✅ answering...")
            return SessionResult(
                content=result.content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                rounds=rounds,
                tool_calls=tool_calls_total,
                cache_hits=self.perception.cache_hits,
                cache_hit_tokens=cache_hit_tokens,
                cache_miss_tokens=cache_miss_tokens,
            )

        # hit the round cap: force a final answer with one last no-tools call
        _emit(on_event, "✅ answering...")
        messages.append(
            {
                "role": "user",
                "content": (
                    "Tool budget exhausted. Do not call any more tools. Give "
                    "your final answer to the question now, based only on "
                    "everything you have observed so far."
                ),
            }
        )
        final = self.reasoning.chat(
            messages,
            tools=None,
            max_tokens=self.final_max_tokens,
        )
        prompt_tokens += final.prompt_tokens
        completion_tokens += final.completion_tokens
        cache_hit_tokens += final.cache_hit_tokens
        cache_miss_tokens += final.cache_miss_tokens
        final_content = (final.content or last_content or "").strip()
        if not final_content:
            final_content = "[deepsight] reached max look rounds without a final answer."
        return SessionResult(
            content=final_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            rounds=rounds + 1,
            tool_calls=tool_calls_total,
            cache_hits=self.perception.cache_hits,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
        )


def _json_dumps(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args)


def _emit(on_event: Callable[[str], None] | None, message: str) -> None:
    if on_event is not None:
        on_event(message)

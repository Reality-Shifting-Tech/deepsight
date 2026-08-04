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

from .backends import ComputerUseBackend, NativeVisionBackend, ReasoningBackend, SearchBackend
from .cache import PerceptionCache
from .perception import TOOL_DEFINITIONS, Perception, ToolDefinition

ACTION_TOOL_DEFINITIONS = [
    ToolDefinition(
        name="click",
        description=(
            "Click at a position on screen. Coordinates are percentages "
            "(0-100) of the screen dimensions for consistency with the "
            "locate tool — use locate first to find an object's position, "
            "then click on it. Requires Accessibility permission on macOS."
        ),
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "left edge, % of screen width"},
                "y": {"type": "number", "description": "top edge, % of screen height"},
            },
            "required": ["x", "y"],
        },
    ),
    ToolDefinition(
        name="type",
        description=(
            "Type text into the currently focused input field. Use click "
            "first to focus the right field, then type. Supports any text "
            "including spaces, punctuation, and special characters."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the text to type"},
            },
            "required": ["text"],
        },
    ),
    ToolDefinition(
        name="key",
        description=(
            "Press a keyboard key or combo. Use for keyboard shortcuts "
            "like 'cmd+s' (save), 'return', 'tab', 'escape', 'ctrl+c', "
            "'cmd+shift+4' (screenshot). Single keys: 'return', 'tab', "
            "'escape', 'up', 'down', 'left', 'right', 'space', 'delete'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "keys": {"type": "string",
                         "description": "key or key combo, e.g. 'cmd+s', 'return', 'escape'"},
            },
            "required": ["keys"],
        },
    ),
    ToolDefinition(
        name="scroll",
        description=(
            "Scroll the active window. Defaults to 3 clicks down. Use "
            "negative clicks or 'up' direction to scroll upward."
        ),
        parameters={
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["down", "up"],
                    "description": "scroll direction",
                },
                "clicks": {
                    "type": "integer",
                    "description": "number of scroll clicks (default 3)",
                },
            },
        },
    ),
    ToolDefinition(
        name="open",
        description=(
            "Launch or switch to an application by name. Uses `open -a`. "
            "Examples: 'Terminal', 'Safari', 'Xcode', 'Finder', "
            "'/Applications/SomeApp.app'. If the app is already running, "
            "it is brought to front."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "application name (e.g. 'Terminal') or full path",
                },
            },
            "required": ["name"],
        },
    ),
    ToolDefinition(
        name="focus",
        description=(
            "Bring a window to front by matching its title. Use `apps` "
            "first to see window titles, then focus one. Example: "
            "focus('Terminal') focuses the first Terminal window."
        ),
        parameters={
            "type": "object",
            "properties": {
                "window": {
                    "type": "string",
                    "description": "substring of the window title to focus",
                },
            },
            "required": ["window"],
        },
    ),
    ToolDefinition(
        name="apps",
        description=(
            "List all visible running applications and their window "
            "titles. Use this to discover what's open before deciding "
            "which app or window to focus, click in, or capture."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
    ),
    ToolDefinition(
        name="window",
        description=(
            "Resize or reposition a window. Coordinates are percent of "
            "screen (0-100) matching click/locate convention. If no "
            "'name' is given, operates on the frontmost window. "
            "Examples: window(x=25, y=25, w=50, h=50) centers it; "
            "window(name='Terminal', w=80, h=90) resizes it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "optional window title substring to target",
                },
                "x": {"type": "number", "description": "left edge, % of screen width"},
                "y": {"type": "number", "description": "top edge, % of screen height"},
                "w": {"type": "number", "description": "width, % of screen width"},
                "h": {"type": "number", "description": "height, % of screen height"},
            },
        },
    ),
    ]

VisionBackendType = NativeVisionBackend

SYSTEM_PROMPT = (
    "You are DeepSight, an AI with interactive vision and desktop control. "
    "You can inspect images by calling tools.\n\n"
    "An image is attached to this conversation. A scene sketch describing it was generated "
    "first. Use the sketch as your primary source: answer directly from it whenever it contains "
    'the information you need. If the sketch has an "answer" field, respond with that value '
    "immediately and stop: do not re-derive it, do not call tools.\n\n"
    "The following VISION tools are always available for inspecting the image:\n"
    "  look, ocr, zoom, count — inspect regions and read text\n"
    "  locate — find an object by description and get its coordinates\n"
    "  ground — search the web to verify a fact (requires an API key)\n"
    "  capture — take a screenshot of the live screen\n"
    "  watch — monitor the screen over time for changes\n\n"
    "IMPORTANT: If you are analyzing a static image (not a live screen), "
    "do NOT call ground, capture, or watch — they are for live desktop use. "
    "For a static image, use look, ocr, zoom, count, or locate only.\n\n"
    "The sketch may include OCR text (jersey/logo/UI text), scene labels, body-pose counts, "
    "face attributes, a color palette, object counts, and a sports line naming detected "
    "sports/equipment concepts (e.g. `sports: baseball(0.7)`). "
    "If the sports line lists two or more DISTINCT sports, the photo shows a group of athletes "
    "from different sports, typically the same city's teams posing together, NOT fans of one "
    "team. Treat it as an athletes' group photo, state the multi-sport nature explicitly, and "
    "use OCR team names plus any city hints to name the teams.\n\n"
    "Use the vision tools only when the sketch is missing or ambiguous: "
    "read text with `ocr`, zoom into small areas with `zoom`, inspect regions with `look`, "
    "and count objects with `count` (one call counts them all; do not count one by one). "
    "Be surgical: ask only for what you actually need, then answer. You may issue "
    "MULTIPLE tool calls in a single turn (for example, look at several regions at once); batch "
    "them rather than inspecting one region per turn.\n\n"
    "Answer the user's question as soon as you have enough information. Counting questions: "
    "prefer the sketch's object list or \"answer\" field; only call `count` "
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
    "If the sketch has an \"answer\" field, use that value as ground truth (it may be an "
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


def _tool_defs(include_actions: bool = True) -> list[dict[str, Any]]:
    """Merge vision tools with action tools.

    ``include_actions=False`` excludes desktop-action tools (click, type,
    etc.) for vision-only sessions where no ComputerUseBackend is present.
    """
    all_tools = list(TOOL_DEFINITIONS)
    if include_actions:
        all_tools.extend(ACTION_TOOL_DEFINITIONS)
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in all_tools
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
        search_backend: SearchBackend | None = None,
        computer: ComputerUseBackend | None = None,
    ) -> None:
        self.reasoning = reasoning
        self.perception = Perception(
            vision,
            cache,
            sketch_enabled=sketch_enabled,
            tool_max_output_tokens=vision_tool_max_tokens,
            search_backend=search_backend,
        )
        self.computer = computer
        self.max_look_rounds = max_look_rounds
        self.tool_round_max_tokens = tool_round_max_tokens
        self.final_max_tokens = final_max_tokens

    # -- main entry --------------------------------------------------------------

    def run(
        self,
        image_url: str,
        user_text: str,
        on_event: Callable[[str], None] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> SessionResult:
        """Execute a full vision session; returns the answer + usage.
        at each milestone (``"👁️ viewing image..."``, ``"✏️ sketching..."``,
        ``"🔍 looking..."``, ``"✅ answering..."``) so clients can show live
        progress while the session runs.

        ``response_format`` (optional) is forwarded to the reasoning model's
        chat-completions call, enabling structured/JSON output. Pass
        ``{"type": "json_object"}`` or
        ``{"type": "json_schema", "json_schema": {...}}``.
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
                tools=_tool_defs(include_actions=self.computer is not None),
                max_tokens=self.tool_round_max_tokens,
                response_format=response_format,
            )
            if not result.tool_calls and result.finish_reason == "length":
                # the output budget truncated the model mid-thought; retry
                # this round once with the final budget so no turn is wasted
                _emit(on_event, "🔁 widening output budget...")
                result = self.reasoning.chat(
                    messages,
                    tools=_tool_defs(include_actions=self.computer is not None),
                    max_tokens=self.final_max_tokens,
                    response_format=response_format,
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
                action_names = {"click", "type", "key", "scroll", "open", "focus", "apps", "window"}
                for call in result.tool_calls:
                    tool_calls_total += 1
                    if call.name in action_names and self.computer is not None:
                        _emit(on_event, f"🖱️ {call.name}...")
                        observation = self.computer.execute(call.name, call.arguments)
                    elif call.name in action_names:
                        observation = (
                            f"{call.name}: computer use not available — "
                            "no ComputerUseBackend configured"
                        )
                    else:
                        _emit(on_event, f"🔍 looking ({call.name})...")
                        observation = self.perception.execute(
                            call.name, call.arguments, image
                        )
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
            response_format=response_format,
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

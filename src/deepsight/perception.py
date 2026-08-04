"""Perception: scene sketching and the vision tool protocol.

The orchestrator runs a *vision session*: the reasoning model (a
text-only LLM) is handed a compact scene sketch plus tool definitions,
and issues ``look`` / ``crop`` / ``ocr`` / ``zoom`` calls which the
vision model answers with *targeted* passes. One sketch (~60-120 tokens)
plus on-demand looks replaces the one-shot 300-500 token description —
that token saving is the project's efficiency claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .backends import OllamaVisionBackend, OpenAICompatibleVisionBackend
from .cache import PerceptionCache

VisionBackendType = OllamaVisionBackend | OpenAICompatibleVisionBackend

SKETCH_PROMPT = (
    "You are the perception module of a vision proxy. Analyze the image and "
    "produce a COMPACT scene inventory. Be terse; this sketch is injected "
    "into a reasoning model's context, so token efficiency matters.\n\n"
    "Output EXACTLY this JSON shape (no markdown, no prose):\n"
    '{"objects": ["brief list of visible objects/regions"], '
    '"text": ["visible text fragments, transcribed"], '
    '"layout": "one line describing spatial arrangement", '
    '"palette": ["dominant colors"], '
    '"anomalies": ["anything notable: errors, highlights, warnings"], '
    '"answer": "if the question below can be answered from the image, give '
    'the answer here directly, otherwise omit this key"}\n'
    "If a list is empty, output [].\n\n"
    "The reasoning model will be asked a QUESTION about this image. Make sure "
    "the sketch contains everything needed to answer it: exact counts of "
    "objects, exact text values, labels, axis values, and measurements. For "
    "counting questions, state the exact number of each countable object type "
    'in the objects list AND put the final count in "answer".'
)

TOOL_REGION_PROMPT = (
    "You are the perception module of a vision proxy. The reasoning model "
    "asked a targeted question about a REGION of the image. Look ONLY at the "
    "region shown and answer directly and tersely (1-2 sentences). Do not "
    "mention the crop or the framing."
)

TOOL_OCR_PROMPT = (
    "You are the perception module of a vision proxy. Transcribe ALL text "
    "visible in the region shown, exactly as written, preserving line breaks. "
    "Output only the transcription."
)

TOOL_COUNT_PROMPT = (
    "You are the perception module of a vision proxy. Count the number of "
    "objects matching the description given, visible in the image shown. "
    "Look carefully; count every instance. Output ONLY the integer count, "
    "with no other text."
)

LOOK_RE = re.compile(r"\[LOOK\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", re.IGNORECASE)


@dataclass(slots=True)
class ToolDefinition:
    """A vision tool exposed to the reasoning model."""

    name: str
    description: str
    parameters: dict[str, Any]


TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="look",
        description=(
            "Inspect a rectangular region of the image (x, y, width, height as "
            "percentages 0-100) and describe exactly what is there. Use for any "
            "question about a specific part of the image."
        ),
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "left edge, % of width"},
                "y": {"type": "number", "description": "top edge, % of height"},
                "w": {"type": "number", "description": "width, % of image width"},
                "h": {"type": "number", "description": "height, % of image height"},
            },
            "required": ["x", "y", "w", "h"],
        },
    ),
    ToolDefinition(
        name="ocr",
        description=(
            "Transcribe all text inside a rectangular region (percentages). Use "
            "when exact text matters: error messages, labels, UI text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "left edge, % of width"},
                "y": {"type": "number", "description": "top edge, % of height"},
                "w": {"type": "number", "description": "width, % of image width"},
                "h": {"type": "number", "description": "height, % of image height"},
            },
            "required": ["x", "y", "w", "h"],
        },
    ),
    ToolDefinition(
        name="zoom",
        description=(
            "Zoom into a rectangular region (percentages) for a closer look at "
            "small details. The region is upscaled before the vision pass."
        ),
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "left edge, % of width"},
                "y": {"type": "number", "description": "top edge, % of height"},
                "w": {"type": "number", "description": "width, % of image width"},
                "h": {"type": "number", "description": "height, % of image height"},
            },
            "required": ["x", "y", "w", "h"],
        },
    ),
    ToolDefinition(
        name="count",
        description=(
            "Count how many objects matching a description appear in the image "
            "(or a region of it). Describe the object precisely: color, shape, "
            "type. Returns a single integer. Use for counting questions instead "
            "of inspecting items one by one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "description of the objects to count",
                },
                "x": {"type": "number", "description": "left edge, % of width"},
                "y": {"type": "number", "description": "top edge, % of height"},
                "w": {"type": "number", "description": "width, % of image width"},
                "h": {"type": "number", "description": "height, % of image height"},
            },
            "required": ["what"],
        },
    ),
]


class Perception:
    """Encapsulates image handling, sketch generation, and tool execution.

    Every vision call is routed through :class:`PerceptionCache`, so
    repeated questions about the same region cost zero tokens.
    """

    def __init__(
        self,
        vision: VisionBackendType,
        cache: PerceptionCache | None = None,
        sketch_enabled: bool = True,
        tool_max_output_tokens: int | None = None,
    ) -> None:
        self.vision = vision
        self.cache = cache or PerceptionCache()
        self.sketch_enabled = sketch_enabled
        self.tool_max_output_tokens = tool_max_output_tokens
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.cache_hits = 0

    # -- image -----------------------------------------------------------------

    def _region_bytes(
        self, image: Any, x: float, y: float, w: float, h: float, zoom: bool = False
    ) -> tuple[bytes, tuple[int, int, int, int] | None]:
        """Crop a percentage region from a PIL image, optionally upscaling."""
        from PIL import Image

        img: Image.Image = image
        iw, ih = img.size
        box = (
            max(0, int(x / 100 * iw)),
            max(0, int(y / 100 * ih)),
            min(iw, int((x + w) / 100 * iw)),
            min(ih, int((y + h) / 100 * ih)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            box = (0, 0, iw, ih)
        crop = img.crop(box)
        if zoom and max(crop.size) < 512:
            scale = max(1, 512 // max(crop.size))
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        import io

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue(), box

    # -- calls ------------------------------------------------------------------

    def _ask(
        self,
        prompt: str,
        image_bytes: bytes,
        kind: str,
        region: tuple[int, int, int, int] | None,
    ) -> tuple[str, bool]:
        """Ask the vision model, honoring the cache. Returns (text, cache_hit).

        Tool observations are capped via ``tool_max_output_tokens`` so the
        vision model answers tersely; the scene sketch is never capped.
        """
        if self.cache is not None:
            hit = self.cache.get(self.cache.image_hash(image_bytes), region, kind)
            if hit is not None:
                self.cache_hits += 1
                return hit, True
        cap = self.tool_max_output_tokens if not kind.startswith("sketch") else None
        result = self.vision.ask(prompt, image_bytes, max_output_tokens=cap)
        self.total_prompt_tokens += result.prompt_tokens
        self.total_completion_tokens += result.completion_tokens
        if self.cache is not None:
            self.cache.put(self.cache.image_hash(image_bytes), region, kind, result.text)
        return result.text, False

    def sketch(self, image: Any, question: str | None = None) -> str:
        """Produce the compact scene inventory JSON (or '' if disabled)."""
        if not self.sketch_enabled:
            return ""
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        prompt = SKETCH_PROMPT
        if question:
            prompt = prompt + f"\nQuestion to prepare for: {question}"
        kind = f"sketch:{question}" if question else "sketch"
        text, _ = self._ask(prompt, buf.getvalue(), kind, None)
        return text

    # -- tools ------------------------------------------------------------------

    def execute(self, name: str, args: dict[str, Any], image: Any) -> str:
        """Execute one vision tool call against the image."""
        if name not in {"look", "ocr", "zoom", "count"}:
            return f"unknown tool: {name}"
        x = float(args.get("x", 0))
        y = float(args.get("y", 0))
        w = float(args.get("w", 100))
        h = float(args.get("h", 100))
        zoom = name == "zoom"
        region_bytes, box = self._region_bytes(image, x, y, w, h, zoom=zoom)
        if name == "ocr":
            text, _ = self._ask(TOOL_OCR_PROMPT, region_bytes, "ocr", box)
            return f"OCR of region ({x:.0f}%,{y:.0f}%,{w:.0f}%,{h:.0f}%): {text}"
        if name == "count":
            what = str(args.get("what", "objects"))
            prompt = TOOL_COUNT_PROMPT + f"\nCount: {what}"
            text, _ = self._ask(prompt, region_bytes, f"count:{what}", box)
            return f"count of '{what}' in region ({x:.0f}%,{y:.0f}%,{w:.0f}%,{h:.0f}%): {text}"
        prompt = TOOL_REGION_PROMPT + f"\nRegion: x={x:.0f}%, y={y:.0f}%, w={w:.0f}%, h={h:.0f}%."
        text, _ = self._ask(prompt, region_bytes, name, box)
        return f"{name} region ({x:.0f}%,{y:.0f}%,{w:.0f}%,{h:.0f}%): {text}"  # noqa: E501

    def parse_text_markers(self, content: str) -> list[tuple[str, dict[str, Any]]]:
        """Parse ``[LOOK x,y,w,h]`` markers from a tool-less model's output."""
        calls: list[tuple[str, dict[str, Any]]] = []
        for m in LOOK_RE.finditer(content):
            calls.append(
                (
                    "look",
                    {
                        "x": int(m.group(1)),
                        "y": int(m.group(2)),
                        "w": int(m.group(3)),
                        "h": int(m.group(4)),
                    },
                )
            )
        return calls

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }

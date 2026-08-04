"""Perception: scene sketching and the vision tool protocol.

The orchestrator runs a *vision session*: the reasoning model (a
text-only LLM) is handed a compact scene sketch plus tool definitions,
and issues ``look`` / ``ocr`` / ``zoom`` / ``count`` calls which the
vision model answers with *targeted* passes. One sketch (~60-120 tokens)
plus on-demand looks replaces the one-shot 300-500 token description —
that token saving is the project's efficiency claim.

Spatial grounding: the ``locate`` tool (added in v2.1 of the Apple Vision
binary) returns bounding-box coordinates for detected objects (text,
faces, humans, animals, rectangles, salient objects), enabling UI
automation and coordinate-aware agent actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .backends import NativeVisionBackend, SearchBackend
from .cache import PerceptionCache

VisionBackendType = NativeVisionBackend

SKETCH_PROMPT = (
    "You are the perception module of DeepSight. Analyze the image and "
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
    "You are the perception module of DeepSight. The reasoning model "
    "asked a targeted question about a REGION of the image. Look ONLY at the "
    "region shown and answer directly and tersely (1-2 sentences). Do not "
    "mention the crop or the framing."
)

TOOL_OCR_PROMPT = (
    "You are the perception module of DeepSight. Transcribe ALL text "
    "visible in the region shown, exactly as written, preserving line breaks. "
    "Output only the transcription."
)

TOOL_COUNT_PROMPT = (
    "You are the perception module of DeepSight. Count the number of "
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
    ToolDefinition(
        name="locate",
        description=(
            "Find the bounding box of an object matching a description in the "
            "image. Uses on-device Apple Vision detection (faces, humans, "
            "text, animals, rectangles, salient objects) — zero tokens, zero "
            "network. Returns the object's normalized bounding box coordinates "
            "(x, y, w, h as percentages 0-100) and confidence. Accepts a "
            "target description string. Examples: 'the logo', 'a person', "
            "'WEBSITES text', 'cat', 'the red button'. Returns coordinates "
            "the agent can use for targeted inspection, clicks, or UI "
            "automation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "description of the object to locate, e.g. 'a person', 'the main heading text', 'a cat'",
                },
                "x": {
                    "type": "number",
                    "description": "optional region left edge, % of width (omit for full image)",
                },
                "y": {
                    "type": "number",
                    "description": "optional region top edge, % of height",
                },
                "w": {
                    "type": "number",
                    "description": "optional region width, % of image width",
                },
                "h": {
                    "type": "number",
                    "description": "optional region height, % of image height",
                },
            },
            "required": ["what"],
        },
    ),
    ToolDefinition(
        name="ground",
        description=(
            "Search the web to verify a fact, identify an entity, or find "
            "current information. Returns search results with snippets, "
            "fetches the top matching page, and provides a verification "
            "summary against the question or claim. Powered by Brave Search "
            "— requires DEEPSIGHT_SEARCH_API_KEY. Returns 'grounding "
            "unavailable' when no API key is configured."
        ),
        parameters={
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "the claim, entity, or fact to ground in web search results",
                },
                "query": {
                    "type": "string",
                    "description": "optional specific search query (defaults to the 'what' text)",
                },
            },
            "required": ["what"],
        },
    ),
    ToolDefinition(
        name="capture",
        description=(
            "Capture the current screen (or a specific window/region) and "
            "analyze it. This gives you live, real-time vision of what is "
            "happening on screen — your own output, running apps, build "
            "results, game renders, websites, whatever is visible. Returns "
            "a full scene description with OCR text, detected objects, and "
            "any changes since the last capture. After capture, you can use "
            "other vision tools (look, ocr, zoom, count, locate) to inspect "
            "specific areas of the captured screen. Call this whenever you "
            "need to see what you or another program just produced."
        ),
        parameters={
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "optional region: 'screen' (full display, default), or a window title substring to capture a specific window",
                },
            },
            "required": [],
        },
    ),
    ToolDefinition(
        name="watch",
        description=(
            "Watch the screen over time: captures a series of screenshots "
            "and reports what changed. Use to monitor builds, watch "
            "animations, detect UI transitions, or wait for something to "
            "appear ('the button', 'the error message'). Uses perceptual "
            "hashing to skip identical frames — only full-analyzes frames "
            "where the screen content actually changes. Returns a timeline "
            "of what happened and when."
        ),
        parameters={
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "total duration to watch (default 10, max 60)",
                },
                "interval": {
                    "type": "number",
                    "description": "seconds between captures (default 1.0, min 0.5)",
                },
                "until": {
                    "type": "string",
                    "description": "optional: stop early when this text appears on screen (substring match on OCR)",
                },
            },
            "required": [],
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
        search_backend: SearchBackend | None = None,
    ) -> None:
        self.vision = vision
        self.cache = cache or PerceptionCache()
        self.sketch_enabled = sketch_enabled
        self.tool_max_output_tokens = tool_max_output_tokens
        self.search_backend = search_backend
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.cache_hits = 0
        self._captured_image: Any = None  # set by capture tool
        self._prev_hash: str = ""  # dhash of last captured frame
        self._prev_ocr: set[str] = set()  # OCR items from last capture

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

    @staticmethod
    def _match_box(box: dict[str, Any], what: str) -> float:
        """Score how well a detection box matches a description. Returns 0 or >0."""
        wl = what.lower().strip()
        label = box.get("label", "").lower().strip()
        typ = box.get("type", "").lower().strip()

        # direct label match
        if wl == label:
            return 1.0
        if label in wl or wl in label:
            return 0.9

        # type-level match: "person" -> human/face
        wl_t = wl.replace("person", "").replace("people", "").strip()
        if wl_t != wl and typ in ("human", "face"):
            return 0.8
        if "animal" in wl and typ == "animal":
            return 0.8
        if "text" in wl and typ == "text":
            return 0.8

        return 0.0

    def _locate(self, args: dict[str, Any], image: Any) -> str:
        """Execute the ``locate`` tool: find objects matching a description.

        Calls the device-native ``vision_eyes`` binary for detection boxes,
        then filters by the ``what`` description. Returns coordinates as
        percentages (0-100) for consistency with other tools.
        """
        what = str(args.get("what", ""))
        if not what:
            return "locate: no description provided"

        # optionally constrain to a region
        x = float(args.get("x", 0))
        y = float(args.get("y", 0))
        w = float(args.get("w", 100))
        h = float(args.get("h", 100))
        has_region = not (x == 0 and y == 0 and w == 100 and h == 100)

        # convert PIL image to PNG bytes for the binary
        import io

        if has_region:
            region_bytes, _ = self._region_bytes(image, x, y, w, h, zoom=False)
            img_bytes = region_bytes
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()

        all_boxes = self.vision.boxes(img_bytes)
        if not all_boxes:
            return "locate: no detections found in image"

        scored = []
        for box in all_boxes:
            score = self._match_box(box, what)
            if score > 0:
                scored.append((score, box))

        if not scored:
            types = set(b["type"] for b in all_boxes)
            return (
                f"locate: no match for '{what}'. "
                f"Available detections: {', '.join(sorted(types))}. "
                f"Try describing one of those types."
            )

        scored.sort(key=lambda x: (-x[0], -x[1].get("confidence", 0)))
        lines: list[str] = []
        for score, box in scored:
            pct_x = box["x"] * 100
            pct_y = box["y"] * 100
            pct_w = box["w"] * 100
            pct_h = box["h"] * 100
            label = box["label"]
            conf = box["confidence"]
            lines.append(
                f"  {label} ({conf:.0%}): "
                f"x={pct_x:.0f}% y={pct_y:.0f}% w={pct_w:.0f}% h={pct_h:.0f}%"
            )
        return "locate results for '" + what + "':\n" + "\n".join(lines[:10])

    def _ground(self, args: dict[str, Any]) -> str:
        """Execute the ``ground`` tool: search + fetch + summarize.

        Returns verification text with citations. Graceful degradation
        when no search API key is configured.
        """
        what = str(args.get("what", ""))
        if not what:
            return "ground: no claim or entity provided"

        if self.search_backend is None:
            return (
                "ground: unavailable — no search API key configured. "
                "Set DEEPSIGHT_SEARCH_API_KEY in your environment or .env."
            )

        query = str(args.get("query", what))
        results = self.search_backend.search(query, count=3)
        if not results:
            return (
                f"ground: no search results for '{query}'. "
                "The API key may be invalid, the search backend "
                "may be unreachable, or no results were returned."
            )

        lines: list[str] = [f"ground results for '{what}' (query: '{query}'):"]
        for idx, r in enumerate(results):
            match = "TOP MATCH" if idx == 0 else f"result {idx + 1}"
            lines.append(f"  {match}: {r.title}")
            lines.append(f"  url: {r.url}")
            lines.append(f"  snippet: {r.snippet[:300]}")
            if idx == 0:
                content = self.search_backend.fetch(r.url)
                if content:
                    lines.append(f"  page content ({len(content)} chars, first 600):")
                    lines.append(f"    {content[:600].strip()}")
            lines.append("")
        lines.append(f"fetched {len(results)} result(s). "
                     f"Top page body: {'available' if results and self.search_backend.fetch(results[0].url) else 'not fetched (error or unavailable)'}.")
        return "\n".join(lines)

    @staticmethod
    def _dhash(image: Any, hash_size: int = 8) -> str:
        """Compute a perceptual hash (difference hash) of a PIL Image.

        Returns a hex string. Images with the same content produce
        the same hash; small changes produce minor bit differences.
        """
        from PIL import Image

        img = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.get_flattened_data())
        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                left = pixels[row * (hash_size + 1) + col]
                right = pixels[row * (hash_size + 1) + col + 1]
                diff.append("1" if left < right else "0")
        return hex(int("".join(diff), 2))[2:]

    def _capture(self, args: dict[str, Any]) -> str:
        """Execute the ``capture`` tool: screenshot → analyze → return description.

        Captures the current screen using macOS screencapture, runs the
        full vision pipeline (sketch + OCR + detection), and stores the
        captured image so subsequent tool calls operate on the live screen.
        """
        import os
        import subprocess
        import sys
        import tempfile

        if sys.platform != "darwin":
            return "capture: only supported on macOS (screencapture not available)"

        region = str(args.get("region", "screen"))

        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            if region and region != "screen":
                result = subprocess.run(
                    [
                        "osascript", "-e",
                        'tell application "System Events" to '
                        f'get id of first window whose title contains "{region}"',
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    win_id = result.stdout.strip()
                    subprocess.run(
                        ["screencapture", "-l", win_id, "-x", tmp_path],
                        capture_output=True, timeout=10,
                    )
                else:
                    subprocess.run(
                        ["screencapture", "-x", tmp_path],
                        capture_output=True, timeout=10,
                    )
            else:
                subprocess.run(
                    ["screencapture", "-x", tmp_path],
                    capture_output=True, timeout=10,
                )

            if not os.path.getsize(tmp_path):
                return "capture: screenshot produced an empty image"

            from PIL import Image
            import io

            img = Image.open(tmp_path).convert("RGB")
            self._captured_image = img

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            sketch = self.sketch(img)
            boxes = self.vision.boxes(img_bytes) if hasattr(self.vision, "boxes") else []

            text_items: list[str] = [
                b["label"] for b in boxes
                if b["type"] == "text" and len(b["label"]) < 100
            ] if boxes else []

            parts = [f"screen captured ({img.width}x{img.height})"]
            if sketch:
                parts.append(f"scene: {sketch[:400]}")

            # Cross-capture diff: hash + OCR comparison
            cur_hash = self._dhash(img)
            if self._prev_hash and cur_hash == self._prev_hash:
                parts.append("screen: unchanged since last capture")
            elif self._prev_hash:
                cur_ocr = set(text_items)
                new_text = cur_ocr - self._prev_ocr
                gone_text = self._prev_ocr - cur_ocr
                changes = []
                if new_text:
                    changes.append(f"new text: {' | '.join(list(new_text)[:5])}")
                if gone_text:
                    changes.append(f"text disappeared: {' | '.join(list(gone_text)[:3])}")
                if changes:
                    parts.append("changes: " + "; ".join(changes))
                else:
                    parts.append("screen: content changed (layout/pixels different)")
                self._prev_ocr = cur_ocr
            else:
                self._prev_ocr = set(text_items)
            self._prev_hash = cur_hash

            if boxes:
                by_type: dict[str, list] = {}
                for b in boxes:
                    by_type.setdefault(b["type"], []).append(b)
                parts.append(f"detected: {', '.join(sorted(by_type.keys()))}")
                if text_items:
                    parts.append(f"OCR: {' | '.join(text_items[:8])}")

            return "\n".join(parts) or "capture: screen captured but no content detected"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _watch(self, args: dict[str, Any]) -> str:
        """Execute the ``watch`` tool: capture N frames over time, report changes."""
        import time

        seconds = min(float(args.get("seconds", 10)), 60)
        interval = max(float(args.get("interval", 1.0)), 0.5)
        until = str(args.get("until", "")).lower().strip()

        timeline: list[str] = []
        prev_hash = ""
        start = time.monotonic()
        deadline = start + seconds
        capture_num = 0
        found_until = False

        while time.monotonic() < deadline:
            t_elapsed = time.monotonic() - start
            # Capture using the same mechanism as _capture but without
            # storing the image (we analyze inline).
            import os as _os
            import subprocess as _sp
            import tempfile as _tf

            fd, tmp = _tf.mkstemp(suffix=".png")
            _os.close(fd)
            try:
                _sp.run(
                    ["screencapture", "-x", tmp],
                    capture_output=True, timeout=10,
                )
                if not _os.path.getsize(tmp):
                    continue

                from PIL import Image as _PImage
                import io as _io

                frame = _PImage.open(tmp).convert("RGB")
                cur_hash = self._dhash(frame)
                if cur_hash == prev_hash:
                    continue  # skip identical frames
                prev_hash = cur_hash

                # Full analysis for changed frames
                buf = _io.BytesIO()
                frame.save(buf, format="PNG")
                boxes = self.vision.boxes(buf.getvalue()) if hasattr(self.vision, "boxes") else []
                text_items = [
                    b["label"] for b in boxes
                    if b["type"] == "text" and len(b["label"]) < 100
                ] if boxes else []

                capture_num += 1
                entry = f"t={t_elapsed:.1f}s"
                if text_items:
                    entry += f" OCR: {' | '.join(text_items[:5])}"
                timeline.append(entry)

                # Check until condition
                if until and any(until in t.lower() for t in text_items):
                    timeline.append(f"→ stopped early at t={t_elapsed:.1f}s: found '{until}'")
                    found_until = True
                    break
            finally:
                try:
                    _os.unlink(tmp)
                except OSError:
                    pass

            # Sleep until next capture (respect deadline)
            remaining = deadline - time.monotonic()
            if remaining > 0 and not found_until:
                time.sleep(min(interval, remaining))

        if not timeline:
            return "watch: screen didn't change during the observation period"
        elapsed = time.monotonic() - start
        lines = [
            f"watched for {elapsed:.1f}s, {capture_num} unique frame(s):",
            *timeline,
        ]
        return "\n".join(lines)

    def execute(self, name: str, args: dict[str, Any], image: Any = None) -> str:
        """Execute one vision tool call.

        When ``capture`` has been called, all subsequent tools operate
        on the live captured screen instead of the original image.
        """
        if name == "capture":
            return self._capture(args)
        if name == "watch":
            return self._watch(args)
        if name == "ground":
            return self._ground(args)

        # Resolve active image: captured screen > explicit image argument
        active_image = self._captured_image if self._captured_image is not None else image
        if active_image is None:
            return "no image available — call capture first or provide an image"

        if name == "locate":
            return self._locate(args, active_image)
        if name not in {"look", "ocr", "zoom", "count"}:
            return f"unknown tool: {name}"
        x = float(args.get("x", 0))
        y = float(args.get("y", 0))
        w = float(args.get("w", 100))
        h = float(args.get("h", 100))
        zoom = name == "zoom"
        region_bytes, box = self._region_bytes(active_image, x, y, w, h, zoom=zoom)
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

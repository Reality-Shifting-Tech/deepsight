"""Backend clients: the reasoning model (text-only LLM) and the eyes.

The eyes are the device's own vision framework (Apple Vision via the compiled
``vision_eyes`` binary): zero tokens, zero model downloads, zero GPU. The
reasoning backend is an optional OpenAI-compatible chat endpoint used by the
vision-session loop. Both are plain HTTP/subprocess clients with zero heavy
dependencies (no torch, no transformers).

Spatial grounding: the Apple Vision binary now emits ``box:`` lines for
every detected object (text, faces, humans, animals, rectangles, salient
objects) with normalized coordinates, enabling the ``locate`` vision tool.
"""

from __future__ import annotations

import json
import subprocess
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
        response_format: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """Run one chat turn; return content + any tool calls.

        ``max_tokens`` overrides the constructor default per call (used by
        the orchestrator to budget tool rounds). ``response_format`` enables
        structured output (``{"type": "json_object"}`` or
        ``{"type": "json_schema", "json_schema": {...}}``)
        for schema-constrained responses.
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
        if response_format is not None:
            payload["response_format"] = response_format
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
class SearchResult:
    """One web search result with content for grounding."""

    url: str
    title: str
    snippet: str
    content: str = ""


class SearchBackend:
    """Web search for factual grounding via Brave Search API.

    Gated by api_key: when unset, all methods return empty/false results
    so the ground tool degrades gracefully for OSS users who haven't
    configured a key. Enable with ``DEEPSIGHT_SEARCH_API_KEY``.

    .. code::

        sb = SearchBackend(api_key="BSA-...")
        results = sb.search("who won the super bowl 2026")
        text = sb.fetch(results[0].url)

    """

    BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, count: int = 3) -> list[SearchResult]:
        """Search the web, return top results with snippets.

        Returns empty list when no API key is configured or when the
        search API is unreachable.
        """
        if not self.api_key:
            return []
        try:
            resp = httpx.get(
                self.BRAVE_SEARCH_URL,
                params={"q": query, "count": min(count, 10)},
                headers={"X-Subscription-Token": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, httpx.TimeoutException, ValueError):
            return []

        raw = (data.get("web") or {}).get("results") or []
        return [
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("description", ""),
            )
            for r in raw[:count]
        ]

    def fetch(self, url: str) -> str | None:
        """Fetch a URL and extract visible text content.

        Returns None on any error (timeout, bad status, etc.).
        """
        if not self.api_key:
            return None
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": "DeepSight/1.0 (grounding)"},
                timeout=self.timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException):
            return None

        # Simple text extraction: strip tags, collapse whitespace. For
        # serious use, consider readability-lxml or trafilatura.
        text = resp.text
        import re

        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]  # 8K chars is plenty for verification


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

    @staticmethod
    def _parse_boxes(stdout: str) -> list[dict[str, Any]]:
        """Parse ``box:`` lines from vision_eyes output into structured boxes.

        Apple Vision convention (bottom-left origin) is converted to
        top-left origin for the reasoning model. Returns::

            {"type": str, "confidence": float,
             "x": float, "y": float, "w": float, "h": float,
             "label": str}
        """
        boxes: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped.startswith("box:"):
                continue
            parts = stripped.split(":", 7)
            if len(parts) < 8:
                continue
            try:
                typ = parts[1]
                conf = float(parts[2])
                nx = float(parts[3])
                ny = float(parts[4])
                nw = float(parts[5])
                nh = float(parts[6])
                label = parts[7].strip()
            except (ValueError, IndexError):
                continue
            # Flip Y: Apple Vision bottom-left -> top-left
            top_left_y = 1.0 - ny - nh
            boxes.append({
                "type": typ,
                "confidence": conf,
                "x": nx,
                "y": top_left_y,
                "w": nw,
                "h": nh,
                "label": label,
            })
        return boxes

    def boxes(self, image_bytes: bytes) -> list[dict[str, Any]]:
        """Run the vision binary and return parsed detection boxes.

        Same subprocess as ``ask()`` but returns structured spatial data
        for the ``locate`` tool: what objects are where.
        """
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
                return []
            return self._parse_boxes(proc.stdout)
        finally:
            os.unlink(tmp)

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


def _screen_size() -> tuple[int, int]:
    """Return (width, height) of the main display via tkinter (stdlib)."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return (w, h)
    except Exception:
        return (1920, 1080)


class ComputerUseBackend:
    """macOS desktop automation via osascript (built-in) or cliclick (opt-in).

    Provides click, type, key-combo, and scroll actions. Cliclick offers
    more reliable cursor positioning — install with ``brew install cliclick``.
    Without cliclick, uses ``osascript`` (requires Accessibility permission
    in System Settings > Privacy & Security > Accessibility).
    """

    def __init__(self) -> None:
        self._has_cliclick = (
            subprocess.run(["which", "cliclick"], capture_output=True).returncode == 0
        )
        self._screen_size = _screen_size()

    def _pixels(self, x_pct: float, y_pct: float) -> tuple[int, int]:
        w, h = self._screen_size
        return (int(x_pct / 100 * w), int(y_pct / 100 * h))

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch one computer-use tool call."""
        if name == "click":
            return self._click(args)
        if name == "type":
            return self._type(args)
        if name == "key":
            return self._key(args)
        if name == "scroll":
            return self._scroll(args)
        if name == "open":
            return self._open(args)
        if name == "focus":
            return self._focus(args)
        if name == "apps":
            return self._apps(args)
        if name == "window":
            return self._window(args)
        return f"unknown action: {name}"

    def _click(self, args: dict[str, Any]) -> str:
        x_pct = float(args.get("x", 50))
        y_pct = float(args.get("y", 50))
        px, py = self._pixels(x_pct, y_pct)
        if self._has_cliclick:
            subprocess.run(
                ["cliclick", f"c:{px},{py}"],
                capture_output=True, timeout=10,
            )
        else:
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to click at {{{px}, {py}}}'],
                capture_output=True, timeout=10,
            )
        return f"clicked at ({x_pct:.0f}%, {y_pct:.0f}%) → ({px}, {py})"

    def _type(self, args: dict[str, Any]) -> str:
        text = str(args.get("text", ""))
        if not text:
            return "type: no text provided"
        display = text[:60] + ("..." if len(text) > 60 else "")
        if self._has_cliclick:
            subprocess.run(
                ["cliclick", f"t:{text}"],
                capture_output=True, timeout=30,
            )
        else:
            safe = text.replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "System Events" to keystroke "{safe}"'],
                capture_output=True, timeout=30,
            )
        return f"typed '{display}'"

    def _key(self, args: dict[str, Any]) -> str:
        keys = str(args.get("keys", ""))
        if not keys:
            return "key: no keys provided"

        KEY_MAP = {
            "return": 36, "enter": 36, "tab": 48, "escape": 53, "esc": 53,
            "space": 49, "delete": 51, "backspace": 51,
            "up": 126, "down": 125, "left": 123, "right": 124,
            "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
        }
        parts = keys.lower().split("+")
        key = parts[-1]
        modifiers = parts[:-1]

        # osascript modifier map
        osa_mods = {
            "cmd": "command down", "command": "command down",
            "shift": "shift down", "ctrl": "control down",
            "alt": "option down", "option": "option down",
        }

        if self._has_cliclick:
            cli_mods = {"cmd": "cmd", "ctrl": "ctrl", "alt": "alt", "shift": "shift",
                        "option": "alt", "command": "cmd"}
            if len(parts) == 1 and key in KEY_MAP:
                subprocess.run(["cliclick", f"kp:{key}"], capture_output=True, timeout=10)
            else:
                ms = [cli_mods.get(m, m) for m in modifiers]
                cmd = [f"kd:{m}" for m in ms] + [f"kp:{key}"] + [f"ku:{m}" for m in reversed(ms)]
                subprocess.run(["cliclick"] + cmd, capture_output=True, timeout=10)
        else:
            if len(parts) == 1 and key in KEY_MAP:
                code = KEY_MAP[key]
                subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to key code {code}'],
                    capture_output=True, timeout=10,
                )
            elif len(parts) == 1:
                subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to keystroke "{key}"'],
                    capture_output=True, timeout=10,
                )
            else:
                using = ", ".join(osa_mods.get(m, m) for m in modifiers)
                subprocess.run(
                    ["osascript", "-e",
                     f'tell application "System Events" to keystroke "{key}" using {{{using}}}'],
                    capture_output=True, timeout=10,
                )
        return f"pressed {keys}"

    def _scroll(self, args: dict[str, Any]) -> str:
        direction = str(args.get("direction", "down"))
        clicks = int(args.get("clicks", 3))
        if self._has_cliclick:
            subprocess.run(
                ["cliclick", f"w:{clicks}"],
                capture_output=True, timeout=10,
            )
        else:
            code = 124 if direction == "down" else 126
            subprocess.run(
                ["osascript", "-e",
                 f'repeat {clicks} times\n'
                 f'tell application "System Events" to key code {code}\n'
                 f'end repeat'],
                capture_output=True, timeout=30,
            )
        return f"scrolled {direction} {clicks} clicks"

    def _run_osa(self, script: str) -> tuple[int, str]:
        """Run an osascript one-liner, return (exit_code, stdout)."""
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15,
        )
        return (r.returncode, r.stdout.strip())

    def _open(self, args: dict[str, Any]) -> str:
        """Launch or activate an application by name."""
        name = str(args.get("name", "")).strip()
        if not name:
            return "open: no app name provided"
        try:
            subprocess.run(
                ["open", "-a", name],
                capture_output=True, timeout=15,
            )
            return f"opened {name}"
        except subprocess.TimeoutExpired:
            return f"open: timed out launching {name}"

    def _focus(self, args: dict[str, Any]) -> str:
        """Focus a window whose title contains the given substring."""
        title = str(args.get("window", "")).strip()
        if not title:
            return "focus: no window title provided"
        # Find the app owning the first matching window and activate it
        rc, out = self._run_osa(
            f'tell application "System Events"\n'
            f'  set matches to every window whose title contains "{title}"\n'
            f'  if (count of matches) > 0 then\n'
            f'    set win to item 1 of matches\n'
            f'    set appName to name of first process whose every window contains win\n'
            f'    tell application appName to activate\n'
            f'    return appName & ": " & name of win\n'
            f'  end if\n'
            f'  return "not found"\n'
            f'end tell'
        )
        if rc != 0 or "not found" in out:
            return f"focus: no window matching '{title}' found"
        return f"focused: {out}"

    def _apps(self, args: dict[str, Any]) -> str:
        """List running applications with their window titles."""
        rc, out = self._run_osa(
            'set lines to {}\n'
            'tell application "System Events"\n'
            '  set procs to every process whose background only is false\n'
            '  repeat with p in procs\n'
            '    set pname to name of p\n'
            '    set ws to every window of p\n'
            '    set end of lines to pname & " (" & (count of ws) & " windows)"\n'
            '    repeat with w in ws\n'
            '      set end of lines to "  - " & name of w\n'
            '    end repeat\n'
            '  end repeat\n'
            'end tell\n'
            'return lines as text'
        )
        if rc != 0:
            return "apps: could not retrieve app list"
        lines = [ln for ln in out.split("\n") if ln.strip()]
        if not lines:
            return "apps: no visible applications found"
        return "running applications:\n" + "\n".join(lines[:30])

    def _window(self, args: dict[str, Any]) -> str:
        """Resize or move a window. x/y/w/h in screen % coords."""
        name = str(args.get("name", "")).strip() or None
        has_x = "x" in args
        has_y = "y" in args
        has_w = "w" in args or "width" in args
        has_h = "h" in args or "height" in args

        if not (has_x or has_y or has_w or has_h):
            return "window: specify at least one of x, y, w, h"

        screen_w, screen_h = self._screen_size

        def pct(v: float, total: int) -> int:
            return int(v / 100 * total)

        tgt_line = (
            f'set tgt to first window whose title contains "{name}"'
            if name else
            'set tgt to window 1 of first process whose background only is false'
        )
        x_line = f"set bx to {pct(float(args['x']), screen_w)}" if has_x else ""
        y_line = f"set by to {pct(float(args['y']), screen_h)}" if has_y else ""
        w_val = args.get("w", args.get("width", 0))
        h_val = args.get("h", args.get("height", 0))
        w_line = f"set bw to {pct(float(w_val), screen_w)}" if has_w else ""
        h_line = f"set bh to {pct(float(h_val), screen_h)}" if has_h else ""

        script = (
            'tell application "System Events"\n'
            f'  {tgt_line}\n'
            '  set b to bounds of tgt\n'
            '  set bx to item 1 of b\n'
            '  set by to item 2 of b\n'
            '  set bw to item 3 of b\n'
            '  set bh to item 4 of b\n'
            + (f"  {x_line}\n" if x_line else "")
            + (f"  {y_line}\n" if y_line else "")
            + (f"  {w_line}\n" if w_line else "")
            + (f"  {h_line}\n" if h_line else "")
            + '  set bounds of tgt to {bx, by, bx + bw, by + bh}\n'
            + '  return "window: " & bx & "," & by & " " & bw & "x" & bh\n'
            + "end tell"
        )
        rc, out = self._run_osa(script)
        if rc != 0:
            return f"window: failed to resize — {out[:100]}"
        return out

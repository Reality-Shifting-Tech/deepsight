# Adding a new vision tool end-to-end

Pattern for adding a tool to the deepsight vision session loop. Covers all layers: Swift binary, Python backend, perception tool definition, orchestration, tests, and cross-platform support.

## 1. Swift binary (scripts/vision_eyes.swift)

If the tool needs new Apple Vision detection data, extend the binary:

- Add detection request (e.g. `VNRecognizeAnimalsRequest`, `VNDetectHumanRectanglesRequest`)
- In the completion handler, emit `box:<type>:<confidence>:<x>:<y>:<w>:<h>:<label>` lines for each detected instance
- Format: colon-separated, 7 fields after `box:`, coordinates normalized 0-1 (Apple Vision bottom-left origin)
- Emit lines alongside existing output (existing parsers silently skip unrecognized lines)
- Compile: `SDKROOT=... swiftc -target arm64-apple-macos14 scripts/vision_eyes.swift -o scripts/vision_eyes`

## 2. Python backend (backends.py)

If the tool returns structured spatial data:

- Add a `boxes()` method to `NativeVisionBackend` if not present (runs binary, parses box lines)
- Add `_parse_boxes(stdout)` static method for parsing box: lines
- Y-flip: `top_left_y = 1.0 - ny - nh` (Apple Vision origin is bottom-left)
- Return `[{type, confidence, x, y, w, h, label}]`

### Cross-platform vision backend

For Windows, create a `WindowsVisionBackend` class with the same `ask()` / `boxes()` interface:

- `ask()` — PIL for basic scene analysis (ImageStat for brightness, quantize for dominant colors, histogram for entropy). Optional pytesseract for OCR.
- `boxes()` — pytesseract `image_to_data` output → per-word boxes with confidence filtering (>0.3).
- Graceful degradation: when pytesseract is unimportable, boxes() returns [] and ask() returns PIL-only analysis.
- Import pattern: `try: import pytesseract as _pt; self._tesseract = _pt except ImportError: pass`
- Both backends share the same interface — the type alias is `VisionBackendType = NativeVisionBackend | WindowsVisionBackend`.
- Update `perception.py` and `orchestrator.py` VisionBackendType imports.

## 3. Perception layer (perception.py)

Three things:

a) **Tool definition** — add a `ToolDefinition` to `TOOL_DEFINITIONS`. Each tool has:
   - `name` — short verb (e.g. `locate`, `count`)
   - `description` — what it does, when to use, examples. Written for the reasoning model, not humans.
   - `parameters` — JSON schema: type=object, properties (with type/description per param), required array.

b) **Execution** — add a method (e.g. `_locate`) to `Perception`. First branch in `execute()`:

   ```python
   def execute(self, name, args, image):
       if name == "locate":
           return self._locate(args, image)
   ```

   Return a user-facing string the reasoning model reads as tool output.

c) **Matching heuristic** (for spatial tools) — `_match_box(box, what)` scores label/type overlap:
   - Exact label match = 1.0
   - Substring containment = 0.9
   - Type-level ("person" → human/face, "animal" → animal, "text" → text) = 0.8
   - Default = 0.0 (no match)

## 4. Orchestrator (orchestrator.py)

- SYSTEM_PROMPT may need updated tool list mentions
- `max_look_rounds` should account for locate rounds (same pool as look/count)
- If the tool returns coordinate data, mention coords as percentages in the prompt

## 5. Tests

- **backends**: Mock `subprocess.run` with `FakeProc`; test `_parse_boxes` with crafted stdout containing `box:` lines
- **perception**: Use `FakeVision(boxes_data=[...])` fixture; test matching, no-match, empty, region-scoped calls
- **orchestrator**: Ensure `FakeReasoning` accepts new tool names; SYSTEM_PROMPT mentions

## Non-vision tools (grounding)

Some tools don't go through the vision backend at all. The `ground` tool pattern:

### 1. Backend (backends.py)

Add a standalone backend class that doesn't inherit or wrap the vision backend:

```python
@dataclass(slots=True)
class SearchResult:
    url: str
    title: str
    snippet: str
    content: str = ""

class SearchBackend:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0): ...
    def search(self, query, count=3) -> list[SearchResult]: ...
    def fetch(self, url) -> str | None: ...
```

- Gated by API key: when unset, methods return empty/false results.
- Use existing deps (httpx) — no new package imports.
- Add the env var to `config.py`: `search_api_key: str = ""`.

### 2. Perception (perception.py)

- Import the backend class.
- `Perception.__init__` gets an optional `search_backend: SearchBackend | None = None` param.
- Tool definition in `TOOL_DEFINITIONS` — describe the dependency (e.g. "requires DEEPSIGHT_SEARCH_API_KEY") so the reasoning model knows when it might not work.
- Execution method (e.g. `_ground`) takes `args` only (no image). First branch in `execute()`.

### 3. Orchestrator (orchestrator.py)

- Import + pass `search_backend` through to `Perception.__init__`.
- No changes to the vision session loop itself — the tool just needs to be declared.

### 4. Tests

- Add `FakeSearchBackend` to `conftest.py` — `search()`/`fetch()` return canned data.
- Add `fake_search` fixture factory that accepts `results` and `fetch_text`.
- Test: no backend, no description, no results, full path (results + fetched content), custom query override.
- No mock needed: the fake IS the test backend.

## Principles

### Computer-use tools (action tools — orchestrator level)

Tools like `click`, `type`, `key`, `scroll`, `open`, `focus`, `apps`, and
`window` don't go through the vision backend at all. They're routed at the
orchestrator level.

**1. Backend (backends.py)**

Add a standalone class with an `execute(name, args)` dispatcher:

```python
class ComputerUseBackend:
    def __init__(self):
        # Platform detection
        self._is_windows = lambda: sys.platform == "win32"
        self._has_cliclick = subprocess.run(["which", "cliclick"], ...).returncode == 0
        self._screen_size = _screen_size()  # tkinter helper

    def _pixels(self, x_pct, y_pct): ...
    def execute(self, name, args):
        if name == "click": return self._click(args)
        ...
```

- **macOS**: cliclick (brew install) preferred, osascript fallback
- **Windows**: ctypes + user32.dll for clicks/scroll, PowerShell SendKeys for typing/key combos, cmd /c start for apps
- `_screen_size()` via tkinter.Tk().winfo_screenwidth/height (stdlib, cross-platform)
- Percentage → pixel: `int(x_pct / 100 * screen_w)`
- Import pattern for ctypes on Windows: `import ctypes` inside the method, protected by `_is_windows()`. Add `# type: ignore[attr-defined]` to each `ctypes.windll.user32.*` call — mypy on macOS will flag it otherwise.

**Windows-specific helper methods:**

```python
def _powershell(self, cmd: str, timeout: float = 15) -> tuple[int, str]:
    r = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, timeout=timeout)
    return (r.returncode, r.stdout.strip())
```

Used for: typing (`SendKeys::SendWait`), app listing (`Get-Process`), keyboard combos.

**2. Tool definitions (orchestrator.py)**

Add as `ACTION_TOOL_DEFINITIONS` — NOT in perception's `TOOL_DEFINITIONS`. Each tool has name, description, and JSON schema parameters matching standard function-calling:

```python
ACTION_TOOL_DEFINITIONS = [
    ToolDefinition(name="click", ...),
    ToolDefinition(name="type", ...),
    ...
]
```

`_tool_defs()` merges both lists. **IMPORTANT**: `_tool_defs()` accepts `include_actions: bool = True`. When no `ComputerUseBackend` is configured, pass `include_actions=False` — this excludes action tools from the model's tool list, preventing the reasoning model from wasting rounds on click/type/etc during vision-only sessions (static image analysis). The orchestrator's `run()` method passes `include_actions=self.computer is not None`.

**3. Routing (orchestrator.py tool loop)**

In the `for call in result.tool_calls:` loop, check `call.name` against action tool names:

```python
action_names = {"click", "type", "key", "scroll", "open", "focus", "apps", "window"}
if call.name in action_names and self.computer is not None:
    observation = self.computer.execute(call.name, call.arguments)
elif call.name in action_names:
    observation = "computer use not available — no backend configured"
else:
    observation = self.perception.execute(call.name, call.arguments, image)
```

**⚠** The `action_names` set is a hardcoded duplicate of `[t.name for t in ACTION_TOOL_DEFINITIONS]`. When adding a tool, update BOTH the definition list and this set.

**4. Orchestrator init**

Accept `computer: ComputerUseBackend | None = None` as an optional parameter, store as `self.computer`.

**5. Tests**

- Mock `subprocess.run` with `FakeProc` for backend tests
- Test `_tool_defs()` includes action tool names (verify shape)
- Test orchestrator's routing: send `click` in a tool call, verify it goes to `computer.execute()` not `perception.execute()`

**6. SYSTEM_PROMPT update guidance**

When adding a new vision tool, update the VISION tools list section in SYSTEM_PROMPT (orchestrator.py). The guidance is critical: the model must know which tools are for static image analysis (look, ocr, zoom, count, locate) vs live desktop use (ground, capture, watch). Without this, the model wastes rounds trying to capture/ground/watch on a static JPEG — the eval regression that hit golden-1980-philly and golden-sga-okc in August 2026. Always add both the tool name AND a one-line usage rule.

### Capture tool (perception level)

The `capture` tool captures the screen instead of taking an image argument.

**1. macOS capture**

```python
subprocess.run(["screencapture", "-x", tmp_path], capture_output=True, timeout=10)
```

- `-x` suppresses the capture sound
- For window capture: `osascript -e 'tell app "System Events" to get id of first window whose title contains "NAME"'` → `screencapture -l <id> -x`
- Temp file cleaned in `finally` block

**2. Windows capture**

```python
from PIL import ImageGrab
img_capture = ImageGrab.grab()
img_capture.save(tmp_path, format="PNG")
```

- No separate screencapture binary needed — PIL.ImageGrab is cross-platform
- Replace `screencapture -x` calls with `ImageGrab.grab().save()` on Windows

**3. Active image switching**

`Perception._captured_image` stores the captured PIL Image. Set after capture, used by `execute()` on subsequent calls:

```python
def execute(self, name, args, image=None):
    active_image = self._captured_image if self._captured_image is not None else image
```

Add `self._captured_image = None` to `__init__`.

**4. Perceptual hashing (cross-capture memory)**

dhash algorithm: resize to 9×8 grayscale → compare adjacent column pixels → "1" if left < right else "0" → hex string. Store previous hash + OCR text set on Perception for diff output.

### Watch tool (perception level — temporal monitoring)

The `watch` tool loops capture + dhash + optional analysis:

```python
while time.monotonic() < deadline:
    capture → load → dhash
    if hash == prev_hash: continue
    analyze with vision backend → append to timeline
    check "until" condition → break if text found
    sleep(interval)
```

- Capture platform-aware: screencapture on macOS, PIL.ImageGrab on Windows
- Temp file cleaned in `finally`
- dhash breaks early when frames are identical (no token cost)
- Timeline format: `t=0.0s OCR: text1 | text2`
- `until` does substring match on OCR text items (case-insensitive)
- Capped at 60s, interval min 0.5s

### Window/app management tools (action tools)

These tools manage the desktop environment so the model can switch between applications, arrange its workspace, and discover what's running.

**Backend methods (ComputerUseBackend)**

- `_open(args)` — macOS: `open -a`. Windows: `cmd /c start`. Simple subprocess call.
- `_focus(args)` — macOS: multi-line osascript searching System Events windows by title. Windows: `user32.FindWindowW(None, title)` + `SetForegroundWindow(hwnd)` via ctypes.
- `_apps(args)` — macOS: osascript iterate processes with `background only is false`. Windows: PowerShell `Get-Process | Where-Object {$_.MainWindowTitle -ne ""}`.
- `_window(args)` — macOS: set-bounds AppleScript. Windows: `user32.GetWindowRect` + `MoveWindow` via ctypes. Coordinates as % of screen.

**Helper for macOS**: `_run_osa(script)` runs multi-line osascript one-liners. Returns `(exit_code, stdout)`.

**Helper for Windows**: `_powershell(cmd)` runs PowerShell commands. Returns `(exit_code, stdout)`.

**Testing**: mock subprocess.run for backend tests. Test `_tool_defs()` includes action tools by name.

## Principles

- Every new tool adds ~3-5 test cases minimum
- Keep the tool description shorter than 200 chars (reasoning model context budget)
- Tool parameters: use `x/y/w/h` as % (0-100) for region tools, `what` for query tools
- Locate tools should return coordinates as % for agentic actions (clicking, targeting)
- Vision is device-native (Apple Vision on macOS, PIL on Windows). Action tools are local-only via osascript/cliclick (macOS) or ctypes/PowerShell (Windows).
- Non-vision tools: gate behind an env var, degrade gracefully, test with fakes
- Cross-platform: when adding new functionality, implement BOTH macOS and Windows paths. Use `sys.platform == "win32"` branches. Add mypy `# type: ignore[attr-defined]` on Windows-only ctypes calls.

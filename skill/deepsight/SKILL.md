---
name: deepsight
description: "Full DeepSight system: device-native vision, live screen capture, desktop automation, and web grounding for text-only LLMs."
version: 2.9.1
author: agent
created_by: agent
tags: [deepsight, vision, hermes, auxiliary, llm, proxy, launchd, deepseek]
---

# DeepSight

Device-native vision for text-only LLMs: a compiled Apple Vision binary (`vision_eyes`) answers image questions with zero tokens. Repo: `Reality-Shifting-Tech/deepsight`, checkout at `~/projects/deepsight`. **The repo has NO server** — `serve` was removed (AGENTS.md hard rule: no ports/serve in the repo). Hermes wiring uses a local-only proxy script that shells to the binary.

Use this when: working with the deepsight repo, wiring auxiliary.vision, debugging vision calls, building agent workflows with vision + desktop automation, or updating the README/docs.

## Native vision proxy (Hermes wiring)

`~/.hermes/scripts/native-vision-proxy.py` — stdlib-only OpenAI-compatible `/v1` server on `127.0.0.1:8199`. Each request decodes the `image_url` content part (base64 data URL, file path, or `file://`), writes a temp PNG, runs the `vision_eyes` binary, and returns its parsed output as the assistant message. ~1.3s per image, zero tokens.

```bash
python3 ~/.hermes/scripts/native-vision-proxy.py
```

- Serves one model: `deepsight-native`.
- Health check: `curl -s http://127.0.0.1:8199/v1/models`.
- Supports both `stream=true` (single content chunk + `[DONE]`) and plain responses.

### Env vars (in `.env`, gitignored — never commit, never in /tmp)

- `DEEPSIGHT_REASONING_API_KEY` — required for real reasoning calls (DeepSeek default).
- `DEEPSIGHT_REASONING_MODEL`, `DEEPSIGHT_REASONING_MAX_TOKENS`
- `DEEPSIGHT_MAX_LOOK_ROUNDS`
- `DEEPSIGHT_VISION_BACKEND` — `native` (Apple Vision) | `ollama` | openai-compatible
- `DEEPSIGHT_VISION_BIN` — absolute path to compiled `vision_eyes` binary (native backend only)
- `DEEPSIGHT_PORT` — default 8080 (only relevant if serve is ever restored; currently no server in repo)
- `DEEPSIGHT_VISION_MODEL`, `DEEPSIGHT_VISION_BASE_URL` — for ollama/remote eyes
- `DEEPSIGHT_SEARCH_API_KEY` — optional Brave Search API key for the `ground` tool (web factual verification); omit to disable
- Full field list in `src/deepsight/config.py` (Settings class), e.g. `DEEPSIGHT_REASONING_TOOL_ROUND_MAX_TOKENS`, `DEEPSIGHT_SKETCH_ENABLED`, `DEEPSIGHT_CACHE_TTL_SECONDS`

Keys live ONLY in env/`.env` and the running process — never echo values into chat or transcripts. To capture env from a live process for restart: `ps eww -p <pid> | tr ' ' '\\n' | grep '^DEEPSIGHT' | sed 's/^/export /'`.

### Vision eyes backend (which model "sees")

Eyes are pluggable. The backend interface is ``ask(prompt, image_bytes) -> VisionResult`` and ``boxes(image_bytes) -> list[dict]``, so any Python class implementing these can be a backend.

**macOS (PREFERRED — zero tokens, zero downloads, zero GPU):** `NativeVisionBackend` shells to the compiled Apple Vision binary `vision_eyes` (source: `scripts/vision_eyes.swift`). Set `DEEPSIGHT_VISION_BIN` to the binary path. Compile once:
```bash
cd ~/.hermes/skills/apple/macos-vision-framework/scripts && env SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk swiftc -target arm64-apple-macos14 vision_eyes.swift -o vision_eyes
```

**Windows (PIL + optional Tesseract):** `WindowsVisionBackend` uses PIL for basic scene analysis (colors, brightness, entropy) and optionally pytesseract for OCR + bounding boxes. Graceful when pytesseract is absent — PIL-only returns color/brightness/detail. Install: `winget install UB-Mannheim.TesseractOCR` then `pip install pytesseract`. Both backends share the same interface.

- `ollama` (minicpm-v) — local VLM, works but burns RAM. Not available on Windows without WSL.

`NativeVisionBackend` (backends.py) writes image bytes to a temp PNG, runs the binary, parses OCR lines + saliency counts, returns zero token usage. The vision tool loop (look/ocr/zoom/count/locate) crops regions first, so native OCR still answers region questions. The binary also emits `box:` lines for spatial grounding (see below).

### Spatial grounding (locate tool)

`vision_eyes.swift` v2.1+ emits `box:<type>:<confidence>:<x>:<y>:<w>:<h>:<label>` lines for every detected object — faces, humans, animals, text regions, rectangles, and salient objects. Coordinates are Apple Vision normalized (bottom-left origin 0-1). The Python side flips Y to top-left.

- `NativeVisionBackend._parse_boxes(stdout)` — parses box lines from binary output, returns `[{type, confidence, x, y, w, h, label}]` with Y-flipped coords.
- `NativeVisionBackend.boxes(image_bytes)` — runs the binary and returns parsed boxes in one call.
- `Perception._locate(args, image)` — the `locate` tool the reasoning model calls. Accepts `what` (description) + optional region coords. `_match_box(box, what)` scores matches: exact label match = 1.0, substring = 0.9, type-level ("person" → human/face) = 0.8. Returns sorted results with coords as percentages 0-100.
- Tool definition in `perception.TOOL_DEFINITIONS` (name="locate"). The reasoning model can ask "where is the logo" and get `x=5% y=10% w=20% h=8%` for clickable coords.

### Structured output (response_format)

`ReasoningBackend.chat()` and `Orchestrator.run()` accept an optional `response_format` dict. Pass `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {...}}` for schema-constrained answers (invoice extraction, structured receipts, UI element lists). Forwarded directly to the API payload — works with any OpenAI-compat backend (DeepSeek, etc.). For tool rounds, response_format is included alongside tools; the model decides whether to answer directly or call tools.

### Web grounding (ground tool)

The `ground` tool searches the web to verify facts, identify entities, or find current information. Gated by `DEEPSIGHT_SEARCH_API_KEY` (Brave Search API). When unset, the tool gracefully reports "unavailable" instead of erroring.

- `SearchBackend` (backends.py) — Brave Search API client: `search(query, count=3) → [SearchResult]` + `fetch(url) → str | None`. Returns `[]` / `None` when no API key is configured, so OSS users see a clear message.
- `Perception._ground(args)` — the tool handler. Takes `what` (claim) and optional `query` override. Searches → fetches top-1 page → returns structured text with title, URL, snippet, page content excerpt, and match count.
- Tool definition in `perception.TOOL_DEFINITIONS` (name="ground", description explains Brave dependency).
- `Perception.__init__` and `Orchestrator.__init__` accept optional `search_backend: SearchBackend | None` — when omitted, ground is disabled.
- Test pattern: `FakeSearchBackend(results=..., fetch_text=...)` returns canned data.

The ground tool is NOT a vision backend — it is a reasoning-layer tool. It is optional and clearly gated.

### Action tool gating (include_actions)

`_tool_defs(include_actions=self.computer is not None)` excludes desktop-action tools (click, type, key, scroll, open, focus, apps, window) when no ComputerUseBackend is configured. This prevents the model from wasting vision-only session rounds on action tools. Without this guard, eval images like golden-sga-okc and public-baseball regressed from max-rounds exhaustion — the model tried click/type/open on static JPEGs instead of focusing on look/ocr/zoom/count.

The parameter is threaded through all three `reasoning.chat()` call sites in the orchestrator loop: normal round, truncation retry, and capped final.

### Computer-use tools (action tools)

Four action tools let the reasoning model interact with the desktop after seeing it — closing the see-decide-act loop:

- `click(x%, y%)` — clicks at percentage coordinates (matching locate output). macOS: osascript or cliclick. Windows: user32.SetCursorPos + mouse_event.
- `type("text")` — types into the active input. macOS: osascript keystroke. Windows: PowerShell SendKeys.
- `key("cmd+s")` — keyboard combo or single key. Modifiers: cmd, shift, ctrl, alt/option. Special keys mapped to key codes. macOS: osascript or cliclick. Windows: PowerShell SendKeys with `{ENTER}`, `{TAB}`, etc.
- `scroll("down", 3)` — scrolls active window. macOS: key code 124/126. Windows: mouse_event with delta.

Backend: `ComputerUseBackend` in backends.py. Detects platform via `_is_windows()` and dispatches accordingly. `_powershell()` helper for Windows commands. `_run_osa()` helper for macOS. Screen pixel conversion via tkinter (cross-platform).

### Window/app management (action tools)

Four additional action tools for context switching — the model can launch, focus, list, and arrange applications:

- `open("Terminal")` — launches an app by name. macOS: `open -a`. Windows: `cmd /c start`.
- `focus("Safari")` — finds a window whose title contains the given substring and brings it to front. macOS: osascript System Events. Windows: user32 FindWindowW + SetForegroundWindow.
- `apps()` — lists every visible (non-background) application with its window count and per-window titles. macOS: osascript. Windows: PowerShell Get-Process.
- `window(name="Terminal", x=25, y=25, w=50, h=50)` — resizes/repositions a window using percentage screen coordinates. macOS: set-bounds AppleScript. Windows: GetWindowRect + MoveWindow (user32).

`apps()` first is the critical discovery step — the model calls this to understand what is open before deciding what to focus, capture, or click.

Full cycle pattern: `` apps() → open("Terminal") → type("npm run build") → key("return") → focus("Safari") → capture("Chrome") → locate("build button") → click(50, 30) → watch(until="done") ``

## Hermes vision wiring

`auxiliary.vision` routes through the native proxy:

```bash
hermes config set auxiliary.vision.provider custom
hermes config set auxiliary.vision.model deepsight-native
hermes config set auxiliary.vision.base_url http://127.0.0.1:8199/v1
hermes config set auxiliary.vision.api_key local
hermes config set auxiliary.vision.timeout 60
```

- Use `hermes config set`, NOT direct file edit — `~/.hermes/config.yaml` is a protected file.
- NO gateway restart needed: `auxiliary.*` config is read from disk per call.
- Verify resolution: `python -c "from agent.auxiliary_client import _resolve_task_provider_model; print(_resolve_task_provider_model('vision'))"` — expect `('custom', 'deepsight-native', 'http://127.0.0.1:8199/v1', 'local', None)`.

### Response shape

The proxy returns the full parsed signal dump as the message: `Image: WxH`, `OCR:` lines, `Scene:`, `Faces/Humans/Rectangles/Animals/Sports`, `Colors`. The agent replies from this dump — no raw tool output reaches the user.

## Persistent server (launchd)

Session-bound background processes die between sessions — that is the #1 cause of "vision broken" reports. Use the LaunchAgent. Installed copy at `~/Library/LaunchAgents/com.realityshifting.native-vision.plist` (logs → `~/.hermes/logs/native-vision.log`):

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.realityshifting.native-vision.plist
```

- RunAtLoad + KeepAlive → survives reboot and crashes.
- `launchctl bootstrap` is BLOCKED from inside the Hermes gateway process. The USER must run that one line from their own shell.
- If the session-bound proxy is already on :8199 when the user bootstraps, launchd's instance retries until the session one dies.

## Pitfall: vision "format bug" 400 = proxy down

Symptom: `vision_analyze` returns a 400 like `unknown variant 'image_url', expected 'text'` (Rust-serde style from api.deepseek.com).

Cause: native proxy is DOWN → gateway falls back to main text-only model → API rejects multimodal payload.

Diagnose:
1. `grep "Auxiliary vision" ~/.hermes/logs/agent.log | tail` — `connection error on custom` proves routing to proxy.
2. `curl -s http://127.0.0.1:8199/v1/models` — connection refused = proxy down.
3. Fresh-process resolution check — correct tuple means config is fine.

## Pitfall: 30s auxiliary timeout

Symptom: proxy IS up but `vision_analyze` returns 400 anyway; agent.log shows `Request timed out`.

Cause: default auxiliary timeout is 30s. A hung invocation (crash, first compile, huge image) exceeds it.

Fix: `hermes config set auxiliary.vision.timeout 60` (read from disk per call, no restart).

### Screen capture (capture tool)

The `capture` tool gives the reasoning model live, real-time vision of the desktop.

- macOS: shells `screencapture -x` to a temp PNG.
- Windows: uses `PIL.ImageGrab.grab()`.
- `capture("Chrome")` — targets a specific window by title substring (macOS only; osascript FindWindow).

After capture, the image is stored as `Perception._captured_image`. ALL subsequent vision tools (`look`, `ocr`, `zoom`, `count`, `locate`) automatically operate on the captured screen instead of the original image.

### Temporal monitoring (watch tool)

`watch(seconds=10, interval=1, until="error")`:

- Loops: capture → dhash → if changed, full vision analysis.
- Skips identical frames entirely — zero tokens for static screens.
- Returns a timeline of unique frames with OCR.
- `until` param stops early when target text appears.
- Capped at 60s, min 0.5s interval.
- macOS: screencapture. Windows: PIL.ImageGrab.

Uses dhash (difference hash): 9x8 grayscale resize → 64-bit hex fingerprint, ~1us per frame.

### Cross-capture memory (dhash + OCR diff)

After every `capture`:
1. dhash match → "screen: unchanged since last capture"
2. dhash mismatch → OCR diff: "new text: Build Failed", "text disappeared: Loading"
3. First capture baselines (no diff)

State: `_prev_hash: str` and `_prev_ocr: set[str]` on Perception instance. Per-session memory.

## Pitfall: README env names != actual env prefix

The repo README shows no-prefix names (`REASONING_API_KEY`). The code reads `DEEPSIGHT_*` only. Trust config.py + this skill.

## CLI commands

Three CLI commands:

- `deepsight describe <image>` — run the vision binary on a local image and print parsed output (OCR, scene, colors, detections).
- `deepsight doctor` — check vision binary exists, reasoning backend responds, and print connectivity status per component.
- `deepsight setup` — one-command onboarding: compiles the Swift binary (macOS), creates a `.env` file with defaults, prints next steps. On Windows, detects platform and prints instructions for optional Tesseract install.

The `setup` command is the entry point for agent integration. Run after `git clone` + `uv sync`:

```bash
uv run deepsight setup
# → compiles vision_eyes, creates .env
uv run deepsight doctor
# → confirms everything works
```

## Smoke test

POST to the proxy with `image_url` (base64 data URL) and model `deepsight-native` — expect a full signal dump in ~1-2s.

### Response style for image answers

Give the direct answer only. Describe what you see — subject, colors, text, layout, objects. **No meta-commentary** about the vision pipeline: no "deepsight caught X but missed Y," no "the locate/OCR loop found this," no system internals. Just what is in the picture.

Examples:

- ✓ "LeBron James in a Lakers jersey, gold with purple trim. Nike swoosh, bibigo sponsor patch, gold chain. Studio lighting, white background."
- ✗ "Deepsight caught the chain but missed the Korean characters."
- ✗ "The OCR loop found the LAKERS script at the bottom."

Single concise reply, no raw tool-output flood, no pipeline commentary.

### README documentation conventions

Lead with a concrete, plain-English hook naming the target user. First paragraph says what it can DO, not how it works. Example: "Give DeepSeek (or any text-only model) eyes and hands. It can look at images you send, take screenshots of your desktop, read text on screen, click buttons, type into fields, open apps, and search the web to verify facts."

**Images:**
- Architecture diagrams: use **SVG** (vector, readable text, renders on GitHub). Write by hand — AI-generated images with text are illegible.
- Conceptual/capability images: use the configured image generation provider (`image_generate` tool — model not selectable by agent). Prompt for a dark-mode Mac desktop with vision overlays, or an infographic-style layout with icons. Acceptable: no readable text, abstract/illustrative, evocative of the concept. Avoid: AI-generated text, garbled labels, hallucinations of UI.
- Terminal demos: use **actual terminal screenshots** saved as PNG. No AI generation.
- One architecture image at the top, no duplicates later.
- No em-dashes in captions or alt text.
- After generating images, verify they committed and pushed correctly. GitHub caches README images — hard refresh may be needed.
- FLUX-generated images tend to be artistic/interpretive rather than informative. For conceptual headers (abstract capability illustration) this is fine. For explanatory diagrams, use SVG instead.
- **After generating images, ALWAYS verify they committed and pushed.** A common failure: the image was generated and saved to disk but forgot to `git add`. Check with `git ls-files docs/images/<name>.png` + `file docs/images/<name>.png`. Verify on GitHub's CDN: `curl -sI "https://raw.githubusercontent.com/Reality-Shifting-Tech/deepsight/main/docs/images/<name>.png" | head -2` (expect `HTTP/2 200`).

### FAL image generation via fal_client

The `image_generate` tool uses FAL FLUX 2 Klein 9B (fixed backend). For other models (gpt-image-2, flux-pro, etc.), use `fal_client` directly:

```python
import os, fal_client
os.environ["FAL_KEY"] = "<key_id>:<key_secret>"

result = fal_client.subscribe(
    "fal-ai/gpt-image-2",
    arguments={
        "prompt": "description...",
        "image_size": "landscape_16_9",
    },
)
# result["images"] is list of dicts with "url" key
```

Set the key from the FAL dashboard (Settings > API Keys). Save to shell profile or project `.env.local` (gitignored). For Hermes: `echo 'export FAL_KEY=...' >> ~/.hermes/.env`.

gpt-image-2 produces clearer, more photorealistic images than FLUX 2, especially for UI/infographic concepts with rendered text and interface elements. It handles Mac desktop layouts, icons, and split-screen workflows much better.

### Agent Integration

Copy-paste block for any agent (Hermes, Codex, Claude, etc.):

```bash
git clone https://github.com/Reality-Shifting-Tech/deepsight.git
cd deepsight
uv sync
uv run deepsight setup
```

```python
from deepsight.backends import NativeVisionBackend, ReasoningBackend
from deepsight.orchestrator import Orchestrator

vision = NativeVisionBackend(bin_path="scripts/vision_eyes")
reasoning = ReasoningBackend(base_url="https://api.deepseek.com/v1", api_key="...", model="deepseek-v4-flash")
agent = Orchestrator(vision=vision, reasoning=reasoning)
result = agent.run("data:image/png;base64,...", "Describe this image")
```

On Windows, use `WindowsVisionBackend()` instead of `NativeVisionBackend`.

## Weakest-hypothesis principle (eval design)

Source: Bennett, "The Optimal Choice of Hypothesis Is the Weakest, Not the Shortest" (AGI-23, arXiv:2301.12987). Compression (MDL, Ockham-as-length) is neither necessary nor sufficient for generalization; the optimal choice of hypothesis is the WEAKEST valid one — the least-specific claim that still fits the evidence (generalized 1.1-5x better than MDL in binary-arithmetic experiments).

Applied to deepsight eval authoring:

- Prefer the weakest assertion that still catches the regression: `scene_contains` at a low rank over `scene_top1`, `sports_present` over a single sport, `ocr_contains` over exact string. Matches golden-jellyfish-art ("weak scene confidences (0.08) are the honest ceiling").
- Do NOT over-fit assertions to one golden image's specifics. A hypothesis that only explains the golden image is a short, strong hypothesis — exactly what the paper shows generalizes worst.
- Answer style: when scene/signal confidence is low, the model should state the most general valid description, not the most specific guess (weakest, not shortest).

## Dev notes

- Skill source of truth: `skill/deepsight/` in the repo (Reality-Shifting-Tech/deepsight). Install for any agent: `curl -fsSL https://raw.githubusercontent.com/Reality-Shifting-Tech/deepsight/main/scripts/install-skill.sh | bash` (or `cp -R skill/deepsight ~/.hermes/skills/` from a checkout). The deepsight-repo cron watcher (~/.hermes/scripts/deepsight-skill-sync.py, daily 10:00) reports repo drift for prose review; clean days silent.
- Eval tiers: tier 0 = `uv run python eval/run_eval.py` (~30s CI), tier 1 = `uv run python eval/run_loop_eval.py` (~$0.10). Targets: tier 0 >=99%, tier 1 >=95%. Current: 319/320 (99.7%), 70/72 (97.2%).
- See references/tool-protocol.md for the tool-addition pattern. 16 tools total (8 vision + 8 action). Update action_names set in orchestrator loop when adding action tools.
- Cross-platform: WindowsVisionBackend (PIL + optional pytesseract) mirrors NativeVisionBackend interface. ComputerUseBackend detects platform via sys.platform and dispatches to user32 ctypes on Windows vs osascript on macOS. _capture uses PIL.ImageGrab on Windows. Test on the target platform.
- Action tool gating: `_tool_defs(include_actions=self.computer is not None)` prevents the model from seeing click/type/open in vision-only sessions. Without this, eval regressed (golden-sga-okc, public-baseball hit max rounds).
- System prompt design: when the tool set exceeds 4-5 tools, the system prompt needs explicit usage guidance. The eval regressed because the model called ground/capture/watch on static images. Fix: add "IMPORTANT: for static images, use look/ocr/zoom/count/locate only. Do NOT call ground, capture, or watch."
- reasoning_tool_round_max_tokens default 256; orchestrator escalates once on truncated rounds.
- Run tests: `.venv/bin/pytest -q`.
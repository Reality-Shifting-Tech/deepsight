# FAL gpt-image-2 Usage

## Model endpoint

`fal-ai/gpt-image-2` — OpenAI GPT-4o image generation hosted on FAL.

Key format: `<key_uuid>:<secret>` from FAL dashboard (Settings > API Keys).

## ⚠ Critical: os.environ pitfall

`fal_client` reads the key from `os.environ` only. If the key is set in the
shell (`.bashrc`, `.zshrc`) but not exported to the Python process — or if
Hermes' environment filtering strips it (MCP servers, cron jobs, subprocesses)
— `fal_client` raises `MissingCredentialsError`.

**Fix:** Set `os.environ["FAL_KEY"]` explicitly in the same Python process
BEFORE importing `fal_client`:

```python
import os
os.environ["FAL_KEY"] = "<uuid>:<secret>"
import fal_client    # now reads from os.environ
```

For Hermes sessions, add to `~/.hermes/.env` so the gateway sources it:

```bash
echo 'export FAL_KEY=<uuid>:<secret>' >> ~/.hermes/.env
```

But even this doesn't reach the Python process if Hermes uses filtered
environments. The `os.environ` call is the sure path for ad-hoc Python scripts.

## Setup

```bash
# Export in current shell
export FAL_KEY="<uuid>:<secret>"

# Save permanently for Hermes
echo 'export FAL_KEY=<uuid>:<secret>' >> ~/.hermes/.env

# Save for project (not committed)
echo 'FAL_KEY=<uuid>:<secret>' > project/.env.local
echo '.env.local' >> project/.gitignore
```

## Call pattern

```python
import os
os.environ["FAL_KEY"] = "<uuid>:<secret>"
import fal_client

result = fal_client.subscribe(model_id, arguments={...})
# result["images"] → [{"url": "https://...", ...}, ...]
# Download: curl -sL <url> -o path.png
```

Generation takes 60-105 seconds.

## Effective prompts

### Capabilities hero (split-screen AI + desktop)

```
A clean split-screen illustration showing an AI assistant in a dark chat
interface on the left, and a Mac desktop on the right with terminal windows,
browser, and code editor visible. Glowing purple arrows connect the chat to
the desktop actions. Vision bounding boxes and OCR highlights appear overlaid
on the screen. Minimalist dark UI style, purple and teal color scheme,
professional tech look. No readable text needed.
```

gpt-image-2 renders actual Mac UI elements (menu bar, dock, window chrome)
correctly. It also handles conceptual overlays (arrows, bounding boxes,
labels) without corrupting the underlying UI.

### Tools infographic (2x2 glassmorphism — best result)

```
A clean dark UI illustration showing 4 labeled sections in a 2x2 grid layout.
Section 1 (top-left, purple): 'VISION' with labels look, ocr, zoom, count,
locate. Section 2 (top-right, cyan): 'LIVE' with labels capture, watch.
Section 3 (bottom-left, teal): 'GROUND' with label ground. Section 4
(bottom-right, green): 'ACTION' with labels click, type, key, scroll, open,
focus, apps, window. Each section is a rounded card with a subtle glow.
Dark background. Professional clean tech style, no extra decoration, all text
clearly readable.
```

This produced a glassmorphism 2x2 panel with ALL 16 tools correctly labeled,
each with a unique icon. Far superior to the earlier 6-icon infographic.

### Tools infographic (6-icon cycle — earlier attempt, less clear)

```
A clean 16:9 infographic showing six icons arranged in a circle connected by
glowing arrows. The icons are: an eye (look/see), a camera (capture), a
magnifying glass (search/locate), a mouse cursor (click), a window with app
icons (open/focus), and a globe (web). At the center is a glowing brain icon.
Dark background with neon purple, cyan, and pink gradients. Professional AI
tech illustration, minimal and clean. No readable text needed.
```

Only 6 icons, too abstract. Use the 2x2 grid prompt instead — it labels
every tool and the text actually renders cleanly.

### RST Philadelphia tech banner

```
Dark and futuristic tech banner with Philadelphia skyline silhouette at the
bottom in dark grey. Above the skyline, abstract geometric tech patterns,
glowing cyan (#06B6D4) and neon yellow-green (#E7FF02) laser grid lines.
A subtle VR headset outline made of light particles floats in the upper right.
Dark near-black background (#0C0C0C). Clean, professional, tech-forward.
No text. 16:9 landscape aspect ratio.
```

Successfully renders recognizable Philly landmarks (Liberty Place, Comcast
Center, Ben Franklin Bridge) with brand-consistent cyan + acid green neon.

## Image sizes

- `landscape_16_9` → 1088x608, ~700KB PNG
- Use `square` for profile/icon-style images

## Verification

After generating an image and pushing to GitHub, ALWAYS verify it committed
and resolves:

```bash
# 1. Check it's on disk
file docs/images/<name>.png
# → PNG image data ...

# 2. Check it's git-tracked
git ls-files docs/images/<name>.png
# → docs/images/<name>.png

# 3. Check it resolves on GitHub CDN
curl -sI "https://raw.githubusercontent.com/<org>/<repo>/main/docs/images/<name>.png" | head -2
# → HTTP/2 200
```

GitHub caches README images with 300s max-age. Hard refresh may be needed.
If the image was generated but the curl returns 404, it was never committed
— re-add and push.

## Comparison with FLUX 2

| Aspect | FLUX 2 Klein 9B | gpt-image-2 |
|--------|-----------------|-------------|
| UI/interface rendering | Abstract/interpretive, garbled text, window shapes approximate | Recognizable Mac UI elements, proper layout |
| Icons | Fills prompts with decorative details not requested | Renders what the prompt asks |
| Photorealism | Good for landscapes, portraits, artistic scenes | Better for structured/layered compositions |
| Speed | ~30-60s | ~60-105s |
| Cost per image | Depends on FAL config | Depends on FAL config |
| Text rendering | Garbled | Renders short labels surprisingly well |

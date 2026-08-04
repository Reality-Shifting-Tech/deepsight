# DeepSight

> Device-native vision. Describe any image with the capabilities already built
> into your device. No server required. Zero tokens, zero model downloads.

![License: MIT](https://img.shields.io/github/license/Reality-Shifting-Tech/deepsight)
![Python](https://img.shields.io/pypi/pyversions/deepsight)
![PyPI](https://img.shields.io/pypi/v/deepsight)
![CI](https://img.shields.io/github/actions/workflow/status/Reality-Shifting-Tech/deepsight/ci.yml?branch=main)
![Status: early development](https://img.shields.io/badge/status-early%20development-orange)

DeepSight is an open-source (MIT) vision toolkit that runs **on the device**
using the OS's own vision frameworks. On macOS it uses Apple Vision directly:
OCR, scene classification, saliency, face/human/rectangle detection. No server
process, no network call, no tokens burned, no model downloads. Fast and free.

```sh
deepsight describe screenshot.png
# OCR text:
#   MIKE
#   MYERS
#   crypto.com
# Scene: adult(0.91), people(0.91)
# faces: 1
```

## How it works

### describe: device-native (no server)

```
+----------------------+
|  deepsight describe  |  shells the OS vision binary directly
+----------------------+
        |
        v
+---------------------------------------------------+
| Apple Vision (on-device, zero tokens, zero cost)  |
|  - OCR text extraction                            |
|  - scene classification (1000+ classes)           |
|  - attention + object saliency maps               |
|  - face / human / rectangle detection             |
+---------------------------------------------------+
```

The eyes are the compiled Apple Vision binary (`vision_eyes`, see
[docs/architecture.md](docs/architecture.md) for the compile command). It runs
in milliseconds on-device and burns zero tokens, zero GPU, zero network.

## Quickstart

Prerequisites: Python >= 3.11, [uv](https://docs.astral.sh/uv) (or pip), and on
macOS a compiled Apple Vision binary (see
[docs/architecture.md](docs/architecture.md) for the compile command).

```bash
# Install from source (pre-release; PyPI publish is tracked in CHANGELOG)
git clone https://github.com/Reality-Shifting-Tech/deepsight
cd deepsight
uv venv
uv pip install -e .

# Describe any image with the device's own vision. No server, no tokens.
deepsight describe screenshot.png
```

Set `DEEPSIGHT_VISION_BIN=/path/to/vision_eyes` if the binary is not on your
`PATH`.

## Configuration

All configuration is via environment variables with a `DEEPSIGHT_` prefix,
optionally in a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `DEEPSIGHT_VISION_BIN` | `vision_eyes` | Path to the native Apple Vision binary |
| `DEEPSIGHT_REASONING_BASE_URL` | `https://api.deepseek.com/v1` | Optional OpenAI-compatible reasoning endpoint (vision-session loop) |
| `DEEPSIGHT_REASONING_API_KEY` | *(optional)* | API key for the reasoning endpoint |
| `DEEPSIGHT_REASONING_MODEL` | `deepseek-v4-flash` | Reasoning model name |
| `DEEPSIGHT_MAX_LOOK_ROUNDS` | `5` | Maximum look/crop/ocr/zoom rounds per session |
| `DEEPSIGHT_CACHE_TTL_SECONDS` | `3600` | Perception cache TTL in seconds |

## Development

```bash
uv venv
uv pip install -e '.[dev]'

make lint        # ruff check . (zero-warning policy)
make typecheck   # mypy src
make test        # pytest
make bench       # benchmark comparison (baseline modes, see docs/benchmarks.md)
make all         # lint + typecheck + test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Agents
working in this repository should read [AGENTS.md](AGENTS.md) first.

## Credits

DeepSight's design draws on two open-source projects:

- [visionbridge](https://github.com/) - one-shot image description bridging
  for text-only models; the baseline this project improves on (MIT).
- [minicpm-mcp](https://github.com/) - MCP tools around Ollama `minicpm-v`;
  inspiration for the targeted vision-pass protocol (MIT).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full list.

## License

[MIT](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

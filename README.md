# DeepSight

> Interactive vision for text-only LLMs. Drop-in OpenAI-compatible server.

![License: MIT](https://img.shields.io/github/license/Reality-Shifting-Tech/deepsight)
![Python](https://img.shields.io/pypi/pyversions/deepsight)
![PyPI](https://img.shields.io/pypi/v/deepsight)
![CI](https://img.shields.io/github/actions/workflow/status/Reality-Shifting-Tech/deepsight/ci.yml?branch=main)
![Status: early development](https://img.shields.io/badge/status-early%20development-orange)

DeepSight is an open-source (MIT) vision proxy that gives text-only LLMs
interactive vision through a **vision-session loop**: instead of one giant,
vague description of the whole image, the reasoning model gets a compact scene
sketch up front and then asks for exactly what it needs using tool calls
(`look`, `crop`, `ocr`, `zoom`). Each tool call is answered by a targeted,
small vision-model pass. Cost scales with curiosity, not with image size.

Point any OpenAI-compatible client at `http://localhost:8080/v1` and text-only
models such as DeepSeek V4 Flash start seeing images. The default stack is
DeepSeek for reasoning and Ollama `minicpm-v` for vision, but both backends are
swappable.

## How it works

```
+----------------------+
|  Any OpenAI client   |  curl, openai SDK, OpenWebUI, LibreChat
+----------------------+
        |
        |  POST /v1/chat/completions  (image_url content)
        v
+----------------------+
|   deepsight server   |  FastAPI, OpenAI-compatible /v1
+----------------------+
        |
        |  vision session loop
        v
+---------------------------------------------------+
| 1. sketch(image)      one VLM pass -> compact JSON |
|                       scene inventory (objects,    |
|                       text regions, layout)        |
| 2. inject sketch + tool schemas into the context   |
| 3. reasoning model emits look / crop / ocr / zoom  |
| 4. bridge answers each call with a TARGETED VLM    |
|    pass over just that region                      |
| 5. repeat until the model is satisfied, then       |
|    produce the final answer                        |
+---------------------------------------------------+
        |                           |
        v                           v
+------------------+     +----------------------+
| reasoning backend |     |   vision backend     |
| DeepSeek (default)|     | Ollama minicpm-v     |
| or any OpenAI /   |     | or any VLM           |
| Anthropic URL     |     |                      |
+------------------+     +----------------------+
        |                           |
        +------------+--------------+
                     v
            +------------------+
            | perception cache |
            | content-addressed|
            | by image + crop |
            | hash, TTL       |
            +------------------+
```

A perception cache makes repeated questions about the same image nearly free:
every sketch and crop answer is content-addressed by image hash and crop hash,
so the same region is never re-analyzed twice within the TTL.

### Why this beats the one-shot description

| | One-shot description bridge | Interactive session (DeepSight) |
|---|---|---|
| First pass | 300-500 token scene novel | ~60-120 token sketch |
| "Error in bottom-right?" | model guesses or re-reads everything | targeted crop, exact answer |
| 4K UI screenshot | 500 tokens, still vague | sketch + on-demand zooms |
| Cost per question | fixed per image | proportional to looking |
| Quality ceiling | description quality | the model's own curiosity |

The full architecture and design rationale live in
[docs/architecture.md](docs/architecture.md).

## Quickstart

Prerequisites: Python >= 3.11, [uv](https://docs.astral.sh/uv) (or pip),
and an Ollama instance with `minicpm-v` pulled
(`ollama pull minicpm-v`). A DeepSeek API key is needed for the default
reasoning backend.

```bash
# Install from source (pre-release; PyPI publish is tracked in CHANGELOG)
git clone https://github.com/Reality-Shifting-Tech/deepsight
cd deepsight
uv venv
uv pip install -e .

# Configure
export REASONING_API_KEY=sk-...
export VISION_BASE_URL=http://localhost:11434/v1   # Ollama (default)

# Run
deepsight
# or: python -m deepsight
```

The server listens on `http://localhost:8080/v1` by default. Once the first
stable release is on PyPI you will also be able to `pip install deepsight` or
`uv tool install deepsight`.

## Configuration

All configuration is via environment variables.

| Variable | Default | Description |
|---|---|---|
| `REASONING_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible reasoning backend base URL |
| `REASONING_API_KEY` | *(required)* | API key for the reasoning backend |
| `REASONING_MODEL` | `deepseek-v4-flash` | Reasoning model name |
| `VISION_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible vision backend base URL (Ollama) |
| `VISION_MODEL` | `minicpm-v` | Vision model name |
| `PORT` | `8080` | HTTP port for the DeepSight server |
| `MAX_TOOL_ROUNDS` | `8` | Maximum look/crop/ocr/zoom rounds per session |
| `CACHE_TTL` | `3600` | Perception cache TTL in seconds |

## Client examples

**curl**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What does the error message say?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/screenshot.png"}}
        ]
      }
    ]
  }'
```

**openai Python SDK**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize this dashboard."},
            {"type": "image_url", "image_url": {"url": "https://example.com/dashboard.png"}},
        ],
    }],
)
print(resp.choices[0].message.content)
```

**OpenWebUI / LibreChat**

Add a custom OpenAI-compatible provider pointing at
`http://localhost:8080/v1` with any API key. Models served by DeepSight appear
automatically via `/v1/models`.

## Benchmarks

DeepSight is measured against two baselines on three public benchmarks:
ChartQA, MathVista, and OCRBench-v2. The harness streams rows from the
Hugging Face datasets-server API, so no datasets are downloaded locally.
The headline metric is **tokens per correct answer**: efficiency and
capability, not just accuracy.

| Benchmark | Accuracy | Tokens per correct answer | Avg latency (s) |
|---|---|---|---|
| ChartQA (test) | TBD - run `make bench` | TBD | TBD |
| MathVista (testmini) | TBD - run `make bench` | TBD | TBD |
| OCRBench-v2 (test) | TBD - run `make bench` | TBD | TBD |

Methodology, scoring rules, and how to reproduce the numbers are in
[docs/benchmarks.md](docs/benchmarks.md).

## Development

```bash
uv venv
uv pip install -e '.[dev]'

make lint        # ruff check . (zero-warning policy)
make typecheck   # mypy src
make test        # pytest
make dev         # uvicorn with hot reload on :8080
make bench       # three-way benchmark comparison (see docs/benchmarks.md)
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

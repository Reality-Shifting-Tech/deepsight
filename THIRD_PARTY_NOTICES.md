# Third-Party Notices

DeepSight is built on open-source software. This file credits the projects
that made it possible.

## Design inspiration

DeepSight's architecture draws directly on two open-source projects. Both are
MIT licensed, and both are credited in the README:

- [visionbridge](https://github.com/) - one-shot image description bridging
  for text-only models; the category baseline DeepSight improves on. MIT.
- [minicpm-mcp](https://github.com/) - MCP tooling around Ollama `minicpm-v`;
  inspiration for the targeted vision-pass tool protocol. MIT.

(Repository URLs for visionbridge and minicpm-mcp will be linked here once
their canonical upstreams are confirmed during the integration step.)

## Runtime dependencies

- [FastAPI](https://fastapi.tiangolo.com) - MIT
- [Uvicorn](https://www.uvicorn.org) - BSD-3-Clause
- [httpx](https://www.python-httpx.org) - BSD-3-Clause
- [Pillow](https://python-pillow.org) - HPND
- [Pydantic](https://docs.pydantic.dev) - MIT
- [pydantic-settings](https://github.com/pydantic/pydantic-settings) - MIT
- [python-multipart](https://github.com/Kludex/python-multipart) - Apache-2.0

## Development dependencies

- [pytest](https://pytest.org) - MIT
- [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio) - Apache-2.0
- [ruff](https://astral.sh/ruff) - MIT
- [mypy](http://mypy-lang.org) - MIT

## Benchmark datasets

The benchmark harness streams evaluation rows from the Hugging Face
datasets-server API; no datasets are downloaded locally. The datasets used are
published under their respective licenses:

- [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
  (`HuggingFaceM4/ChartQA`, test split) - chart reasoning
- [MathVista](https://huggingface.co/datasets/AI4Math/MathVista)
  (`AI4Math/MathVista`, testmini split) - visual math
- [OCRBench-v2](https://huggingface.co/datasets/lmms-lab/OCRBench-v2)
  (`lmms-lab/OCRBench-v2`, test split) - OCR and screenshot QA

See [docs/benchmarks.md](docs/benchmarks.md) for how they are used.

# DeepSight development tooling.
# Requires: uv (https://docs.astral.sh/uv), Python >= 3.11.

UV          ?= uv
PYTHON      ?= python3

# Benchmark defaults (see docs/benchmarks.md for the full methodology)
BENCH_LIMIT         ?= 20
BENCH_SLEEP         ?= 0
BENCH_OUT           ?= bench/results

.PHONY: help all test lint typecheck format bench bench-direct bench-bridge

help: ## Show available targets
	@echo "deepsight targets:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

all: lint typecheck test ## Full pre-push gate

test: ## Run the test suite (pytest)
	$(UV) run pytest

lint: ## Ruff lint (zero-warning policy)
	$(UV) run ruff check .

typecheck: ## Static typecheck (mypy over src/)
	$(UV) run mypy src

format: ## Auto-format with ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

bench: bench-direct bench-bridge ## Benchmark comparison (baseline modes)

bench-direct: ## Baseline A: direct VLM (capability ceiling)
	@test -n "$(DIRECT_ENDPOINT)" || (echo "DIRECT_ENDPOINT is required (e.g. https://token.sensenova.ai/v1)"; exit 1)
	@test -n "$(DIRECT_MODEL)" || (echo "DIRECT_MODEL is required (e.g. sensenova-6.7-flash-lite)"; exit 1)
	@test -n "$(DIRECT_API_KEY)" || (echo "DIRECT_API_KEY is required"; exit 1)
	$(PYTHON) bench/harness.py --bench all --limit $(BENCH_LIMIT) --endpoint $(DIRECT_ENDPOINT) --model $(DIRECT_MODEL) --api-key $(DIRECT_API_KEY) --sleep $(BENCH_SLEEP) --out $(BENCH_OUT)_direct.json

bench-bridge: ## Baseline B: one-shot description bridge (visionbridge style)
	@test -n "$(BRIDGE_ENDPOINT)" || (echo "BRIDGE_ENDPOINT is required"; exit 1)
	@test -n "$(BRIDGE_MODEL)" || (echo "BRIDGE_MODEL is required"; exit 1)
	@test -n "$(BRIDGE_API_KEY)" || (echo "BRIDGE_API_KEY is required"; exit 1)
	$(PYTHON) bench/harness.py --bench all --limit $(BENCH_LIMIT) --endpoint $(BRIDGE_ENDPOINT) --model $(BRIDGE_MODEL) --api-key $(BRIDGE_API_KEY) --sleep $(BENCH_SLEEP) --out $(BENCH_OUT)_bridge.json

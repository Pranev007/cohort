.DEFAULT_GOAL := help
.PHONY: help install demo generate scan evaluate serve test lint format typecheck docker clean check baseline figures report real-eval

PY ?= python
DOCS ?= 15000
SEED ?= 1337

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	$(PY) -m pip install -e '.[dev]'

demo:  ## Generate, scan and evaluate in one shot (this is the 60-second tour)
	cohort demo --documents $(DOCS)

generate:  ## Build the synthetic corpus
	cohort generate --documents $(DOCS) --seed $(SEED) --out artifacts/corpus

scan:  ## Run the pipeline and write findings
	cohort scan

report:  ## Scan and write the self-contained HTML report (the demo artifact)
	cohort scan --findings 300 --html
	@echo "open artifacts/reports/scan_report.html"

figures:  ## Regenerate the README charts from the committed result JSONs
	$(PY) scripts/make_figures.py

real-eval:  ## Run the real-corpus experiments (downloads ~450 MB on first run)
	cohort real-eval --config configs/real.yaml

evaluate:  ## Score against ground truth, with ablations
	cohort evaluate --out artifacts/reports

serve:  ## Run the findings API on :8000
	uvicorn cohort.api.main:app --reload --port 8000

test:  ## Run the test suite
	pytest -q

test-fast:  ## Skip the slow end-to-end tests
	pytest -q -m "not slow"

lint:  ## Lint
	ruff check src tests scripts

format:  ## Auto-format
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:  ## Static types
	mypy src/cohort

check: lint typecheck test  ## Everything CI runs

baseline:  ## Record the current evaluation as the regression baseline
	cohort generate --documents $(DOCS) --seed $(SEED) --out artifacts/corpus
	cohort evaluate --out artifacts/reports
	mkdir -p benchmarks
	cp artifacts/reports/evaluation.json benchmarks/baseline.json
	@echo "baseline recorded at benchmarks/baseline.json"

docker:  ## Build the container image
	docker build -t cohort:latest .

clean:  ## Remove generated artifacts and caches
	rm -rf artifacts .pytest_cache .ruff_cache .mypy_cache coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

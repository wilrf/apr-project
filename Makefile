.PHONY: help install test test-all lint format typecheck features evaluate ab reproduce dashboard dashboard-dev clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install deps (incl. lstm + dev)
	python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,lstm]"

test:  ## Fast test suite (excludes slow LSTM model/trainer tests)
	python3 -m pytest tests/ --ignore=tests/models/test_lstm_model.py --ignore=tests/models/test_lstm_trainer.py -q

test-all:  ## Full test suite including slow LSTM tests
	python3 -m pytest tests/ -q

lint:  ## Ruff + black --check
	python3 -m ruff check src/ tests/ && black --check src/ tests/

format:  ## Auto-format with black and ruff --fix
	black src/ tests/ && python3 -m ruff check --fix src/ tests/

typecheck:  ## Run mypy
	python3 -m mypy

features:  ## Regenerate train.csv + test.csv
	python3 -m src.data.generate_features

evaluate:  ## Evaluate on the held-out test set
	python3 -m src.models.evaluate_test_set

ab:  ## Run the spread A/B experiment (quick: LR + XGB)
	python3 -m src.models.run_ab_experiment --quick

reproduce: features ab evaluate  ## Deterministic end-to-end regeneration (numbers stay anchored to results/)

dashboard:  ## Build the frontend and serve the dashboard on localhost (Track C)
	@echo "Dashboard target is wired up in the Track C plan."

dashboard-dev:  ## Run the dashboard in dev mode with hot reload (Track C)
	@echo "Dashboard-dev target is wired up in the Track C plan."

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__

PYTHON ?= python

.PHONY: help install test validate check-dataset ci eval benchmark benchmark-smoke

help:
	@echo "Common commands:"
	@echo "  make install          Install the package in editable mode"
	@echo "  make test             Run unit tests"
	@echo "  make validate         Validate labeled JSONL records"
	@echo "  make check-dataset    Check raw/labeled/split consistency"
	@echo "  make ci               Run local non-API CI checks"
	@echo "  make eval             Run full live eval with the default prompt"
	@echo "  make benchmark        Run live dev/test benchmark"
	@echo "  make benchmark-smoke  Run one live example per split"

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_labels.py

check-dataset:
	$(PYTHON) scripts/check_dataset.py

ci: test validate check-dataset

eval:
	$(PYTHON) scripts/run_eval.py

benchmark:
	$(PYTHON) scripts/run_benchmark.py

benchmark-smoke:
	$(PYTHON) scripts/run_benchmark.py --limit-per-split 1

# Pipeline stages land incrementally; unbuilt targets fail loudly rather than
# silently do nothing. PY is overridable so CI can run `make test PY=python`.
PY ?= .venv/bin/python

.PHONY: data fit backtest forecast freeze-forecast figures reproduce test venv

venv:  ## create local env and install package + dev deps
	python3 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e ".[dev]"

data:  ## build content-addressed Parquet snapshot from data/raw/results.csv
	$(PY) -m wc26.data.ingest

fit:  ## fit the production model at the latest as_of -> artifacts/
	@echo "make fit: not implemented yet" && exit 1

backtest:  ## walk-forward backtest over World Cup folds
	@echo "make backtest: not implemented yet" && exit 1

forecast:  ## Monte Carlo simulation of the 2026 tournament -> artifacts
	@echo "make forecast: not implemented yet" && exit 1

freeze-forecast:  ## forecast + write ledger/$$(date +%F).json
	@echo "make freeze-forecast: not implemented yet" && exit 1

figures:  ## regenerate README figures from artifacts
	@echo "make figures: not implemented yet" && exit 1

reproduce: data fit backtest forecast figures  ## end-to-end, seeded

# Coverage gate (--cov --cov-fail-under=85) switches on once the model packages
# carry enough statements to measure; pytest-cov errors out on an empty package.
test:  ## ruff + mypy + pytest
	$(PY) -m ruff check src tests
	$(PY) -m mypy
	$(PY) -m pytest

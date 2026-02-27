UV ?= uv

.PHONY: sync install lock help run-carob-baseline run-carob-feature-prepare run-carob-model-compare run-carob-interpretability run-carob-interpretability-report run-carob-rule-causal-aipw run-baseline run-feature-prepare run-model-compare run-causal-pilot run-model-select-tune run-ablation-eval run-interpretability-report test lint format typecheck verify clean

sync:
	$(UV) sync --all-groups

install: sync

lock:
	$(UV) lock

help:
	@echo "Primary CAROB targets:"
	@echo "  make run-carob-baseline"
	@echo "  make run-carob-feature-prepare"
	@echo "  make run-carob-model-compare"
	@echo "  make run-carob-interpretability"
	@echo "  make run-carob-rule-causal-aipw"
	@echo ""
	@echo "Legacy paddy targets (kept for compatibility):"
	@echo "  make run-model-select-tune"
	@echo "  make run-ablation-eval"
	@echo "  make run-interpretability-report"

run-carob-baseline:
	$(UV) run python src/paddy_yield_ml/pipelines/carob_baseline.py

run-carob-feature-prepare:
	$(UV) run python src/paddy_yield_ml/pipelines/carob_feature_prepare.py

run-carob-model-compare:
	$(UV) run python src/paddy_yield_ml/pipelines/carob_model_compare.py

run-carob-interpretability:
	$(UV) run python src/paddy_yield_ml/pipelines/carob_interpretability.py --run-tag iter3_defensible_v5

run-carob-interpretability-report:
	$(UV) run python src/paddy_yield_ml/pipelines/carob_interpretability_report.py --run-tag latest

run-carob-rule-causal-aipw:
	$(UV) run python src/paddy_yield_ml/pipelines/carob_rule_causal_aipw.py --run-tag rule_aipw_v2

# Deprecated aliases: prefer run-carob-* targets for active CAROB workflows.
run-baseline:
	@echo "Deprecated alias: use 'make run-carob-baseline'"
	$(MAKE) run-carob-baseline

run-feature-prepare:
	@echo "Deprecated alias: use 'make run-carob-feature-prepare'"
	$(MAKE) run-carob-feature-prepare

run-model-compare:
	@echo "Deprecated alias: use 'make run-carob-model-compare'"
	$(MAKE) run-carob-model-compare

run-causal-pilot:
	@echo "Deprecated alias: use 'make run-carob-rule-causal-aipw'"
	$(MAKE) run-carob-rule-causal-aipw

run-model-select-tune:
	$(UV) run python src/paddy_yield_ml/pipelines/model_select_tune.py

run-ablation-eval:
	$(UV) run python src/paddy_yield_ml/pipelines/ablation_eval.py

run-interpretability-report:
	$(UV) run python src/paddy_yield_ml/pipelines/interpretability_report.py

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests scripts streamlit_app.py

format:
	$(UV) run ruff format src tests scripts streamlit_app.py

typecheck:
	$(UV) run ty check src tests

verify: lint typecheck test

clean:
	$(UV) run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.mypy_cache','htmlcov']]"

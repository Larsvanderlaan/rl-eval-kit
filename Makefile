PYTHON ?= $(if $(wildcard ./.venv/bin/python),./.venv/bin/python,python3)
PYTEST ?= $(PYTHON) -m pytest
LATEXMK ?= latexmk
TEXINPUTS_NEURIPS_BELLMAN := ../../common/tex//:

.PHONY: test-packages test-bellman-trees test-calibration smoke-fqe test-soft-fqi smoke-irl-conference smoke-irl-journal check-assets papers paper-fqe paper-calibration paper-softfqi paper-bat paper-irl-conference paper-irl-journal

test-packages:
	PYTHONPATH=.:packages/fqe:packages/occupancy-ratio $(PYTEST) packages/fqe/tests packages/occupancy-ratio/occupancy_ratio_benchmark/tests -q

test-bellman-trees:
	PYTHONPATH=packages/bellman-trees $(PYTEST) packages/bellman-trees/tests -q -m "not slow"

test-calibration:
	PYTHONPATH=.:experiments/neurips_bellman/fqe_stationary_weighting:experiments/neurips_bellman/hopper_fqe_benchmark MPLCONFIGDIR=/tmp/rltools_neurips_bellman_mplconfig $(PYTEST) experiments/neurips_bellman/value_calibration/FQE_calibration_neurips/tests -q

smoke-fqe:
	PYTHONPATH=. $(PYTHON) -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage smoke --output-root /tmp/rltools_fqe_smoke

test-soft-fqi:
	PYTHONPATH=experiments/neurips_bellman/soft_fqi_stationary_weighting $(PYTEST) experiments/neurips_bellman/soft_fqi_stationary_weighting/tests -q

smoke-irl-conference:
	PYTHONPATH=experiments/irl/conference_genpqr/repro $(PYTHON) experiments/irl/conference_genpqr/repro/experiments/run_paper_experiments.py --replicates 1 --jobs 1 --output-dir /tmp/irl_genpqr_repro_main_check
	PYTHONPATH=experiments/irl/conference_genpqr/repro $(PYTHON) experiments/irl/conference_genpqr/repro/experiments/action_scaling_runner.py --replicates 1 --jobs 1 --train-trajectories 100 --test-trajectories 50 --horizon 5 --action-counts 5 --output-dir /tmp/irl_genpqr_repro_action_check

smoke-irl-journal:
	PYTHONPATH=.:experiments/irl/journal_debiased_irl:experiments/irl/conference_genpqr/repro MPLCONFIGDIR=/tmp/rl_evaluation_suite_mplconfig $(PYTHON) -c "import experiments.irl.journal_debiased_irl.jrssb_simulation as js; print('journal IRL module import ok')"

check-assets:
	$(PYTHON) tools/check_assets.py

papers: paper-fqe paper-calibration paper-softfqi paper-bat paper-irl-conference paper-irl-journal

paper-fqe:
	cd papers/neurips_bellman/papers/fqe && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-calibration:
	cd papers/neurips_bellman/papers/calibration && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-softfqi:
	cd papers/neurips_bellman/papers/soft_fqi_stationary_weighting && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-bat:
	cd papers/neurips_bellman/papers/bellman_aggregation_trees && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-irl-conference:
	cd papers/irl/conference_genpqr && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main_neurips.tex

paper-irl-journal:
	cd papers/irl/journal_debiased_irl && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main_jasa.tex

PYTHON ?= $(if $(wildcard ./.venv/bin/python),./.venv/bin/python,python3)
PYTEST ?= $(PYTHON) -m pytest
LATEXMK ?= latexmk
TEXINPUTS_NEURIPS_BELLMAN := ../common/tex//:

.PHONY: test-packages test-calibration smoke-fqe test-soft-fqi smoke-irl-conference smoke-irl-journal check-assets papers paper-fqe paper-calibration paper-softfqi paper-bat paper-irl-conference paper-irl-journal

test-packages:
	PYTHONPATH=.:packages/fqe:packages/genpqr:packages/occupancy-ratio:packages/causal-ope-benchmark $(PYTEST) packages/fqe/tests packages/genpqr/tests packages/occupancy-ratio/occupancy_ratio_benchmark/tests packages/causal-ope-benchmark/tests -q

test-calibration:
	PYTHONPATH=.:submissions/neurips-bellman/experiments/fqe_stationary_weighting:submissions/neurips-bellman/experiments/hopper_fqe_benchmark:submissions/neurips-bellman/experiments/value_calibration MPLCONFIGDIR=/tmp/rlevalkit_neurips_bellman_mplconfig $(PYTEST) submissions/neurips-bellman/experiments/value_calibration/FQE_calibration_neurips/tests -q

smoke-fqe:
	PYTHONPATH=submissions/neurips-bellman/experiments/fqe_stationary_weighting $(PYTHON) -m FQE_neurips.controlled_discounted_benchmark.run_experiment --stage smoke --output-root /tmp/rlevalkit_fqe_smoke

test-soft-fqi:
	PYTHONPATH=submissions/neurips-bellman/experiments/soft_fqi_stationary_weighting $(PYTEST) submissions/neurips-bellman/experiments/soft_fqi_stationary_weighting/tests -q

smoke-irl-conference:
	PYTHONPATH=submissions/irl/experiments/conference_genpqr/repro $(PYTHON) submissions/irl/experiments/conference_genpqr/repro/experiments/run_paper_experiments.py --replicates 1 --jobs 1 --output-dir /tmp/irl_genpqr_repro_main_check
	PYTHONPATH=submissions/irl/experiments/conference_genpqr/repro $(PYTHON) submissions/irl/experiments/conference_genpqr/repro/experiments/action_scaling_runner.py --replicates 1 --jobs 1 --train-trajectories 100 --test-trajectories 50 --horizon 5 --action-counts 5 --output-dir /tmp/irl_genpqr_repro_action_check

smoke-irl-journal:
	PYTHONPATH=submissions/irl/experiments/journal_debiased_irl:submissions/irl/experiments/conference_genpqr/repro MPLCONFIGDIR=/tmp/rlevalkit_mplconfig $(PYTHON) -c "import jrssb_simulation as js; print('journal IRL module import ok')"

check-assets:
	$(PYTHON) tools/check_assets.py

papers: paper-fqe paper-calibration paper-softfqi paper-bat paper-irl-conference paper-irl-journal

paper-fqe:
	cd submissions/neurips-bellman/papers/fqe && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-calibration:
	cd submissions/neurips-bellman/papers/calibration && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-softfqi:
	cd submissions/neurips-bellman/papers/soft_fqi_stationary_weighting && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-bat:
	cd submissions/neurips-bellman/papers/bellman_aggregation_trees && TEXINPUTS="$(TEXINPUTS_NEURIPS_BELLMAN)" $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

paper-irl-conference:
	cd submissions/irl/papers/conference_genpqr && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main_neurips.tex

paper-irl-journal:
	cd submissions/irl/papers/journal_debiased_irl && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main_jasa.tex

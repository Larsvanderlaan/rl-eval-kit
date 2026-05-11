#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.scripts.audit_rescue_submission import audit_rescue_submission  # noqa: E402
from FQE_calibration_neurips.scripts.inspect_paper_draft import inspect_paper_draft  # noqa: E402
from FQE_calibration_neurips.scripts.run_suite import run_suite  # noqa: E402
from FQE_calibration_neurips.src.plotting import make_plots  # noqa: E402


DEFAULT_REPLICATIONS = {
    "debug": 1,
    "pilot": 5,
    "confirm": 10,
    "final": 100,
}


def run_rescue_stage(
    *,
    stage: str,
    suite_config: str | Path,
    replications: int | None = None,
    results_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    continue_on_failure: bool = False,
    skip_validation_gate: bool = False,
    skip_plots: bool = False,
) -> dict[str, Path | list[Path]]:
    stage = str(stage)
    reps = DEFAULT_REPLICATIONS.get(stage) if replications is None else int(replications)
    results = run_suite(
        suite_config,
        mode=stage,
        replications=reps,
        continue_on_failure=continue_on_failure,
        skip_validation_gate=skip_validation_gate,
        results_dir=results_dir,
    )
    figures = Path(figures_dir) if figures_dir is not None else ROOT / "figures" / results.name
    made: list[Path] = []
    if not skip_plots:
        made = make_plots(results, figures, allow_invalid=stage in {"debug", "pilot", "confirm"})
        inspect_paper_draft(results, figures)
    audit_outputs = audit_rescue_submission(results)
    return {"results_dir": results, "figures_dir": figures, "plots": made, **audit_outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one staged rescue experiment and audit it.")
    parser.add_argument("--stage", choices=["debug", "pilot", "confirm", "final"], default="debug")
    parser.add_argument(
        "--suite_config",
        default=str(ROOT / "configs/rescue_neurips_suite.yaml"),
        help="Staged rescue suite config.",
    )
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--figures_dir", default=None)
    parser.add_argument("--continue_on_failure", action="store_true")
    parser.add_argument("--skip_validation_gate", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()
    outputs = run_rescue_stage(
        stage=args.stage,
        suite_config=args.suite_config,
        replications=args.replications,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        continue_on_failure=args.continue_on_failure,
        skip_validation_gate=args.skip_validation_gate,
        skip_plots=args.skip_plots,
    )
    for name, value in outputs.items():
        if name == "plots":
            print(f"Wrote plots: {len(value)}")
        else:
            print(f"Wrote {name}: {value}")


if __name__ == "__main__":
    main()

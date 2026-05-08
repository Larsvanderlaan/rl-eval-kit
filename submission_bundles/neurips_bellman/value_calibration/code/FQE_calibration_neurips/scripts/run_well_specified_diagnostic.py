#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.scripts.run_experiment import run_config  # noqa: E402
from FQE_calibration_neurips.src.utils import ensure_dir, load_config  # noqa: E402
from FQE_calibration_neurips.src.validation import GateConfig, evaluate_well_specified_gate  # noqa: E402


def summarize_diagnostic(rows: list[dict], output_dir: Path, bias_threshold: float, mse_threshold: float) -> Path:
    gate_passed, merged = evaluate_well_specified_gate(
        rows,
        output_dir,
        GateConfig(bias_threshold=float(bias_threshold), mse_threshold=float(mse_threshold)),
    )
    out = output_dir / "well_specified_diagnostic_summary.csv"
    merged.to_csv(out, index=False)
    warnings = merged[merged["pass_well_specified"] == False]  # noqa: E712
    warning_path = output_dir / "well_specified_warnings.txt"
    with warning_path.open("w") as handle:
        if warnings.empty:
            handle.write("All well-specified diagnostic groups passed configured thresholds.\n")
        else:
            handle.write("Well-specified diagnostic failures; do not use these rows as evidence without explanation.\n")
            handle.write(warnings.to_string(index=False))
            handle.write("\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the required well-specified correctness diagnostic.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/well_specified_debug.yaml"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results/well_specified_debug"))
    parser.add_argument("--bias_threshold", type=float, default=2.0)
    parser.add_argument("--mse_threshold", type=float, default=8.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    config = load_config(args.config)
    rows = run_config(config, output_dir, debug=args.debug, run_mode="debug" if args.debug else "standalone", suite_name="well_specified_debug")
    out = summarize_diagnostic(rows, output_dir, args.bias_threshold, args.mse_threshold)
    print(f"Wrote well-specified diagnostic summary to {out}")


if __name__ == "__main__":
    main()

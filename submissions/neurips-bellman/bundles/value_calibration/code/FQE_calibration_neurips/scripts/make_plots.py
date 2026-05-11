#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.src.plotting import make_plots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Make publication-quality calibration plots.")
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--figures_dir", type=str, default=str(ROOT / "figures"))
    parser.add_argument("--allow_invalid", action="store_true", help="Allow paper plots even if the validation gate failed.")
    args = parser.parse_args()
    made = make_plots(args.results_dir, args.figures_dir, allow_invalid=args.allow_invalid)
    print(f"Wrote {len(made)} plot files to {args.figures_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled soft-FQI stationary-weighting experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--resume", action="store_true", help="Skip completed run keys in an existing raw_results.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing raw_results.csv instead of refusing to run.")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive.")
    raw_path = run_experiment(args.config, resume=args.resume, overwrite=args.overwrite)
    print(f"Wrote raw results to {raw_path}")


if __name__ == "__main__":
    main()

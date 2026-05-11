#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.src.aggregation import aggregate_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate replication-level calibration results.")
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--no_tables", action="store_true")
    args = parser.parse_args()
    out = aggregate_results(args.results_dir, write_tables=not args.no_tables)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

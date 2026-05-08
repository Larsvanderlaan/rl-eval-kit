from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path


def ensure_occupancy_ratio_path() -> None:
    warnings.warn(
        "IRL.fit_occupancy_ratio* imports are deprecated; use the installed "
        "`occupancy_ratio` package instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if importlib.util.find_spec("occupancy_ratio") is not None:
        return
    package_root = Path(__file__).resolve().parents[1] / "RL-Evaluation" / "occupancy-ratio"
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)

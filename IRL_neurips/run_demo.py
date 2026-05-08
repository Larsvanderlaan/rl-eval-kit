from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "experiments" / "irl" / "conference_genpqr" / "repro" / "run_demo.py"
runpy.run_path(str(_TARGET), run_name="__main__")

"""Compatibility package for the relocated GenPQR conference experiments."""

from __future__ import annotations

from pathlib import Path
import sys

_TARGET = Path(__file__).resolve().parents[1] / "experiments" / "irl" / "conference_genpqr" / "repro"
__path__ = [str(_TARGET)]
__file__ = str(_TARGET / "__init__.py")
if str(_TARGET) not in sys.path:
    sys.path.insert(0, str(_TARGET))

with open(__file__, "rb") as _handle:
    exec(compile(_handle.read(), __file__, "exec"), globals(), globals())

"""Compatibility package for the relocated value-calibration experiments."""

from __future__ import annotations

from pathlib import Path

_TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "neurips_bellman"
    / "value_calibration"
    / "FQE_calibration_neurips"
)
__path__ = [str(_TARGET)]

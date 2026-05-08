"""Compatibility package for the relocated Hopper FQE benchmark."""

from __future__ import annotations

from pathlib import Path

_TARGET = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "neurips_bellman"
    / "hopper_fqe_benchmark"
    / "hopper_fqe_benchmark"
)
__path__ = [str(_TARGET)]
__file__ = str(_TARGET / "__init__.py")

with open(__file__, "rb") as _handle:
    exec(compile(_handle.read(), __file__, "exec"), globals(), globals())

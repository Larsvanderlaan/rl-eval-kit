"""Test configuration for the self-contained GenPQR reproducibility code."""

from __future__ import annotations

import sys
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPRO_ROOT))

from __future__ import annotations

import genpqr
from genpqr import GenPQRConfig, GenPQRConfigurationError, list_presets


def test_version_and_presets_are_public() -> None:
    assert isinstance(genpqr.__version__, str)
    presets = list_presets()
    assert "deeppqr_linear" in presets
    assert isinstance(GenPQRConfig.from_preset("bc_boosted_fast"), GenPQRConfig)


def test_unknown_preset_errors_cleanly() -> None:
    try:
        GenPQRConfig.from_preset("missing")
    except GenPQRConfigurationError as exc:
        assert "Unknown GenPQR preset" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown preset did not fail")

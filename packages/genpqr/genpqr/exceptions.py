"""Exceptions raised by :mod:`genpqr`."""

from __future__ import annotations


class GenPQRError(Exception):
    """Base class for GenPQR errors."""


class GenPQRConfigurationError(GenPQRError, ValueError):
    """Raised when a requested configuration is internally inconsistent."""


class GenPQRMissingDependencyError(GenPQRError, ImportError):
    """Raised when a lazy optional backend is requested but unavailable."""


class GenPQRAdapterError(GenPQRError):
    """Raised when an external adapter cannot produce the required contract."""

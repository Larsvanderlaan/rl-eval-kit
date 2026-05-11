from __future__ import annotations


class BenchmarkError(Exception):
    """Base exception for public benchmark-package errors."""


class ConfigurationError(BenchmarkError, ValueError):
    """Raised when a benchmark configuration or user request is invalid."""


class AdapterValidationError(BenchmarkError, ValueError):
    """Raised when adapter inputs or outputs fail public shape checks."""


class MissingOptionalDependency(BenchmarkError, RuntimeError):
    """Raised when an optional integration is requested but unavailable."""

    def __init__(self, dependency: str, message: str | None = None) -> None:
        self.dependency = str(dependency)
        detail = message or f"Optional dependency '{dependency}' is not installed."
        super().__init__(detail)

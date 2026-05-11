"""Backend registry for GenPQR policy and Q estimators."""

from __future__ import annotations

from typing import Any, Callable

from genpqr.exceptions import GenPQRConfigurationError


Factory = Callable[..., Any]
_POLICY_REGISTRY: dict[str, Factory] = {}
_Q_REGISTRY: dict[str, Factory] = {}


def _normalize_name(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("registry names must be nonempty.")
    return key


def register_policy_estimator(name: str, factory: Factory, *, overwrite: bool = False) -> None:
    """Register a named policy-estimator factory."""

    key = _normalize_name(name)
    if key in _POLICY_REGISTRY and not overwrite:
        raise GenPQRConfigurationError(f"Policy estimator '{name}' is already registered.")
    _POLICY_REGISTRY[key] = factory


def register_q_estimator(name: str, factory: Factory, *, overwrite: bool = False) -> None:
    """Register a named Q-estimator factory."""

    key = _normalize_name(name)
    if key in _Q_REGISTRY and not overwrite:
        raise GenPQRConfigurationError(f"Q estimator '{name}' is already registered.")
    _Q_REGISTRY[key] = factory


def available_policy_estimators() -> tuple[str, ...]:
    """Return registered policy-estimator names."""

    return tuple(sorted(_POLICY_REGISTRY))


def available_q_estimators() -> tuple[str, ...]:
    """Return registered Q-estimator names."""

    return tuple(sorted(_Q_REGISTRY))


def resolve_registered_policy_estimator(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered policy estimator."""

    key = _normalize_name(name)
    try:
        factory = _POLICY_REGISTRY[key]
    except KeyError as exc:
        raise GenPQRConfigurationError(f"Unknown policy estimator '{name}'.") from exc
    return factory(**kwargs)


def registered_policy_estimator_factory(name: str) -> Factory:
    """Return a registered policy factory without instantiating it."""

    key = _normalize_name(name)
    try:
        return _POLICY_REGISTRY[key]
    except KeyError as exc:
        raise GenPQRConfigurationError(f"Unknown policy estimator '{name}'.") from exc


def resolve_registered_q_estimator(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered Q estimator."""

    key = _normalize_name(name)
    try:
        factory = _Q_REGISTRY[key]
    except KeyError as exc:
        raise GenPQRConfigurationError(f"Unknown Q estimator '{name}'.") from exc
    return factory(**kwargs)

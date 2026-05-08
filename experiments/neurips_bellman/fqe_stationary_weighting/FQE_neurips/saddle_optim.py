from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


Array = np.ndarray


@dataclass
class ExtragradientConfig:
    """Configuration for Euclidean extragradient / Mirror-Prox updates."""

    step_size: float
    max_iters: int = 2_000
    tol: float = 1e-7
    averaging: bool = True
    log_every: int = 100
    project_x: Optional[Callable[[Array], Array]] = None
    project_y: Optional[Callable[[Array], Array]] = None


@dataclass
class SaddlePointResult:
    """Output of a saddle-point solver."""

    x: Array
    y: Array
    history: dict[str, list[float]] = field(default_factory=dict)


def _maybe_project(x: Array, projector: Optional[Callable[[Array], Array]]) -> Array:
    if projector is None:
        return x
    return projector(x)


def extragradient(
    x0: Array,
    y0: Array,
    grad_x: Callable[[Array, Array], Array],
    grad_y: Callable[[Array, Array], Array],
    config: ExtragradientConfig,
    objective: Optional[Callable[[Array, Array], float]] = None,
) -> SaddlePointResult:
    """
    Solve min_x max_y L(x, y) using the extragradient / Mirror-Prox update.

    This is a standard stable choice for smooth monotone saddle-point problems and
    is substantially more reliable than plain simultaneous gradient descent/ascent.
    """

    x = np.asarray(x0, dtype=np.float64).copy()
    y = np.asarray(y0, dtype=np.float64).copy()

    avg_x = np.zeros_like(x)
    avg_y = np.zeros_like(y)
    history: dict[str, list[float]] = {"gap_proxy": []}
    if objective is not None:
        history["objective"] = []

    for it in range(1, config.max_iters + 1):
        gx = grad_x(x, y)
        gy = grad_y(x, y)

        x_mid = _maybe_project(x - config.step_size * gx, config.project_x)
        y_mid = _maybe_project(y + config.step_size * gy, config.project_y)

        gx_mid = grad_x(x_mid, y_mid)
        gy_mid = grad_y(x_mid, y_mid)

        x_new = _maybe_project(x - config.step_size * gx_mid, config.project_x)
        y_new = _maybe_project(y + config.step_size * gy_mid, config.project_y)

        step_norm = float(
            np.sqrt(np.linalg.norm(x_new - x) ** 2 + np.linalg.norm(y_new - y) ** 2)
        )
        history["gap_proxy"].append(step_norm)
        if objective is not None and (it == 1 or it % config.log_every == 0 or it == config.max_iters):
            history["objective"].append(float(objective(x_new, y_new)))

        x, y = x_new, y_new
        if config.averaging:
            weight = 1.0 / it
            avg_x = (1.0 - weight) * avg_x + weight * x
            avg_y = (1.0 - weight) * avg_y + weight * y

        if step_norm < config.tol:
            break

    if config.averaging and it > 0:
        x_out, y_out = avg_x, avg_y
    else:
        x_out, y_out = x, y

    return SaddlePointResult(x=x_out, y=y_out, history=history)


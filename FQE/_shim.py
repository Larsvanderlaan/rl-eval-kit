from __future__ import annotations

from importlib import import_module


def legacy_exports(module_name: str) -> dict[str, object]:
    module = import_module(f"legacy.FQE.{module_name}")
    return {
        name: value
        for name, value in module.__dict__.items()
        if not (name.startswith("__") and name.endswith("__"))
    }

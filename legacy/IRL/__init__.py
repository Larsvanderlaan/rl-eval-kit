from __future__ import annotations

from importlib import import_module

__all__ = [
    "JRSSBConfig",
    "JRSSBOracle",
    "run_bias_acceptance",
    "run_example1a_policy_audit",
    "run_example1b_policy_audit",
    "run_example2_method_selection",
    "run_example2_nuisance_audit",
    "run_example2_paper_comparison",
    "run_example2_semi_oracle_audit",
    "run_example2_shakedown",
    "run_example2_smoke_check",
    "run_monte_carlo",
    "run_single_replication",
    "run_validation_suite",
    "summarize_results",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module("legacy.IRL.jrssb_simulation")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Package Architecture

The public package is organized around stable user APIs and smaller
contributor-facing implementation modules.

## Public Entry Points

Use these import paths in user code:

```python
from occupancy_ratio import fit_discounted_occupancy_ratio
from occupancy_ratio import fit_discounted_occupancy_ratio_neural
from occupancy_ratio import tune_occupancy_ratio_auto
```

The facade modules are also stable:

- `occupancy_ratio.boosted`
- `occupancy_ratio.neural`
- `occupancy_ratio.nuisance`
- `occupancy_ratio.tuning`

Compatibility modules remain available:

- `occupancy_ratio.fit_occupancy_ratio`
- `occupancy_ratio.fit_occupancy_ratio_neural`

These paths preserve existing research code and tests that import lower-level
helpers.

## Boosted Implementation

Boosted FORI code is grouped by responsibility:

| Module | Responsibility |
| --- | --- |
| `configs` | Boosted config dataclasses and presets. |
| `models` | Fitted boosted model class and prediction helpers. |
| `validation` | Shape checks, mode resolution, and terminal/timeout handling. |
| `nuisance_lgbm` | LightGBM nuisance-ratio fitting and source-ratio helpers. |
| `targets` | Fixed-point target builders and transition caches. |
| `stabilization` | Clipping, projection, damping, Huber loss, ESS, and summaries. |
| `_boosted_impl` | Private implementation backing compatibility imports. |

## Neural Implementation

Neural FORI follows the same structure with fewer modules:

| Module | Responsibility |
| --- | --- |
| `neural_configs` | Neural config dataclasses and presets. |
| `neural_models` | Fitted neural model class and compatibility alias. |
| `neural_nuisance` | Neural action/source/transition nuisance fitters. |
| `neural_targets` | Neural target builders and predictor internals. |
| `neural_fit` | Neural fit and CV entrypoints. |
| `_neural_impl` | Private implementation backing compatibility imports. |

## Tuning Implementation

`occupancy_ratio.tuning` remains the public API. Internals are grouped for
debugging:

| Module | Responsibility |
| --- | --- |
| `_tuning_candidates` | Candidate expansion, labeling, and budget caps. |
| `_tuning_cv` | Fold construction and candidate evaluation. |
| `_tuning_refit` | Final refit and stable fallback guardrails. |
| `_tuning_scoring` | Proxy score components and telemetry helpers. |
| `_tuning_impl` | Private implementation backing compatibility imports. |

## Refactor Contract

This layout is behavior-preserving. Defaults, fit signatures, lazy optional
dependencies, legacy dictionaries, and serialization-compatible public module
paths should stay stable.

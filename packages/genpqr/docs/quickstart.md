# GenPQR Quickstart

Use `fit_genpqr(...)` with row arrays or a `TransitionDataset`. Core import is
NumPy-only; optional learners are loaded only when selected.

```python
from genpqr import DiscreteNormalizationPolicy, GenPQRConfig, fit_genpqr

result = fit_genpqr(
    dataset=dataset,
    gamma=0.95,
    normalization_policy=DiscreteNormalizationPolicy.uniform(2),
    config=GenPQRConfig.from_preset("bc_boosted_fast"),
)
```

For neural DeepGenPQR workflows, the pooled default is action-space aware:

```python
from genpqr import DeepGenPQRConfig, fit_deep_genpqr

result = fit_deep_genpqr(
    dataset=dataset,
    gamma=0.95,
    env=env,
    config=DeepGenPQRConfig(q_mode="pooled_fqe"),
)
```

Finite-action data use `auto_neural_fqe`, which routes to action-head neural
FQE. This is the recommended pooled default when anchor actions are rare or when
you want a reusable all-action Q model. Use `q_mode="anchor_deeppqr"` when the
anchor normalization is central and anchor support is healthy.

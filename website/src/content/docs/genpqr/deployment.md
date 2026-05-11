---
title: genPQR Deployment
description: Serialization, diagnostics, and lazy dependencies.
---

## Serialization

```python
from genpqr import load_genpqr_result

result.save("genpqr-result")
loaded = load_genpqr_result("genpqr-result")
```

DeepGenPQR results use the same manifest-backed save path when the fitted
policy, Q, and normalization objects support serialization:

```python
from genpqr import load_deep_genpqr_result

result.save("deepgenpqr-result")
loaded = load_deep_genpqr_result("deepgenpqr-result")
```

Prefer manifest-backed artifacts for portable results. Treat pickle-based
fallbacks as unsafe for untrusted files and require an explicit `allow_pickle`
opt-in when loading them.

## Report checks before use

- Normalization policy and action-space contract.
- Policy-estimation diagnostics and optional adapter status.
- Q-estimation diagnostics from FQE or the selected custom estimator.
- Anchor count, weighted anchor count, anchor fraction, and weak-support flags
  when using DeepPQR anchors.
- Reward quantiles and Monte Carlo normalization standard errors for
  continuous-action workflows.

## Optional dependency policy

The core import remains NumPy-only. External learners live behind lazy adapters
with clear missing-dependency errors. Treat any configured fallback as a
separate method choice rather than an equivalent result.

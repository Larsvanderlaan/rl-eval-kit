# Serialization

Use `result.save(path)` or `save_genpqr_result(result, path)`. GenPQR writes a
JSON manifest plus portable NumPy arrays when the fitted objects are native
GenPQR objects. Unsafe pickle fallback is labeled in the manifest, and
`load_genpqr_result(path)` refuses pickle artifacts unless `allow_pickle=True`.

DeepGenPQR uses the same safety model:

```python
from genpqr import load_deep_genpqr_result, save_deep_genpqr_result

result.save("deepgenpqr-result")
loaded = load_deep_genpqr_result("deepgenpqr-result")
save_deep_genpqr_result(result, "deepgenpqr-result-copy")
```

Portable DeepGenPQR artifacts store the DeepGenPQR mode/config metadata and the
underlying portable GenPQR payload. The finite-action action-head neural FQE
backend is portable when the policy and normalization policy are portable; the
manifest records the resolved `q_backend` as `action_head_neural_fqe`.

If an external policy or Q object is not portable, saving falls back to a
labeled unsafe pickle artifact when possible. Loading then requires
`allow_pickle=True` and should be limited to trusted local artifacts.

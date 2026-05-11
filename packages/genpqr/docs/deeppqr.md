# DeepPQR

`DeepPQRAnchorQEstimator` is the NumPy anchor-value backend.
`NeuralDeepPQRAnchorQEstimator` is a lazy Torch backend for state-only `W(s)`
with action-stratified reconstruction by behavior-policy log-ratios.

Use `fit_deep_genpqr(..., config=DeepGenPQRConfig(q_mode="anchor_deeppqr"))`
when you want the production DeepGenPQR workflow with this anchor-Q
parameterization. Finite-action anchors are exact action indices. Continuous
anchors may be fixed action vectors or callable `anchor_action(states)` values.
An `anchor_selector` can filter genuine anchor rows, but selected rows must
still match `anchor_action` within `anchor_tolerance`; use callable
`anchor_action` for state-dependent anchors instead of relabeling arbitrary
non-anchor rows. `min_anchor_count` guards weak support.

## Practical Guidance

Anchor DeepPQR fits the anchor value only on rows whose observed action matches
the anchor. This is a strength when anchor support is healthy: the
normalization is explicit, and the full stratified Q is reconstructed through
policy log-ratios. It is also the main limitation: if anchor rows are rare or
clustered in a narrow part of state space, the anchor value is not reliably
identified.

The backend reports anchor counts, weighted anchor counts, anchor fractions, and
weak-support flags. Treat weak support as a correctness warning, not just a
statistical nuisance. For controlled experiments, keep
`anchor_fallback="error"` so weak support fails loudly. For production pipelines
where a result is preferable to a failure, use `anchor_fallback="pooled_fqe"` and
review the warning diagnostics.

When anchors are rare but action values can be pooled reliably, prefer
DeepGenPQR's default `q_mode="pooled_fqe"` with `q_backend="auto_neural_fqe"`.
For finite actions this uses action-head neural FQE, which shares a state trunk
while preserving action-specific residual heads. It is a practical fallback for
smooth action-reward surfaces, but it is still extrapolation; rough action
surfaces or near-zero anchor support require extra scrutiny.

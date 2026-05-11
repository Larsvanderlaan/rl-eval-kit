# Deployment Checklist

- Keep `import genpqr` NumPy-only.
- Use `TransitionDataset` or `EpisodeDataset` at API boundaries.
- Run contract checks for custom estimators before registering them.
- Prefer safe portable serialization; use pickle loading only for trusted local artifacts.
  The finite-action action-head neural FQE backend has portable save/load
  support when the policy and normalization policy are portable; nonportable
  policies use the labeled unsafe pickle fallback only by explicit opt-in on
  load.
- Use cross-fitting for statistical screens and inspect fold instability diagnostics.
- Prefer `fit_deep_genpqr(...)` for neural production workflows:
  `q_mode="pooled_fqe"` with `q_backend="auto_neural_fqe"` is the default.
  Finite-action data use action-head neural FQE; continuous-action data use
  generic neural FQE. Anchor/subset DeepPQR should be enabled only when anchor
  support diagnostics are healthy or when `anchor_fallback="pooled_fqe"` is an
  acceptable production fallback.
- For finite-action deployments, inspect action-count diagnostics and compare
  per-action reward behavior on representative screens. Action-head FQE is a
  safer pooled default, but it can still extrapolate poorly when actions are
  essentially unobserved or the action-reward surface is rough.
- For continuous anchor-mode deployments, use `anchor_action(states)` for
  state-dependent anchors. `anchor_selector` is a row filter for genuine anchor
  observations, and selected rows must match `anchor_action` within
  `anchor_tolerance`.
- Run `run_tiny_production_validation()` before changing defaults and inspect
  reward finite rates, normalization residuals, anchor support, runtime, and
  fold instability.
- Enable optional integration tests with `GENPQR_RUN_OPTIONAL_INTEGRATION=1`
  only in environments with the relevant third-party stacks installed.

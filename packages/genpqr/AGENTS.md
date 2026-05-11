# GenPQR Package Notes

Keep this package as a production interface, not a dump of experiment scripts.

- Core import must stay dependency-light: `import genpqr` may import NumPy, but
  must not import Torch, Gym, SB3, `imitation`, d3rlpy, or SCOPE-RL.
- Public APIs should validate array shapes at the boundary and use explicit
  action-space contracts for discrete versus continuous actions.
- External learners belong behind lazy adapters with clear preflight errors.
- User-supplied policy and Q estimators are first-class. Do not make users
  subclass internal base classes if they satisfy the protocol.
- Preserve the DeepPQR anchor-Q parameterization as a separate backend. It
  estimates the state-only anchor value on anchor rows and reconstructs the full
  stratified Q by policy log-ratios; do not silently replace it with pooled FQE.
- Public functions/classes need NumPy-style docstrings.
- Add targeted tests for changes to normalization, action encoding, reward
  recovery, default policy/Q resolution, DeepPQR reconstruction, optional
  dependency behavior, and continuous-action Monte Carlo normalization.

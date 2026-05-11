# Datasets

`TransitionDataset.from_arrays(...)` validates row-wise transitions.
`EpisodeDataset.from_episodes(...)` preserves episode boundaries and can be
flattened for fitters. Cross-fitting respects episode ids by default.

Use `strict_episodes=True` for adapters that require ordered trajectories.
`TransitionDataset.from_d3rlpy(...)` and `TransitionDataset.from_scope_rl(...)`
convert common logged dataset payloads without importing optional libraries.

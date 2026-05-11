# Custom Estimators

Custom policies and Q functions only need to satisfy the public protocols. Use
`genpqr.testing` contract checks while developing adapters, and optionally
register named factories with `register_policy_estimator` or
`register_q_estimator`.

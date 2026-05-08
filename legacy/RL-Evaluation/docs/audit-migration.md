# Audit And Migration Notes

The estimator code now has production package boundaries around `fqe` and
`occupancy_ratio`. Benchmarks are packaged as optional tools and no longer need
package-time `sys.path` mutation.

Known policy decisions:

- LSIF remains the default density-ratio nuisance objective.
- Logistic nuisance estimation is opt-in for boosted and neural implementations.
- Neural cross-fitting uses fold nuisance predictors inside fixed-point target
  construction and keeps final full-data predictors for user-facing prediction.
- Legacy `FQE.*` and `IRL.fit_occupancy_ratio*` imports are transitional; new
  code should import `fqe` and `occupancy_ratio`.

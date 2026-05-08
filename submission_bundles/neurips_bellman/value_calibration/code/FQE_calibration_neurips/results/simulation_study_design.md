# NeurIPS Calibration Simulation Study Design

## Focused NeurIPS Main-Text Plan

The main-text experiment is intentionally compact. It uses the existing nonlinear
continuous-state, discrete-action MDP family and strict value-space Bellman
calibration, and promotes three organic regimes plus one mechanism diagnostic:

1. **Affine/misspecified linear FQE.** A restricted linear FQE learner is fit
   under moderate behavior-target shift. The intended failure is informative
   value ranking with biased slope/intercept. Linear calibration should be
   competitive with isotonic calibration when the distortion is mostly affine.
2. **Finite-iteration FQE.** Random-feature FQE is stopped before long-horizon
   convergence in a high-discount setting. The intended failure is under-scaled
   or horizon-misaligned values. Calibration should reduce Bellman calibration
   error and true value-function MSE.
3. **Temporal reward shift.** A random-feature FQE model is trained on a large
   old-reward behavior batch and calibrated with a small recent current-reward
   batch. Dynamics, policies, and state distributions are shared; only the
   reward scale changes. Evaluation is only against current-regime true-value
   MSE and Bellman calibration error.
4. **Mechanism-only affine versus monotone distortion.** A strong base learner
   is wrapped in controlled affine and monotone saturation distortions. This is
   a diagnostic panel, not an organic learner claim: it shows when linear
   calibration is enough and when isotonic calibration has a structural
   advantage.

Non-temporal focused rows use `calibration_protocols: [cross]`. The temporal
reward-shift row uses `recent_heldout` calibration: old-regime behavior data
trains the raw FQE model, recent current-regime held-out data fits the
calibrator, and an independent current-regime test/diagnostic set evaluates the
claim. All rows use `calibrators: [linear, isotonic]`,
`calibration_targets: [value_bellman]`, and moderate behavior-target shift.
Saddle/minimax estimators, severe overlap stress, split/no-split comparisons,
Hopper, and hybrid/binning calibrators remain appendix or future-work material
unless a later audit explicitly promotes them.

Promotion criteria are deliberately falsifiable. A row is main-text evidence
only if raw predictions are rank-informative against independent oracle
\(V^\pi(s)\), raw predictions are visibly miscalibrated by slope/intercept or a
reliability curve, calibration improves Bellman calibration error on an
independent diagnostic set, and true-value MSE improves. Improvements that rely
on test/oracle leakage, no-split optimism, or scalar MSE cancellation without
Bellman calibration-error improvement are not promoted.

## Final Rescue Audit Readout

The current submission-facing rescue run writes final artifacts to
`results/rescue_final` and `figures/rescue_final`. Promotion is determined only by
`results/rescue_final/rescue_promotion_audit.csv`.

Audited final labels are:

- `promote_main`: 12 rows from final-stage affine misspecification,
  finite-iteration, and temporal reward-shift suites.
- `mechanism_only`: 4 rows from the affine-vs-monotone diagnostic panel.
- `validation_control`: 3 well-specified control rows, not paper claims.
- `reject_mse_only` / `reject_unstable` / `limitation`: do-not-claim rows.

The defensible main-text story is narrower than "calibration always helps":

1. Affine/misspecified linear FQE produces informative raw values with
   slope/intercept bias. Linear calibration is enough for the main affine claim,
   with isotonic included as a robustness comparison.
2. Finite-iteration random-feature FQE produces informative but severely
   mis-scaled long-horizon values. Both linear and isotonic calibration improve
   true-value MSE and Bellman calibration error over the matched all-data raw
   baseline.
3. Temporal reward shift shows the practical use case: an old-regime value
   model can be recalibrated with a small recent current-regime batch, improving
   current-regime true-value MSE and Bellman calibration error without retraining
   from scratch. The same-small-current retrain comparator uses the same learner
   with a prespecified prediction clip so the comparison is finite rather than
   an artifact of numerical explosion.
4. The mechanism panel validates calibrator choice: linear wins or ties under
   affine distortion, while isotonic is better under monotone saturation.

Rows that improve only scalar MSE without Bellman calibration-error support and
unstable rows are explicitly not promoted. Optional saddle, temporal-proxy, and
broader quality diagnostics are skipped in the final core run and should be
treated as limitations or appendix work until they complete the same final-stage
audit.

## Main Scientific Questions

The central question is whether calibration is a useful wrapper for a broad class of Bellman/FQE-style off-policy evaluation estimators, not whether one calibrated estimator beats one uncalibrated estimator. The simulation study asks:

1. When does post-hoc calibration reduce finite-sample value error for Bellman-style OPE?
2. Are gains consistent across learner families: neural FQE, random-feature FQE, regularized Bellman residual methods, saddle-point Bellman residual methods, and ensembles?
3. Which calibration protocol is most reliable under honest data use: cross-calibration, split-calibration, or no-split calibration?
4. Which calibrator is appropriate for affine, monotone nonlinear, and nonmonotone distortions?
5. How do coverage stress, extrapolation, sample size, and misspecification change the calibration story?
6. When does calibration help little or fail?

## Simulation Environments

The suite uses controlled discounted MDPs with continuous states, discrete actions, nonlinear transitions, nonlinear rewards, explicit behavior and target policies, and independent oracle rollouts. The controlled design is deliberate: main claims should rest on interpretable mechanisms with reliable oracle values, not on opaque benchmark simulators alone.

### Regime 1: Well-Specified Debug

Configuration: `configs/well_specified_debug.yaml`.

This regime uses a simple well-specified linear MDP variant with low reward noise, good coverage, and a target-policy oracle computed by independent Monte Carlo rollouts. Linear FQE and the linear random-feature settings are correctly specified or close to correctly specified.

Why included: it is a required estimator sanity check before interpreting harder results.

Success mode tested: uncalibrated estimators should be finite and accurate, MSE should generally improve with sample size, and calibration should not substantially damage an already accurate baseline.

Failure mode tested: leakage, exploding estimates, nonfinite values, unstable calibration, and calibrators that overfit a well-specified baseline.

### Regime 2: Main Nonlinear Synthetic

Configuration: `configs/main_nonlinear.yaml`.

This is the main paper-evidence regime: continuous states, discrete actions, nonlinear rewards, nonlinear transitions, moderate dimension, stochastic behavior actions, and a target policy shifted away from the behavior policy. Oracle value is computed from independent target-policy rollouts.

The paper-mode configuration also supports a mild target/reference shift through independent shifted initial evaluation states. This keeps behavior-policy support usable while making calibration error matter in the target evaluation region, which is a more scientifically meaningful stressor than relying only on near-positivity collapse.

Why included: it creates realistic nonlinear finite-sample errors while keeping the data-generating process reproducible and interpretable.

Success mode tested: flexible estimators can be good but still miscalibrated, so calibration may reduce value MSE across multiple baseline learner families.

Failure mode tested: calibration can have little benefit when the baseline is already accurate or when the calibration target is noisy.

### Regime 3: Coverage / Policy-Shift Stress

Configuration: `configs/coverage_sweep.yaml`.

This regime varies coverage settings from good to extrapolation stress by changing behavior-target policy shift and overlap. It reports aggregate metrics and coverage-stratified diagnostics by behavior-target density-ratio quantile.

Why included: off-policy evaluation is most fragile under weak overlap, and calibration should not be sold as a positivity cure.

Success mode tested: calibration may correct systematic scale or monotone distortions in covered regions.

Failure mode tested: severe extrapolation can make all Bellman learners unreliable, and calibrators trained on observed support cannot fully repair missing support.

### Regime 3b: Undertraining / Finite-Iteration Bias

Configuration: `configs/undertraining_sweep.yaml`.

This regime uses high discount factors and explicitly under-iterated Bellman learners: one-step reward-fit FQE, two- and four-step bootstrap FQE variants, weak ensembles, and well-tuned controls with many Bellman iterations.

Why included: the original paper mechanism is easiest to see when the raw value predictor has a fixed-point scale or iteration bias that is calibratable, rather than arbitrary noise.

Expected diagnostic signature: `actual_bellman_iterations` is small for the undertrained variants, value bias is large, plug-in Bellman calibration error is high, and value-space calibration can improve both value MSE and Bellman calibration error. Well-tuned controls should show little gain.

Fairness guard: undertraining variants are labeled through `learner_quality_regime`, matched to their own all-data uncalibrated comparator, and are not the only baselines in any main comparison.

### Regime 3c: Organic Bellman-Incomplete Feature Pair

Configuration: `configs/bellman_incomplete_sweep.yaml`.

This regime pairs nonlinear reward/transition mechanisms with restricted linear or low-capacity random-feature learners. The intent is not to distort predictions after fitting, but to create an organic projected Bellman fixed point: the value may be roughly ordered by a low-dimensional summary, while its Bellman image contains nonlinear interactions outside the fitted class.

Why included: this is the cleanest non-artificial setting where calibration has something to fix after a stable Bellman learner has converged.

Expected diagnostic signature: finite estimates, many Bellman iterations, non-exploding optimization, persistent Bellman residual, high calibration error, and partial improvement from value-space calibration. `true_v_mse` may improve less than scalar MSE when the remaining structural error is not one-dimensional.

Fairness guard: flexible neural or sufficiently rich random-feature learners are included as negative controls; calibration gains should be smaller when the first-stage class is adequate.

### Regime 3d: Model-Misspecification Sweep

Configuration: `configs/model_misspecification_sweep.yaml`.

This regime includes restricted linear FQE, low-feature random-feature FQE, small or heavily regularized neural FQE, and overregularized Bellman residual fits. It is separate from the mechanism-only prediction distortion suite.

Why included: misspecification is a realistic source of systematic calibration error, but it should not be confused with artificial post-hoc distortion.

Expected diagnostic signature: stable finite predictions, systematic value bias, elevated `true_v_mse`, and calibration improvements when the error is scale-like or monotone. Nonmonotone misspecification should remain only partially correctable.

### Regime 4: Misspecification / Distortion

Configuration: `configs/misspecification_sweep.yaml`.

This regime separates three distortions:

- `affine`: predictions are distorted approximately by an affine transformation, where linear calibration should be effective.
- `monotone_distortion`: predictions undergo a monotone nonlinear distortion, where isotonic and isotonic-histogram calibration should help.
- `nonmonotone`: misspecification is not globally monotone, where monotone calibrators should help only partially or can fail.
- `bellman_incomplete`: rewards and transitions contain nonlinear components that are not closed under simple linear Bellman feature classes, so fitted values can rank states and actions reasonably while still having systematic Bellman-scale bias.

Why included: it makes the expected success and failure modes of each calibrator scientifically interpretable.

Success mode tested: linear calibration should help affine distortion; isotonic and hybrid calibration should help monotone distortion.

Failure mode tested: monotone calibration is not a universal solution under nonmonotone errors.

### Regime 5: Baseline / Split Appendix

Configurations: `configs/baseline_family_sweep.yaml` and `configs/split_fraction_sweep.yaml`.

The baseline-family sweep compares learner families under a shared data-generating process. The split-fraction sweep compares split calibration against same-fraction uncalibrated baselines, all-data uncalibrated baselines, fine-tuning, offset correction, residual correction, and regularization toward the first-stage model.

Why included: it separates calibration gains from data-use artifacts and quantifies the cost of reserving calibration data.

Success mode tested: cross-calibration can preserve all-data final training while maintaining leakage-safe calibration targets.

Failure mode tested: split calibration may lose power when the first-stage learner is data hungry, and no-split calibration may be optimistic.

### Regime 6: Calibration Quality Sweep

Configuration: `configs/calibration_quality_sweep.yaml`.

This regime adds a balanced learner-quality axis. Each learner family has a well-tuned variant and at least one realistic poorly calibrated variant, such as one-step reward-fit FQE, few-step bootstrap FQE, misspecified random features, under- or over-regularized Bellman residual fits, a true iterative ill-conditioned saddle approximation, or a weak ensemble.

Mechanism-only variants wrap fitted predictions with affine, monotone, or saturation distortions. These rows are labeled `main_figure_role=mechanism` and are used to show what each calibrator can and cannot repair; they are not presented as organic baseline learners.

### Regime 7: Mechanism Distortion Sweep

Configuration: `configs/mechanism_distortion_sweep.yaml`.

This focused mechanism suite uses a strong random-feature FQE base learner and then applies controlled prediction distortions. It is intentionally narrower than the calibration-quality sweep so that affine and monotone calibration mechanisms are not obscured by first-stage learner noise.

Why included: it checks that linear calibration repairs affine scale/offset error and that isotonic or isotonic-histogram calibration repairs monotone saturation better than linear calibration.

Success mode tested: linear should win or tie under affine distortion; isotonic/hybrid should beat linear under monotone saturation.

Failure mode tested: if these checks fail over 10 replications, mechanism figures should stay diagnostic/appendix until the simulation is retuned.

Why included: calibration should not be evaluated only on strong well-tuned baselines or only on strawman failures. This suite tests both cases side by side and matches every calibrated variant to its own all-data uncalibrated comparator.

Success mode tested: calibration should help more when the baseline is systematically miscalibrated than when the baseline is already well calibrated.

Failure mode tested: unstable or ill-conditioned learners can remain poor after calibration, and such rows should be retained as diagnostics rather than promoted as main evidence.

## Baseline Learners

Implemented learner families:

- `neural_fqe`: iterative fitted Q evaluation with PyTorch networks, target-network stabilization, weight decay support, and CPU debug settings.
- `random_feature_fqe` / `linear_fqe`: fitted Q evaluation with linear or random Fourier features and ridge stabilization.
- `regularized_bellman`: ridge-regularized Bellman residual baseline.
- `saddle_point_bellman`: stabilized closed-form random-feature approximation to an adversarial/saddle-point Bellman residual estimator.
- `saddle_point_iterative`: iterative random-feature saddle-style Bellman learner using finite-dimensional Q and critic classes with gradient descent/ascent updates. It logs primal norm, critic norm, objective path, gradient norm, condition proxy, NaN/exploding flags, and final prediction range. Stable settings can be appendix evidence; ill-conditioned settings are diagnostic-only unless they pass the well-specified gate.
- `ensemble_fqe`: ensemble FQE baseline using multiple random-feature members.

The calibration-quality suite records `learner_variant`, `learner_quality_regime`, `calibration_difficulty`, and `main_figure_role`. These fields distinguish, for example, `neural_fqe_well_tuned` from `neural_fqe_under_iterated` without pretending they are different estimator families.

Simplified methods:

- The iterative saddle method uses random-feature Q and critic classes rather than a neural min-max critic. This keeps the instability mechanism realistic and inspectable without making the whole study hinge on opaque optimizer failures.
- Neural fine-tuning comparators use the existing estimator API and are kept small in debug mode.

Deferred or excluded methods:

- MuJoCo/Gymnasium benchmarks are deferred to appendix work because the core evidence should come from controlled simulations with reliable oracle values.
- Formal OPE confidence intervals are not part of the current suite. Lightweight bootstrap intervals over independent initial evaluation states are reported as diagnostic coverage and length only.

## Calibration Protocols

- All paper/default calibration is value-space Bellman calibration. First-stage learners estimate \(Q(s,a)\), then the target-policy value predictor \(\hat V(s)=\sum_a\pi(a\mid s)\hat Q(s,a)\) is calibrated by an importance-weighted fitted value-iteration map \(g\). The update minimizes \(\sum_i \rho_i[g(\hat V(S_i))-\{R_i+\gamma g_k(\hat V(S_i'))\}]^2\), where \(\rho_i=\pi(A_i\mid S_i)/\mu(A_i\mid S_i)\). We clip action-ratio weights at 20 and normalize them to mean one on each calibration fit.
- `cross`: five-fold cross-calibration in paper configs. Fold Q-learners are fit out-of-fold; held-out \(\hat V(S)\), \(\hat V(S')\), rewards, and action-ratio weights are pooled; one value-space FVI calibrator is fit on the pooled out-of-fold data; and the final Q learner is refit on all training data before applying \(g\) to final all-data \(\hat V\).
- `split`: honest train/calibration split. The Q learner is trained only on the training split and the value-space FVI calibrator is fit only on the held-out calibration split. Split fractions are configurable and include 0.5, 0.6, 0.7, 0.8, and 0.9 in split sweeps.
- `no_split`: in-sample value-space calibration on the same training data used by the Q learner. It is explicitly labeled optimistic and is not the main theoretically clean protocol.
- `uncalibrated_all_data`: main uncalibrated comparator for each learner.
- `uncalibrated_same_fraction`: appendix comparator for split-calibration data-use accounting.
- Legacy Q-space calibration targets are disabled by default and require `enable_legacy_q_calibration: true`; they are not part of the paper evidence.

## Calibrators

- `linear`: weighted affine regression with intercept and slope, used as the update map inside value-space FVI.
- `histogram`: weighted histogram binning with quantile or equal-width bins and minimum-bin safeguards, used as the update map inside value-space FVI.
- `isotonic`: weighted monotone isotonic regression with repeated-value safeguards inherited from scikit-learn, used as the update map inside value-space FVI.
- `isotonic_histogram`: weighted bin-level averaging followed by weighted isotonic regression, used as a stabilized hybrid update map inside value-space FVI.

## Calibration-Like Competitors

Implemented split-sample comparators:

- `fine_tuning_all_layers` and `fine_tuning_final_layer` where feasible through the learner API.
- `offset_correction`, fitting a residual model with the first-stage prediction as an offset.
- `residual_correction`, fitting feature-based residual corrections.
- `regularized_toward_first_stage`, fitting a second model penalized toward first-stage predictions.

These use the same split fractions as split calibration whenever they are enabled.

## Metrics

Each replication saves:

- value estimate, oracle value, value error, squared error;
- held-out Bellman residual and calibration error;
- true-function MSE, reported as `true_v_mse` / `true_value_function_mse`, comparing calibrated or uncalibrated \(g(\hat V(S))\) to an independently approximated true \(V^\pi(S)\). `true_q_mse` remains a backward-compatible diagnostic for the raw first-stage Q learner but is not the main function metric.
- true-reward Bellman-outcome MSE, reported as `bellman_outcome_mse`, `brier_score`, and `bellman_brier_score`, comparing \(g(\hat V(S))\) to \(r_0(S,A)+\gamma g(\hat V(S'))\), where \(r_0\) is the known conditional mean reward and the continuation value uses the estimator's own calibrated or uncalibrated value map. These are retained as diagnostics, not as final promotion criteria;
- weighted 50-bin Bellman \(L^2\) calibration error, reported as `bellman_calibration_error`, using an independent large diagnostic test set and action-ratio weights. The main column is a cross-fitted debiased binned estimate of \(E_\rho[(Y-\Delta)(\hat\gamma(\Delta)-\Delta)]\) with \(Y=r_0+\gamma g(\hat V(S'))\), \(\Delta=g(\hat V(S))\), and \(\hat\gamma\) fit by weighted quantile bins. The plug-in estimate, raw debiased value, number of bins, diagnostic test size, effective sample size, and max weight are also saved.
- runtime, failure flag, and diagnostic warning;
- run mode, suite name, environment tier, oracle-value method, train/calibration/test provenance;
- seed, sample size, state dimension, coverage, policy shift, reward noise, misspecification;
- baseline learner, calibration protocol, calibrator, calibration target, split fraction, data-use flags;
- actual Bellman/FVI iteration counts, feature dimension, ridge strength, Q/value prediction ranges, saddle diagnostics, and optimizer warnings when available;
- diagnostic 95% bootstrap intervals over independent initial evaluation states, reported as interval coverage and length only;
- coverage-stratified held-out Bellman error, coverage-stratified Bellman calibration error, and density-ratio summaries.

Aggregated metrics include mean estimate, bias, Monte Carlo standard error of bias, variance, MSE, Monte Carlo standard error of MSE, relative MSE versus the corresponding all-data uncalibrated baseline, true value-function MSE, Bellman-outcome MSE, Bellman calibration error, action-ratio weight diagnostics, runtime mean and standard error, failure rate, and eligibility fraction.

Split-calibration diagnostics also report seed-matched relative MSE versus the all-data uncalibrated baseline and versus the same-fraction uncalibrated baseline. These diagnostics include quantiles and win rates so that one-seed split-calibration wins cannot be mistaken for stable paper evidence.

## Calibration Evidence Audit

The paper evidence layer does not treat every scalar-MSE improvement as calibration evidence. Aggregation compares each calibrated row to the matched all-data uncalibrated baseline for the same suite, environment tier, sample size, dimension, coverage, policy shift, reward noise, misspecification, learner variant, calibration target, and run mode.

Each grouped row is assigned `calibration_evidence_status`:

- `strong`: relative value MSE is below one and plug-in Bellman calibration error also improves.
- `mse_only`: value MSE improves but calibration metrics do not. These rows are flagged as possible scalar cancellation and excluded from main calibration-error claims.
- `calibration_only`: Bellman calibration error improves but scalar value MSE does not.
- `neutral`: no clear success and no method failure.
- `failed`: nonfinite, failed, or ineligible rows.

The audit writes `calibration_evidence_audit.csv` and `scalar_cancellation_audit.csv`. Main plots use strong rows plus neutral controls; MSE-only rows remain visible in appendix diagnostics.

## Main Figures

Main-paper candidate figures are generated as PNG and PDF from gated paper-mode results:

1. MSE versus sample size.
2. Relative MSE versus sample size.
3. MSE versus coverage or policy-shift severity.
4. Relative MSE versus coverage or policy-shift severity.
5. Calibrator comparison.
6. Calibration protocol comparison.
7. Bias-variance decomposition.
8. Calibration error versus value error.
9. Mechanism-distortion calibrator comparison if the 10-replication mechanism checks pass.
10. Paired value-MSE and Bellman-calibration-error improvements, excluding MSE-only rows from success claims.
11. Undertraining sweep by actual Bellman iteration count.
12. Bellman-incomplete capacity sweep.

## Appendix Figures

Appendix figures include:

1. Well-specified debug accuracy.
2. Split-fraction comparison.
3. Baseline-family comparison.
4. Runtime comparison.
5. Failure-rate comparison.
6. Coverage-stratified error.
7. Misspecification/distortion sweep.
8. Calibration-quality sweep.
9. Split-calibration stability diagnostics.
10. Model-misspecification learner-family sweep.
11. Saddle failure-rate and condition diagnostics.
12. Optional benchmark environment, deferred until a reliable controlled benchmark bridge is available.

## Expected Qualitative Outcomes

- Under correct specification and good coverage, all reasonable estimators should be close, and calibration should not materially degrade them.
- Under affine distortion, linear calibration should reduce calibration error and often value MSE.
- Under monotone nonlinear distortion, isotonic and isotonic-histogram calibration should be strongest.
- Histogram binning can be useful with enough calibration data but higher variance in small samples.
- Cross-calibration should generally be more data efficient than split calibration because the final base learner is refit on all training data.
- Split calibration exposes the cost of holding out data; same-fraction uncalibrated baselines are essential for interpreting it.
- No-split calibration can look strong in finite samples but is labeled in-sample and may overfit.
- Under severe coverage stress and nonmonotone misspecification, calibration can help little or fail.
- In learner-quality sweeps, calibration should help under-iterated or biased variants more often than well-tuned variants, but ill-conditioned methods may remain diagnostic-only.
- Value-space FVI calibration should be most useful when the raw \(\hat V\) has a calibratable scale or monotone Bellman fixed-point error, but it should not rescue structurally nonmonotone or unsupported regions.

## Known Limitations

- Controlled simulations do not replace real-system validation. They are designed to isolate mechanisms and produce reliable oracle comparisons.
- The saddle-point family includes a stabilized closed-form approximation and an iterative random-feature min-max approximation. A full neural adversarial min-max implementation is still deferred.
- Oracle values are Monte Carlo estimates except in simple debug settings, so final paper runs should use enough oracle rollouts and replications for stable conclusions.
- The suite reports Monte Carlo standard errors over seeds and diagnostic bootstrap intervals over initial states, but it does not yet implement formal confidence intervals for OPE estimators.
- Paper-mode results are gated by the well-specified debug study; failed methods are excluded from main evidence tables and retained for diagnostics.

## Implementation Status

Implemented:

- Controlled well-specified, nonlinear, coverage-stress, misspecification, split-fraction, and baseline-family configs.
- Controlled calibration-quality config with well-tuned and poorly calibrated learner variants.
- Focused mechanism-distortion config and paper-draft inspection script.
- Real iterative saddle-style learner with instability diagnostics.
- Genuine under-iterated neural FQE variants, including a one-step reward-fit setting without first-iteration bootstrapping.
- Prediction distortion wrapper for mechanism-only affine, monotone, and saturation diagnostics.
- Bellman-incomplete and target/reference-shift mechanisms.
- All requested calibration protocols: cross, split, and no-split.
- All requested calibrators: linear, histogram, isotonic, and isotonic-histogram, each applied as an importance-weighted value-space FVI update map.
- Main baseline families: neural FQE, linear/random-feature FQE, regularized Bellman residual, saddle-point Bellman approximation, and ensemble FQE.
- Split-sample comparators: fine-tuning, offset correction, residual correction, and regularization toward the first-stage model.
- Validation gate, raw-result saving, aggregation, tables, and publication-format PNG/PDF plots.
- Seed-matched split-stability diagnostics and a paper-draft readout that recommends main figures only after gated paper-mode summaries exist.
- Calibration evidence auditing that demotes scalar-cancellation wins to `mse_only` unless Bellman calibration error also improves.
- Controlled undertraining, Bellman-incomplete, and model-misspecification sweeps with tables and plots keyed by actual iteration count and learner capacity.

Simplified:

- Saddle-point Bellman is stabilized through random-feature critics.
- Debug mode uses fewer learners, seeds, and samples so the full pipeline is CPU-runnable.

Deferred:

- Optional MuJoCo/Gymnasium appendix benchmarks.
- Formal OPE confidence intervals beyond diagnostic bootstrap intervals.
- Large final-production runs with 30-50 replications, which are configured but intentionally not run by default.

# Bellman Calibration Final Experiment Text

Use only final-stage rows labeled `promote_main` in
`rescue_promotion_audit.csv` for main-text claims. Debug, pilot, confirm, and
probe rows were used only for staged tuning and checking.

## Main Claim

Across the controlled nonlinear offline RL suite, raw FQE value scores can be
rank-informative but miscalibrated in value scale. Post-hoc value-space Bellman
calibration repairs practically important scale errors when the raw score still
contains signal. The final audit promotes 12 main rows, all with strict
cross-calibration or recent-heldout temporal calibration, positive raw
value/oracle correlation, finite diagnostics, no test/oracle/no-split leakage,
relative true-value MSE below 0.98, relative Bellman calibration error below
0.95, and seed win rates at least 0.60.

## Promoted Regimes

**Affine/misspecified linear FQE.** Restricted linear FQE under moderate
behavior-target shift yields informative but slope-biased value predictions.
Linear calibration is enough for the affine story: relative true-value MSE is
about 0.75--0.78 and relative Bellman calibration error is about 0.63--0.66 for
the promoted linear rows. Isotonic is competitive but not necessary here.

**Finite-iteration random-feature FQE.** With high discount and deliberately few
Bellman iterations, random-feature FQE remains strongly rank-informative
(Spearman about 0.85) but mis-scaled. Calibration reduces relative true-value
MSE to about 0.78 and Bellman calibration error to about 0.66--0.74 across
2--4 fitted Bellman iterations.

**Temporal reward shift.** The old and current environments share dynamics,
policies, and state distribution, but the current reward is an affine update of
the old reward. A random-feature FQE model trained on 2000 old-regime behavior
transitions is calibrated using 100 recent current-regime transitions and tested
only in the current regime. Linear recent calibration reduces relative
true-value MSE to 0.63 and Bellman calibration error to 0.72. Isotonic has
similar MSE improvement and a smaller calibration-error gain. Same-small-current
retraining is retained as a stabilized clipped comparator and is not promoted;
recent calibration beats it on both metrics with seed win rates above 0.98.

**Mechanism panel.** Controlled affine and monotone distortions are labeled
`mechanism_only`, not organic evidence. They confirm calibrator behavior:
linear wins/ties under affine distortion, while isotonic is clearly better for
monotone saturation.

## Do Not Claim

Do not claim that calibration fixes arbitrary nonmonotone error, severe support
failure, or unstable value learning. Do not promote `mechanism_only`,
`validation_control`, `reject_mse_only`, `reject_unstable`, or `limitation` rows.
Brier/Bellman-outcome metrics remain in raw outputs as diagnostics, but the main
audit and figures use only true-value MSE and Bellman calibration error.

## Reproducibility

Primary artifacts:

- `rescue_promotion_audit.csv`: source of truth for promoted claims.
- `do_not_claim_manifest.csv`: rows excluded from claims.
- `rescue_readout.md`: human-readable audit summary.
- `tables/focused_neurips_main_summary.csv`: compact table input.
- `FQE_calibration_neurips/figures/rescue_final/focused_neurips_calibration_story.pdf`: main figure.

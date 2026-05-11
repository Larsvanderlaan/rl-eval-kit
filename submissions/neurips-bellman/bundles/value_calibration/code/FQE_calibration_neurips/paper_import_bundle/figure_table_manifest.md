# Figure And Table Manifest

This import bundle is built from the audited rescue-final run:

```text
FQE_calibration_neurips/results/rescue_final/
FQE_calibration_neurips/figures/rescue_final/
```

Paths below are relative to `FQE_calibration_neurips/paper_import_bundle/`
unless otherwise stated.

## Main Text

| Role | Artifact | Source | Intended claim |
|---|---|---|---|
| Main Figure 1 | `figures/calibration_story_compact.pdf` | `../figures/rescue_final/focused_neurips_calibration_story.pdf` | Value-space calibration improves true-\(V\) MSE and Bellman calibration error in affine misspecification, finite-iteration FQE, and temporal reward shift; mechanism rows illustrate calibrator choice. |
| Main Table 1 | Embedded in `experiments_section.tex` | `../results/rescue_final/rescue_promotion_audit.csv` | Representative audited rows only; promoted rows satisfy true-\(V\) MSE improvement, Bellman calibration-error improvement, win rates \(\ge 0.60\), zero failures, and no test/oracle/no-split leakage. |

## Do Not Use For Main Success Claims

| Artifact / row class | Reason |
|---|---|
| Debug, pilot, confirm, or probe rows | Tuning and checking only. |
| `mechanism_only` rows | Diagnostic mechanism evidence, not organic learner evidence. |
| `limitation`, `reject_mse_only`, `reject_unstable` rows | Failed one or more audit gates. |
| No-split calibrated rows | In-sample calibration; diagnostic only. |
| Brier/Bellman-outcome metrics | Saved as diagnostics, but not part of the final promotion gate. |

## Integration Note

`experiments_section.tex` uses:

```tex
\providecommand{\fqeCalibFigDir}{FQE_calibration_neurips/paper_import_bundle/figures}
```

Override this macro before `\input{...}` if the figure directory differs in the
manuscript build.

# NeurIPS Experiments Import Bundle

This folder contains the compact experiment text and figure for the audited
rescue-final Bellman calibration package.

Source of truth:

```text
FQE_calibration_neurips/results/rescue_final/rescue_promotion_audit.csv
FQE_calibration_neurips/results/rescue_final/do_not_claim_manifest.csv
```

Main-text claims should use only rows labeled `promote_main`. The mechanism
panel is useful explanatory evidence but remains labeled `mechanism_only`.

## Files

- `experiments_section.tex`: compact main-text experiments section.
- `experiments_appendix.tex`: appendix details from the broader pipeline.
- `figures/calibration_story_compact.pdf`: refreshed from
  `figures/rescue_final/focused_neurips_calibration_story.pdf`.
- `figure_table_manifest.md`: traceability notes and do-not-use guidance.

## Main Paper Integration

If the paper is compiled from the repository root, use:

```tex
\providecommand{\fqeCalibFigDir}{FQE_calibration_neurips/paper_import_bundle/figures}
\input{FQE_calibration_neurips/paper_import_bundle/experiments_section}
```

For the appendix:

```tex
\input{FQE_calibration_neurips/paper_import_bundle/experiments_appendix}
```

# NeurIPS Bellman Migration Snapshot

Migration branch: `codex/neurips-bellman-reorg`

RLtools base commit at migration start:

```text
4fc1c77fac1139c028077facececf5f1ff5f868f
```

Curated source tree:

```text
/Users/larsvanderlaan/repos/paper_presentations/neurips_bellman
```

Source precedence:

- Canonical papers and paper-facing artifacts come from `papers/` and
  `submission_bundles/*/MANIFEST.json` in the curated source tree.
- Current RLtools post-copy source edits win only for
  `FQE_calibration_neurips/scripts/make_hopper_q_paper_figures.py` and
  `FQE_calibration_neurips/scripts/make_isohist_calibration_story.py`.
- Portable bundle commands and documentation win over local absolute-path
  command variants.

Initial RLtools status before migration:

```text
## main...origin/main
 M FQE/__init__.py
 M FQE/fqe_boosted.py
 M IRL/__init__.py
RM IRL/fit_importance_and_transition_ratios.py -> RL-Evaluation/occupancy-ratio/occupancy_ratio/fit_importance_and_transition_ratios.py
RM IRL/fit_occupancy_ratio.py -> RL-Evaluation/occupancy-ratio/occupancy_ratio/fit_occupancy_ratio.py
 M experiments/figures/baird_fqi_comparison.pdf
 M experiments/figures/baird_fqi_comparison_behavior_norm_0.95.pdf
 M experiments/figures/baird_fqi_two_panel.pdf
?? FQE_calibration_neurips/
?? FQE_neurips/
?? IRL_neurips/
?? RL-Evaluation/FQE/
?? RL-Evaluation/docs/
?? hopper_fqe_benchmark/
?? outputs/
?? pyproject.toml
```

The full status contained additional untracked legacy IRL outputs, notebooks,
and CRM snapshots. Those were moved under `legacy/` or `archive/generated/`
rather than deleted.

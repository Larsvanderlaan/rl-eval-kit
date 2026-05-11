# IRL Migration Snapshot

Migration branch: `codex/neurips-bellman-reorg`

Project name after this reorganization: `RL Evaluation Suite`.

Canonical IRL sources copied into this repo:

- Journal paper: `/Users/larsvanderlaan/repos/paper_presentations/IRL_journal`
  - active main text: `main_jasa.tex`
  - active appendix: `main_jasa_appendix.tex`
  - bibliography: `ref.bib`
- Conference paper: `/Users/larsvanderlaan/repos/paper_presentations/IRL_journal/main_neurips-3.tex`
- Conference reproducibility code and paper-visible artifacts: pre-migration
  `IRL_neurips/IRL_neurips_paper_repro`

Inclusion policy:

- keep paper source TeX, bibliography, final copied PDFs, paper-visible figures,
  compact reproducibility code, and smoke-test scripts active;
- move local virtual environments, caches, notebooks, broad exploratory outputs,
  and the loose full `IRL_neurips` tree into `archive/generated/irl_pre_migration`;
- keep root `IRL_neurips` and `IRL` shims only for compatibility.

The unrelated `DRInference__Lars_` JRSSB calibration/rebuttal materials were
not migrated as IRL paper sources.

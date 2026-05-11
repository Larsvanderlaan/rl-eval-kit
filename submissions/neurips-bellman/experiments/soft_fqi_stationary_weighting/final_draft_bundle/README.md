# Soft-FQI Simulation Final Draft Bundle

Minimal manuscript-ready items:

- `main_text_simulation_section.tex`: concise one-column main-text experiment
  section, designed to fit within one NeurIPS page with the figure.
- `main_text_simulation_figure.pdf`: main NeurIPS-style composite figure.
- `main_text_simulation_figure.png`: preview/collaboration copy of the main figure.
- `main_text_simulation_table.tex`: compact supporting table; use in appendix
  unless main-text space permits.
- `appendix_simulation_details.tex`: environment, regime, metric, and annealing
  details.
- `annealing_tau_comparison_200.csv`: source values for the tau/annealing
  appendix table.
- `appendix_temperature_annealing.tex`: standalone copy of the annealing
  robustness paragraph and table, only needed if you prefer a separate
  appendix fragment.

Suggested placement:

- Main text: paste `main_text_simulation_section.tex`.
- Main figure: `main_text_simulation_section.tex` expects
  `main_text_simulation_figure.pdf` in the same folder.
- Supporting table: include `main_text_simulation_table.tex` in the appendix
  if main-text space is tight.
- Appendix: paste `appendix_simulation_details.tex`. The separate
  `appendix_temperature_annealing.tex` is optional.

The main figure/table use 200 offline datasets at `tau=1e-3`. The appendix
annealing table uses 200 datasets for both `tau=1e-3` and `tau=1e-6`.

Requires `graphicx` and `booktabs`.

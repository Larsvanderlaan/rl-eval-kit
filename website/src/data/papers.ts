export type PaperInfo = {
  title: string;
  package: string;
  status: string;
  venue: string;
  summary: string;
  sourceHref: string;
  pdfHref?: string;
};

const repo = 'https://github.com/Larsvanderlaan/rl-eval-kit/blob/main';

export const papers: PaperInfo[] = [
  {
    title: 'Fitted Occupancy-Ratio Iteration for Offline Reinforcement Learning',
    package: 'occupancy-ratio',
    status: 'Draft source',
    venue: 'Working draft',
    summary:
      'Introduces FORI for discounted occupancy ratios, with separate nuisance, regression, stabilization, and finite-iteration error terms.',
    sourceHref: `${repo}/submissions/occupancy-ratio/paper/fori_iclr2026/main.tex`,
  },
  {
    title: 'Fitted Q Evaluation Without Bellman Completeness via Stationary Weighting',
    package: 'fqe',
    status: 'Draft PDF',
    venue: 'Conference draft',
    summary:
      'Shows how stationary target-to-behavior weighting aligns FQE regression with a contractive norm under misspecification.',
    sourceHref: `${repo}/submissions/neurips-bellman/papers/fqe/main.tex`,
    pdfHref: `${repo}/submissions/neurips-bellman/papers/fqe/main.pdf`,
  },
  {
    title: 'Stationary Reweighting Yields Local Convergence of Soft Fitted Q-Iteration',
    package: 'fqe',
    status: 'Draft PDF',
    venue: 'Conference draft',
    summary:
      'Extends the stationary reweighting idea to soft fitted Q iteration and explains local stability without Bellman completeness.',
    sourceHref: `${repo}/submissions/neurips-bellman/papers/soft_fqi_stationary_weighting/main.tex`,
    pdfHref: `${repo}/submissions/neurips-bellman/papers/soft_fqi_stationary_weighting/main.pdf`,
  },
  {
    title: 'Bellman Calibration for V-Learning in Offline Reinforcement Learning',
    package: 'fqe',
    status: 'Draft PDF',
    venue: 'Conference draft',
    summary:
      'Develops post-hoc Bellman calibration diagnostics for fitted value predictors.',
    sourceHref: `${repo}/submissions/neurips-bellman/papers/calibration/main.tex`,
    pdfHref: `${repo}/submissions/neurips-bellman/papers/calibration/main.pdf`,
  },
  {
    title: 'Modular Inverse Reinforcement Learning via Policy Estimation and Q-Evaluation',
    package: 'genPQR',
    status: 'Draft PDF',
    venue: 'Conference draft',
    summary:
      'Presents GenPQR for normalized reward estimation using policy estimation plus Q evaluation.',
    sourceHref: `${repo}/submissions/irl/papers/conference_genpqr/main_neurips.tex`,
    pdfHref: `${repo}/submissions/irl/papers/conference_genpqr/paper_draft.pdf`,
  },
  {
    title: 'Efficient Inference for Inverse Reinforcement Learning and Dynamic Discrete Choice Models',
    package: 'genPQR',
    status: 'Draft PDF',
    venue: 'Journal draft',
    summary:
      'Connects inverse RL, dynamic discrete choice, and debiased inference for reward-dependent functionals.',
    sourceHref: `${repo}/submissions/irl/papers/journal_debiased_irl/main_jasa.tex`,
    pdfHref: `${repo}/submissions/irl/papers/journal_debiased_irl/main_jasa.pdf`,
  },
];

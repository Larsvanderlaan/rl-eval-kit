export type PackageInfo = {
  name: string;
  slug: string;
  importName: string;
  tagline: string;
  description: string;
  install: string;
  href: string;
  audience: string;
  artifact: string;
};

export const packages: PackageInfo[] = [
  {
    name: 'FQE',
    slug: 'fqe',
    importName: 'fqe',
    tagline: 'Policy-value and Q-function evaluation with logged-data shift diagnostics.',
    description:
      'Fit target-policy Q-functions and policy values from logged transitions, with Bellman and calibration diagnostics.',
    install: 'python -m pip install -e "packages/fqe[neural,benchmark]"',
    href: '/fqe/',
    audience: 'Offline RL researchers, OPE practitioners, model-selection experiments',
    artifact: 'Q models, policy values, calibration diagnostics',
  },
  {
    name: 'Discounted Occupancy Ratios',
    slug: 'occupancy-ratio',
    importName: 'occupancy_ratio',
    tagline: 'Discounted occupancy ratios for target-policy reweighting.',
    description:
      "Estimate state-action weights that reweight reference rows toward the target policy's normalized discounted occupancy, with support and tuning diagnostics.",
    install: 'python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"',
    href: '/occupancy-ratio/',
    audience: 'Researchers using discounted density ratios, OPE weights, or weighted FQE',
    artifact: 'State-action ratios, source diagnostics, benchmark reports',
  },
  {
    name: 'genPQR',
    slug: 'genpqr',
    importName: 'genpqr',
    tagline: 'Normalized reward estimation through policy estimation and Q evaluation.',
    description:
      'Estimate normalized reward representations from logged behavior using policy estimation plus Q evaluation.',
    install: 'python -m pip install -e "packages/genpqr[fqe,torch]"',
    href: '/genpqr/',
    audience: 'IRL researchers, reward-modeling teams, imitation-learning practitioners',
    artifact: 'Normalized rewards, diagnostics, serializable results',
  },
];

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
    tagline: 'Policy-value and Q-function evaluation under distribution shift.',
    description:
      'Estimate fixed-policy Q-functions and policy values from logged transitions. Return fitted models, value estimates, and Bellman diagnostics.',
    install: 'python -m pip install -e "packages/fqe[neural,benchmark]"',
    href: '/fqe/',
    audience: 'Offline RL researchers, OPE practitioners, model-selection experiments',
    artifact: 'Q models, policy-value estimates, calibration diagnostics',
  },
  {
    name: 'Discounted Occupancy Ratios',
    slug: 'occupancy-ratio',
    importName: 'occupancy_ratio',
    tagline: 'Discounted occupancy ratios for target-policy reweighting.',
    description:
      'Estimate state-action weights that reweight reference rows toward a target policy. Return ratios, support diagnostics, and tuning reports.',
    install: 'python -m pip install -e "packages/occupancy-ratio[neural,benchmark]"',
    href: '/occupancy-ratio/',
    audience: 'Researchers who need discounted density ratios, OPE weights, and ratio diagnostics',
    artifact: 'State-action ratios, source diagnostics, benchmark reports',
  },
  {
    name: 'genPQR',
    slug: 'genpqr',
    importName: 'genpqr',
    tagline: 'Normalized reward estimation through policy estimation and Q evaluation.',
    description:
      'Estimate normalized reward representations from logged behavior. Use native BC plus FQE first, with optional adapters loaded only when selected.',
    install: 'python -m pip install -e "packages/genpqr[fqe,torch]"',
    href: '/genpqr/',
    audience: 'IRL researchers, reward-modeling teams, imitation-learning practitioners',
    artifact: 'Normalized reward functions, diagnostics, serializable results',
  },
];

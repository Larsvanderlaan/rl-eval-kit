import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

const githubPagesBase = '/rl-eval-kit';

export default defineConfig({
  site: 'https://larsvanderlaan.github.io',
  base: githubPagesBase,
  integrations: [
    starlight({
      title: 'RLEvalKit',
      description:
        'Offline reinforcement-learning evaluation and normalized reward-estimation tools for researchers and practitioners.',
      customCss: ['./src/styles/global.css'],
      favicon: '/favicon.svg',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Larsvanderlaan/rl-eval-kit',
        },
      ],
      sidebar: [
        {
          label: 'Start',
          items: [
            { slug: 'start' },
            { label: 'Papers', link: '/papers/' },
          ],
        },
        {
          label: 'FQE',
          items: [
            { slug: 'fqe' },
            { slug: 'fqe/quickstart' },
            { slug: 'fqe/methods' },
            { slug: 'fqe/diagnostics' },
            { slug: 'fqe/benchmarks' },
          ],
        },
        {
          label: 'Discounted Ratios',
          items: [
            { slug: 'occupancy-ratio' },
            { slug: 'occupancy-ratio/quickstart' },
            { slug: 'occupancy-ratio/methods' },
            { slug: 'occupancy-ratio/diagnostics' },
            { slug: 'occupancy-ratio/benchmarks' },
          ],
        },
        {
          label: 'genPQR',
          items: [
            { slug: 'genpqr' },
            { slug: 'genpqr/quickstart' },
            { slug: 'genpqr/methods' },
            { slug: 'genpqr/workflows' },
            { slug: 'genpqr/deployment' },
          ],
        },
      ],
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },
    }),
  ],
});

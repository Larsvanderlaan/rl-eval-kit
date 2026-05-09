# Google DualDICE

The package can wrap the official Google Research DualDICE implementation as
an optional comparator.

```python
from occupancy_ratio import (
    GoogleDualDICEConfig,
    fit_google_dualdice_occupancy_ratio,
    preflight_google_dualdice,
)

preflight = preflight_google_dualdice("/tmp/google-research")
if not preflight.available:
    raise RuntimeError(preflight.reason)

model = fit_google_dualdice_occupancy_ratio(
    states=states,
    actions=actions,
    next_states=next_states,
    target_actions=target_actions,
    target_next_actions=target_next_actions,
    initial_states=initial_states,
    initial_actions=initial_actions,
    gamma=0.99,
    config=GoogleDualDICEConfig(
        google_research_path="/tmp/google-research",
        num_updates=1000,
        batch_size=128,
    ),
)
```

## Requirements

Install the optional extra and provide a Google Research checkout:

```bash
python -m pip install -e "packages/occupancy-ratio[google-dualdice]"
git clone https://github.com/google-research/google-research /tmp/google-research
```

TensorFlow and the Google Research package are imported only when preflight or
the fitter is called.

## API Compatibility

The wrapper returns a model with `predict_state_action_ratio(...)`,
`predict_state_ratio(...)`, and `predict_action_ratio(...)` helpers. Google
DualDICE directly learns `zeta(s, a)`, so `predict_action_ratio(...)` returns
ones for compatibility rather than a separately fitted nuisance ratio.

Google DualDICE is never used as tuning truth. It is a deployable comparator
and an explicit opt-in AutoML candidate.

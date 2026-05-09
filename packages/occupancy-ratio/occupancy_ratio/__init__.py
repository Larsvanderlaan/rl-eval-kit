from occupancy_ratio.calibration import (
    calibrate_occupancy_bellman_binning,
    estimate_ope_bellman_control_variate,
    occupancy_bellman_calibration_diagnostics,
    plot_occupancy_bellman_calibration_diagnostics,
    recommend_occupancy_bellman_calibration,
)
from occupancy_ratio.comparison import compare_fori_to_google_dualdice
from occupancy_ratio.diagnostics import postprocess_weights, regularization_path_report, weight_summary
from occupancy_ratio.nuisance_lgbm import (
    fit_importance_ratio_lgbm,
    fit_state_density_ratio_lgbm,
    fit_transition_ratio_lgbm,
)
from occupancy_ratio.configs import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
)
from occupancy_ratio.models import DiscountedOccupancyRatioModel
from occupancy_ratio.boosted import (
    fit_discounted_occupancy_ratio,
    fit_occupancy_ratio_lgbm,
    make_forward_occupancy_dataset,
    tune_discounted_occupancy_ratio_cv,
)
from occupancy_ratio.google_dualdice import (
    GoogleDualDICEConfig,
    GoogleDualDICEOccupancyRatioModel,
    GoogleDualDICEPreflight,
    fit_google_dualdice_occupancy_ratio,
    preflight_google_dualdice,
)
from occupancy_ratio.tuning import (
    CandidateResult,
    FoldResult,
    OccupancySearchSpace,
    OccupancyTuningConfig,
    OccupancyTuningResult,
    tune_occupancy_ratio,
    tune_occupancy_ratio_auto,
)

_NEURAL_EXPORTS = {
    "NeuralDiscountedOccupancyRatioModel",
    "DiscountedOccupancyRatioNeuralModel",
    "NeuralActionRatioConfig",
    "NeuralSourceStateRatioConfig",
    "NeuralOccupancyRegressionConfig",
    "NeuralTransitionRatioConfig",
    "fit_source_state_ratio_neural",
    "fit_discounted_occupancy_ratio_neural",
    "tune_discounted_occupancy_ratio_neural_cv",
}


def __getattr__(name: str):
    if name in _NEURAL_EXPORTS:
        from occupancy_ratio import fit_occupancy_ratio_neural as neural

        if name == "DiscountedOccupancyRatioNeuralModel":
            value = getattr(neural, "NeuralDiscountedOccupancyRatioModel")
        else:
            value = getattr(neural, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "calibrate_occupancy_bellman_binning",
    "occupancy_bellman_calibration_diagnostics",
    "recommend_occupancy_bellman_calibration",
    "plot_occupancy_bellman_calibration_diagnostics",
    "estimate_ope_bellman_control_variate",
    "compare_fori_to_google_dualdice",
    "weight_summary",
    "postprocess_weights",
    "regularization_path_report",
    "ActionRatioConfig",
    "SourceStateRatioConfig",
    "TransitionRatioConfig",
    "OccupancyRegressionConfig",
    "DiscountedOccupancyRatioModel",
    "fit_discounted_occupancy_ratio",
    "tune_discounted_occupancy_ratio_cv",
    "GoogleDualDICEConfig",
    "GoogleDualDICEOccupancyRatioModel",
    "GoogleDualDICEPreflight",
    "fit_google_dualdice_occupancy_ratio",
    "preflight_google_dualdice",
    "fit_occupancy_ratio_lgbm",
    "make_forward_occupancy_dataset",
    "fit_importance_ratio_lgbm",
    "fit_state_density_ratio_lgbm",
    "fit_transition_ratio_lgbm",
    "CandidateResult",
    "FoldResult",
    "OccupancySearchSpace",
    "OccupancyTuningConfig",
    "OccupancyTuningResult",
    "tune_occupancy_ratio",
    "tune_occupancy_ratio_auto",
    "NeuralActionRatioConfig",
    "NeuralSourceStateRatioConfig",
    "NeuralTransitionRatioConfig",
    "NeuralOccupancyRegressionConfig",
    "NeuralDiscountedOccupancyRatioModel",
    "DiscountedOccupancyRatioNeuralModel",
    "fit_source_state_ratio_neural",
    "fit_discounted_occupancy_ratio_neural",
    "tune_discounted_occupancy_ratio_neural_cv",
]

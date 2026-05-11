"""Modular generalized Policy-to-Q-to-Reward tools for IRL."""

from genpqr._version import __version__
from genpqr.action_head_fqe import ActionHeadNeuralFQEConfig, ActionHeadNeuralFQEFunction, ActionHeadNeuralFQEstimator
from genpqr.api import GenPQRConfig, GenPQRResult, fit_genpqr, fit_genpqr_auto, list_presets
from genpqr.crossfit import GenPQRCrossFitResult, fit_genpqr_crossfit
from genpqr.datasets import EpisodeDataset, TransitionDataset
from genpqr.deeppqr import DeepPQRAnchorQEstimator, DeepPQRStratifiedQFunction
from genpqr.deepgenpqr import (
    DeepGenPQRAnchorFallback,
    DeepGenPQRConfig,
    DeepGenPQRQMode,
    DeepGenPQRResult,
    fit_deep_genpqr,
    list_deepgenpqr_presets,
)
from genpqr.diagnostics import GenPQRDiagnostics
from genpqr.exceptions import GenPQRAdapterError, GenPQRConfigurationError, GenPQRError, GenPQRMissingDependencyError
from genpqr.neural_deeppqr import NeuralDeepPQRAnchorQEstimator, NeuralDeepPQRStratifiedQFunction
from genpqr.normalization import ContinuousNormalizationPolicy, DiscreteNormalizationPolicy
from genpqr.policies import BehaviorCloningPolicyEstimator, NativeDiscretePolicy, NativeGaussianPolicy
from genpqr.q_estimators import (
    AutoNeuralFQEstimator,
    ConstantFittedQFunction,
    D3RLPYFQEstimator,
    FQEQEstimator,
    ReusableScopeRLQEstimator,
    ScopeRLDatasetBoundQEstimator,
    ScopeRLQEstimator,
)
from genpqr.registry import (
    available_policy_estimators,
    available_q_estimators,
    register_policy_estimator,
    register_q_estimator,
)
from genpqr.recovery import GenPQRRewardFunction
from genpqr.serialization import (
    load_deep_genpqr_result,
    load_genpqr_result,
    save_deep_genpqr_result,
    save_genpqr_result,
)
from genpqr.types import (
    ActionSpaceSpec,
    EstimatedPolicy,
    FittedQFunction,
    NormalizationPolicy,
    PolicyEstimator,
    QEstimator,
    RewardFunction,
)

__all__ = [
    "ActionSpaceSpec",
    "ActionHeadNeuralFQEConfig",
    "ActionHeadNeuralFQEFunction",
    "ActionHeadNeuralFQEstimator",
    "AutoNeuralFQEstimator",
    "BehaviorCloningPolicyEstimator",
    "ContinuousNormalizationPolicy",
    "ConstantFittedQFunction",
    "D3RLPYFQEstimator",
    "DeepGenPQRConfig",
    "DeepGenPQRAnchorFallback",
    "DeepGenPQRQMode",
    "DeepGenPQRResult",
    "DeepPQRAnchorQEstimator",
    "DeepPQRStratifiedQFunction",
    "DiscreteNormalizationPolicy",
    "EpisodeDataset",
    "EstimatedPolicy",
    "FQEQEstimator",
    "FittedQFunction",
    "GenPQRCrossFitResult",
    "GenPQRAdapterError",
    "GenPQRConfig",
    "GenPQRConfigurationError",
    "GenPQRDiagnostics",
    "GenPQRError",
    "GenPQRMissingDependencyError",
    "GenPQRResult",
    "GenPQRRewardFunction",
    "NativeDiscretePolicy",
    "NativeGaussianPolicy",
    "NeuralDeepPQRAnchorQEstimator",
    "NeuralDeepPQRStratifiedQFunction",
    "NormalizationPolicy",
    "PolicyEstimator",
    "QEstimator",
    "RewardFunction",
    "ReusableScopeRLQEstimator",
    "ScopeRLDatasetBoundQEstimator",
    "ScopeRLQEstimator",
    "TransitionDataset",
    "available_policy_estimators",
    "available_q_estimators",
    "fit_deep_genpqr",
    "fit_genpqr",
    "fit_genpqr_auto",
    "fit_genpqr_crossfit",
    "load_deep_genpqr_result",
    "load_genpqr_result",
    "list_deepgenpqr_presets",
    "list_presets",
    "register_policy_estimator",
    "register_q_estimator",
    "save_genpqr_result",
    "save_deep_genpqr_result",
    "__version__",
]

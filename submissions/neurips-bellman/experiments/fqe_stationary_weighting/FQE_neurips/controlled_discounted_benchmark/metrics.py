from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .configs import EvaluationConfig
from .envs import LinearGaussianEnv
from .features import QuadraticStateActionFunction, QuadraticStateValueFunction
from .policies import GaussianLinearPolicy
from .truth import GaussianMixtureDensity, PolicyTruth


@dataclass
class EstimatorMetrics:
    estimator: str
    target_q_mse: float
    target_q_mse_se: float
    behavior_q_mse: float
    behavior_q_mse_se: float
    behavior_target_action_q_mse: float
    behavior_target_action_q_mse_se: float
    policy_value_estimate: float
    policy_value_true: float
    policy_value_error: float
    policy_value_absolute_error: float
    policy_value_squared_error: float
    initial_v_mse: float
    initial_v_mse_se: float
    behavior_v_mse: float
    behavior_v_mse_se: float
    target_v_mse: float
    target_v_mse_se: float
    behavior_bellman_residual_mse: float
    target_bellman_residual_mse: float
    weight_stats: dict[str, object]


@dataclass
class EvaluationSamples:
    target_sa: np.ndarray
    behavior_sa: np.ndarray
    behavior_target_action_sa: np.ndarray
    initial_states: np.ndarray
    target_states: np.ndarray
    behavior_states: np.ndarray


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if values_arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values_arr))
    if values_arr.size <= 1:
        return mean, 0.0
    return mean, float(np.std(values_arr, ddof=1) / np.sqrt(values_arr.size))


def _bellman_residual_mse(
    q_function: QuadraticStateActionFunction,
    *,
    env: LinearGaussianEnv,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    states: np.ndarray,
    actions: np.ndarray,
) -> float:
    value_fn = q_function.to_state_value(target_policy)
    next_mean = env.state_transition_mean(states, actions)
    target = env.expected_reward(states, actions) + gamma * value_fn.expectation_under_transition(
        next_mean,
        env.process_noise_cov,
    )
    residual = q_function.evaluate(states, actions) - target
    return float(np.mean(residual**2))


def draw_evaluation_samples(
    *,
    env: LinearGaussianEnv,
    target_policy: GaussianLinearPolicy,
    evaluation_config: EvaluationConfig,
    target_joint_mixture: GaussianMixtureDensity,
    behavior_joint_mixture: GaussianMixtureDensity,
    target_state_mixture: GaussianMixtureDensity,
    behavior_state_mixture: GaussianMixtureDensity,
    rng: np.random.Generator,
) -> EvaluationSamples:
    target_sa = target_joint_mixture.sample(evaluation_config.q_eval_draws, rng)
    behavior_sa = behavior_joint_mixture.sample(evaluation_config.q_eval_draws, rng)
    behavior_target_action_states = behavior_state_mixture.sample(evaluation_config.q_eval_draws, rng)
    behavior_target_action_sa = np.concatenate(
        [
            behavior_target_action_states,
            target_policy.sample_actions(behavior_target_action_states, rng),
        ],
        axis=1,
    )
    return EvaluationSamples(
        target_sa=target_sa,
        behavior_sa=behavior_sa,
        behavior_target_action_sa=behavior_target_action_sa,
        initial_states=rng.multivariate_normal(
            mean=env.config.initial_mean,
            cov=env.config.initial_cov,
            size=evaluation_config.initial_eval_draws,
        ),
        target_states=target_state_mixture.sample(evaluation_config.state_eval_draws, rng),
        behavior_states=behavior_state_mixture.sample(evaluation_config.state_eval_draws, rng),
    )


def evaluate_estimator(
    *,
    estimator: str,
    q_function: QuadraticStateActionFunction,
    truth: PolicyTruth,
    env: LinearGaussianEnv,
    target_policy: GaussianLinearPolicy,
    gamma: float,
    evaluation_config: EvaluationConfig,
    target_joint_mixture: GaussianMixtureDensity,
    behavior_joint_mixture: GaussianMixtureDensity,
    target_state_mixture: GaussianMixtureDensity,
    behavior_state_mixture: GaussianMixtureDensity,
    rng: np.random.Generator,
    weight_stats: dict[str, object],
    evaluation_samples: EvaluationSamples | None = None,
) -> EstimatorMetrics:
    if evaluation_samples is None:
        evaluation_samples = draw_evaluation_samples(
            env=env,
            target_policy=target_policy,
            evaluation_config=evaluation_config,
            target_joint_mixture=target_joint_mixture,
            behavior_joint_mixture=behavior_joint_mixture,
            target_state_mixture=target_state_mixture,
            behavior_state_mixture=behavior_state_mixture,
            rng=rng,
        )
    target_sa = evaluation_samples.target_sa
    behavior_sa = evaluation_samples.behavior_sa
    behavior_target_action_sa = evaluation_samples.behavior_target_action_sa
    q_true_target = truth.q_function.evaluate(target_sa[:, :2], target_sa[:, 2:])
    q_hat_target = q_function.evaluate(target_sa[:, :2], target_sa[:, 2:])
    q_true_behavior = truth.q_function.evaluate(behavior_sa[:, :2], behavior_sa[:, 2:])
    q_hat_behavior = q_function.evaluate(behavior_sa[:, :2], behavior_sa[:, 2:])
    q_true_behavior_target_action = truth.q_function.evaluate(
        behavior_target_action_sa[:, :2],
        behavior_target_action_sa[:, 2:],
    )
    q_hat_behavior_target_action = q_function.evaluate(
        behavior_target_action_sa[:, :2],
        behavior_target_action_sa[:, 2:],
    )
    target_q_sqerr = (q_hat_target - q_true_target) ** 2
    behavior_q_sqerr = (q_hat_behavior - q_true_behavior) ** 2
    behavior_target_action_q_sqerr = (q_hat_behavior_target_action - q_true_behavior_target_action) ** 2

    value_true = truth.v_function
    value_hat = q_function.to_state_value(target_policy)
    initial_states = evaluation_samples.initial_states
    target_states = evaluation_samples.target_states
    behavior_states = evaluation_samples.behavior_states

    value_estimate = value_hat.expectation_under_gaussian(env.config.initial_mean, env.config.initial_cov)
    value_error = value_estimate - truth.policy_value
    initial_v_sqerr = (value_hat.evaluate(initial_states) - value_true.evaluate(initial_states)) ** 2
    behavior_v_sqerr = (value_hat.evaluate(behavior_states) - value_true.evaluate(behavior_states)) ** 2
    target_v_sqerr = (value_hat.evaluate(target_states) - value_true.evaluate(target_states)) ** 2
    target_q_mse, target_q_mse_se = _mean_and_se(target_q_sqerr)
    behavior_q_mse, behavior_q_mse_se = _mean_and_se(behavior_q_sqerr)
    behavior_target_action_q_mse, behavior_target_action_q_mse_se = _mean_and_se(
        behavior_target_action_q_sqerr
    )
    initial_v_mse, initial_v_mse_se = _mean_and_se(initial_v_sqerr)
    behavior_v_mse, behavior_v_mse_se = _mean_and_se(behavior_v_sqerr)
    target_v_mse, target_v_mse_se = _mean_and_se(target_v_sqerr)

    return EstimatorMetrics(
        estimator=estimator,
        target_q_mse=target_q_mse,
        target_q_mse_se=target_q_mse_se,
        behavior_q_mse=behavior_q_mse,
        behavior_q_mse_se=behavior_q_mse_se,
        behavior_target_action_q_mse=behavior_target_action_q_mse,
        behavior_target_action_q_mse_se=behavior_target_action_q_mse_se,
        policy_value_estimate=float(value_estimate),
        policy_value_true=float(truth.policy_value),
        policy_value_error=float(value_error),
        policy_value_absolute_error=float(abs(value_error)),
        policy_value_squared_error=float(value_error**2),
        initial_v_mse=initial_v_mse,
        initial_v_mse_se=initial_v_mse_se,
        behavior_v_mse=behavior_v_mse,
        behavior_v_mse_se=behavior_v_mse_se,
        target_v_mse=target_v_mse,
        target_v_mse_se=target_v_mse_se,
        behavior_bellman_residual_mse=_bellman_residual_mse(
            q_function,
            env=env,
            target_policy=target_policy,
            gamma=gamma,
            states=behavior_sa[:, :2],
            actions=behavior_sa[:, 2:],
        ),
        target_bellman_residual_mse=_bellman_residual_mse(
            q_function,
            env=env,
            target_policy=target_policy,
            gamma=gamma,
            states=target_sa[:, :2],
            actions=target_sa[:, 2:],
        ),
        weight_stats=dict(weight_stats),
    )

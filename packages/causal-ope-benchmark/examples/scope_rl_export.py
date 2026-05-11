"""Export a benchmark dataset to the SCOPE-RL logged-dataset shape."""

from causal_ope_benchmark import make_benchmark_problem, to_scope_rl_logged_dataset, validate_scope_rl_logged_dataset


def main() -> None:
    problem = make_benchmark_problem("streamretain", sample_size=50, gamma=0.95, seed=0, target_policy="moderate")
    payload = to_scope_rl_logged_dataset(problem.dataset, behavior_policy_name="logged", dataset_id=1)
    validate_scope_rl_logged_dataset(payload)
    print(payload["n_trajectories"], payload["step_per_trajectory"], payload["state_dim"])


if __name__ == "__main__":
    main()

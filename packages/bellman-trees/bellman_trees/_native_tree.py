from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._base import SerializableEstimatorMixin
from ._data import BellmanTransitionData
from ._features import average_next_features, leaf_assignments_to_csr, remap_labels
from ._solver import solve_projected_bellman
from ._weights import effective_sample_size, stabilize_weights


Array = np.ndarray


@dataclass
class _Node:
    node_id: int
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(frozen=True)
class _CandidateSplit:
    node_id: int
    feature_index: int
    threshold: float
    left_id: int
    right_id: int


class BellmanAggregationTree(SerializableEstimatorMixin):
    """Bellman-aware regression tree with a final aggregate Bellman solve."""

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        max_depth: int = 4,
        max_leaves: int = 16,
        max_bins: int = 32,
        min_samples_leaf: int = 20,
        min_weighted_leaf_mass: float = 1e-8,
        min_leaf_ess: float = 5.0,
        complexity_penalty: float = 0.0,
        min_improvement: float = 1e-10,
        ridge: float = 1e-8,
        solver_method: str = "direct",
        solver_max_iter: int = 500,
        solver_tol: float = 1e-8,
        honest: bool = True,
        estimation_fraction: float = 0.35,
        growth_score_fraction: float = 0.35,
        weight_clip_quantile: float | None = 0.995,
        max_weight: float | None = None,
        weight_uniform_mix: float = 0.0,
        target_ess_fraction: float | None = None,
        random_state: int | None = None,
        feature_indices: Array | None = None,
    ) -> None:
        self.gamma = gamma
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_weighted_leaf_mass = min_weighted_leaf_mass
        self.min_leaf_ess = min_leaf_ess
        self.complexity_penalty = complexity_penalty
        self.min_improvement = min_improvement
        self.ridge = ridge
        self.solver_method = solver_method
        self.solver_max_iter = solver_max_iter
        self.solver_tol = solver_tol
        self.honest = honest
        self.estimation_fraction = estimation_fraction
        self.growth_score_fraction = growth_score_fraction
        self.weight_clip_quantile = weight_clip_quantile
        self.max_weight = max_weight
        self.weight_uniform_mix = weight_uniform_mix
        self.target_ess_fraction = target_ess_fraction
        self.random_state = random_state
        self.feature_indices = feature_indices

    def fit(
        self,
        X: Array,
        reward: Array,
        X_next: Array,
        sample_weight: Array | None = None,
    ) -> "BellmanAggregationTree":
        data = BellmanTransitionData(X=X, reward=reward, X_next=X_next, sample_weight=sample_weight)
        weights = stabilize_weights(
            data.sample_weight,
            data.n_samples,
            max_weight=self.max_weight,
            clip_quantile=self.weight_clip_quantile,
            uniform_mix=self.weight_uniform_mix,
            target_ess_fraction=self.target_ess_fraction,
        )
        self.weight_diagnostics_ = weights.diagnostics
        w = weights.values
        rng = np.random.default_rng(self.random_state)
        grow_fit_idx, grow_score_idx, estimation_idx = self._split_roles(data.n_samples, rng)
        self.root_ = _Node(node_id=0, depth=0)
        self._next_node_id = 1
        self.split_history_: list[dict[str, float | int | bool]] = []
        feature_indices = self._candidate_features(data.n_features, rng)
        current_score = self._score_current_tree(data, w, grow_fit_idx, grow_score_idx)

        while len(self._leaf_nodes()) < int(self.max_leaves):
            best: tuple[float, _CandidateSplit] | None = None
            raw_fit = self._apply_raw(data.X[grow_fit_idx])
            for leaf in self._leaf_nodes():
                if leaf.depth >= int(self.max_depth):
                    continue
                local_idx = grow_fit_idx[raw_fit == leaf.node_id]
                if local_idx.size < 2 * int(self.min_samples_leaf):
                    continue
                for feature in feature_indices:
                    for threshold in self._thresholds(data.X[local_idx, feature]):
                        candidate = _CandidateSplit(
                            node_id=leaf.node_id,
                            feature_index=int(feature),
                            threshold=float(threshold),
                            left_id=self._next_node_id,
                            right_id=self._next_node_id + 1,
                        )
                        if not self._candidate_is_admissible(data, w, local_idx, candidate):
                            continue
                        score = self._score_candidate(data, w, grow_fit_idx, grow_score_idx, candidate)
                        if best is None or score < best[0]:
                            best = (score, candidate)
            if best is None:
                break
            best_score, best_candidate = best
            accepted = best_score + float(self.min_improvement) < current_score
            self.split_history_.append(
                {
                    "leaf_node_id": int(best_candidate.node_id),
                    "feature_index": int(best_candidate.feature_index),
                    "threshold": float(best_candidate.threshold),
                    "score": float(best_score),
                    "previous_score": float(current_score),
                    "accepted": bool(accepted),
                }
            )
            if not accepted:
                break
            self._commit_split(best_candidate)
            current_score = best_score

        self.leaf_node_ids_ = self._leaf_node_ids()
        if estimation_idx.size < max(2, int(self.min_samples_leaf)):
            estimation_idx = np.arange(data.n_samples, dtype=np.int64)
        phi = self.transform(data.X[estimation_idx])
        phi_next = self.transform_next(data.X_next[estimation_idx])
        solve = solve_projected_bellman(
            phi,
            phi_next,
            data.reward[estimation_idx],
            w[estimation_idx],
            gamma=float(self.gamma),
            ridge=float(self.ridge),
            method=self.solver_method,
            max_iter=int(self.solver_max_iter),
            tol=float(self.solver_tol),
        )
        self.theta_ = solve.theta
        self.solver_info_ = solve.diagnostics
        self.feature_info_ = {
            "n_leaves": int(phi.shape[1]),
            "n_input_features": int(data.n_features),
            "feature_indices": None if self.feature_indices is None else np.asarray(self.feature_indices).tolist(),
            "honest": bool(self.honest),
            "n_growth_fit": int(grow_fit_idx.size),
            "n_growth_score": int(grow_score_idx.size),
            "n_estimation": int(estimation_idx.size),
        }
        self.diagnostics_ = {
            **self.weight_diagnostics_,
            "initial_score": float(current_score) if not self.split_history_ else float(self.split_history_[0]["previous_score"]),
            "final_growth_score": float(current_score),
            "n_splits": int(len([row for row in self.split_history_ if row["accepted"]])),
            "n_leaves": int(phi.shape[1]),
        }
        return self

    def predict(self, X_eval: Array) -> Array:
        self._check_is_fitted()
        phi = self.transform(X_eval)
        return np.asarray(phi @ self.theta_, dtype=np.float64).reshape(-1)

    def transform(self, X: Array) -> sparse.csr_matrix:
        self._check_tree()
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        raw = self._apply_raw(x)
        assignments, _ = remap_labels(raw, self.leaf_node_ids_)
        return leaf_assignments_to_csr(assignments, n_features=len(self.leaf_node_ids_))

    def transform_next(self, X_next: Array) -> sparse.csr_matrix:
        return average_next_features(self.transform, X_next)

    def _split_roles(self, n: int, rng: np.random.Generator) -> tuple[Array, Array, Array]:
        perm = rng.permutation(n)
        if self.honest and self.estimation_fraction > 0.0 and n >= 10:
            n_est = max(1, int(round(float(self.estimation_fraction) * n)))
            estimation = np.sort(perm[:n_est])
            growth = perm[n_est:]
        else:
            estimation = np.arange(n, dtype=np.int64)
            growth = perm
        if growth.size < 2:
            growth = perm
        n_score = int(round(float(self.growth_score_fraction) * growth.size))
        n_score = min(max(1, n_score), max(growth.size - 1, 1))
        score = np.sort(growth[:n_score])
        fit = np.sort(growth[n_score:]) if growth.size > n_score else np.sort(growth)
        if fit.size == 0:
            fit = score
        return fit.astype(np.int64), score.astype(np.int64), estimation.astype(np.int64)

    def _candidate_features(self, p: int, rng: np.random.Generator) -> Array:
        if self.feature_indices is None:
            return np.arange(p, dtype=np.int64)
        idx = np.asarray(self.feature_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0 or np.min(idx) < 0 or np.max(idx) >= p:
            raise ValueError("feature_indices must be nonempty valid column indices.")
        return idx

    def _thresholds(self, values: Array) -> Array:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        unique = np.unique(finite)
        if unique.size <= 1:
            return np.empty(0, dtype=np.float64)
        if unique.size <= int(self.max_bins) + 1:
            return 0.5 * (unique[:-1] + unique[1:])
        probs = np.linspace(0.0, 1.0, int(self.max_bins) + 2)[1:-1]
        thresholds = np.unique(np.quantile(finite, probs))
        return thresholds[(thresholds > unique[0]) & (thresholds < unique[-1])]

    def _candidate_is_admissible(
        self,
        data: BellmanTransitionData,
        weights: Array,
        leaf_indices: Array,
        candidate: _CandidateSplit,
    ) -> bool:
        vals = data.X[leaf_indices, candidate.feature_index]
        left = leaf_indices[vals <= candidate.threshold]
        right = leaf_indices[vals > candidate.threshold]
        return self._child_ok(weights[left]) and self._child_ok(weights[right])

    def _child_ok(self, weights: Array) -> bool:
        if weights.size < int(self.min_samples_leaf):
            return False
        if float(np.sum(weights)) < float(self.min_weighted_leaf_mass):
            return False
        if self.min_leaf_ess is not None and effective_sample_size(weights) < float(self.min_leaf_ess):
            return False
        return True

    def _score_current_tree(self, data: BellmanTransitionData, weights: Array, fit_idx: Array, score_idx: Array) -> float:
        return self._score_candidate(data, weights, fit_idx, score_idx, candidate=None)

    def _score_candidate(
        self,
        data: BellmanTransitionData,
        weights: Array,
        fit_idx: Array,
        score_idx: Array,
        candidate: _CandidateSplit | None,
    ) -> float:
        transform = lambda x: self._transform_with_candidate(x, candidate)
        phi_fit = transform(data.X[fit_idx])
        phi_next_fit = average_next_features(transform, data.X_next[fit_idx])
        solve = solve_projected_bellman(
            phi_fit,
            phi_next_fit,
            data.reward[fit_idx],
            weights[fit_idx],
            gamma=float(self.gamma),
            ridge=float(self.ridge),
        )
        phi_score = transform(data.X[score_idx])
        phi_next_score = average_next_features(transform, data.X_next[score_idx])
        residual = np.asarray(phi_score @ solve.theta).reshape(-1) - (
            data.reward[score_idx] + float(self.gamma) * np.asarray(phi_next_score @ solve.theta).reshape(-1)
        )
        w = weights[score_idx]
        return float(np.sum(w * residual**2) / max(np.sum(w), 1e-12) + float(self.complexity_penalty) * phi_fit.shape[1])

    def _transform_with_candidate(self, X: Array, candidate: _CandidateSplit | None) -> sparse.csr_matrix:
        raw = self._apply_raw(np.asarray(X, dtype=np.float64))
        leaf_ids = self._leaf_node_ids()
        if candidate is not None:
            mask = raw == candidate.node_id
            if np.any(mask):
                raw = raw.copy()
                go_left = np.asarray(X, dtype=np.float64)[mask, candidate.feature_index] <= candidate.threshold
                masked_pos = np.nonzero(mask)[0]
                raw[masked_pos[go_left]] = candidate.left_id
                raw[masked_pos[~go_left]] = candidate.right_id
            leaf_ids = [leaf for leaf in leaf_ids if leaf != candidate.node_id] + [candidate.left_id, candidate.right_id]
            leaf_ids = sorted(leaf_ids)
        assignments, _ = remap_labels(raw, leaf_ids)
        return leaf_assignments_to_csr(assignments, n_features=len(leaf_ids))

    def _commit_split(self, candidate: _CandidateSplit) -> None:
        node = self._find_node(candidate.node_id)
        node.feature_index = int(candidate.feature_index)
        node.threshold = float(candidate.threshold)
        node.left = _Node(node_id=candidate.left_id, depth=node.depth + 1)
        node.right = _Node(node_id=candidate.right_id, depth=node.depth + 1)
        self._next_node_id += 2

    def _apply_raw(self, X: Array) -> Array:
        self._check_tree()
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        out = np.empty(x.shape[0], dtype=np.int64)

        def visit(node: _Node, indices: Array) -> None:
            if indices.size == 0:
                return
            if node.is_leaf:
                out[indices] = node.node_id
                return
            assert node.feature_index is not None and node.threshold is not None
            vals = x[indices, node.feature_index]
            left_mask = vals <= node.threshold
            visit(node.left, indices[left_mask])  # type: ignore[arg-type]
            visit(node.right, indices[~left_mask])  # type: ignore[arg-type]

        visit(self.root_, np.arange(x.shape[0], dtype=np.int64))
        return out

    def _leaf_nodes(self) -> list[_Node]:
        self._check_tree()
        leaves: list[_Node] = []

        def walk(node: _Node) -> None:
            if node.is_leaf:
                leaves.append(node)
            else:
                walk(node.left)  # type: ignore[arg-type]
                walk(node.right)  # type: ignore[arg-type]

        walk(self.root_)
        return leaves

    def _leaf_node_ids(self) -> list[int]:
        return sorted(node.node_id for node in self._leaf_nodes())

    def _find_node(self, node_id: int) -> _Node:
        self._check_tree()

        def walk(node: _Node) -> _Node | None:
            if node.node_id == node_id:
                return node
            if node.is_leaf:
                return None
            return walk(node.left) or walk(node.right)  # type: ignore[arg-type]

        found = walk(self.root_)
        if found is None:
            raise KeyError(f"Unknown node id {node_id}.")
        return found

    def _check_tree(self) -> None:
        if not hasattr(self, "root_"):
            raise RuntimeError("Estimator is not fitted.")

    def _check_is_fitted(self) -> None:
        self._check_tree()
        if not hasattr(self, "theta_"):
            raise RuntimeError("Estimator is not fitted.")

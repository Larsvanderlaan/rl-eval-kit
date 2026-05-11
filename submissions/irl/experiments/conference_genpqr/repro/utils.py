"""Shared utilities for the self-contained IRL experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


EPS = 1e-8


def set_random_seed(seed: int) -> np.random.Generator:
    """Return a reproducible NumPy random generator."""
    return np.random.default_rng(seed)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.clip(np.sum(exp_shifted, axis=axis, keepdims=True), EPS, None)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """One-hot encode integer actions."""
    actions = np.asarray(actions, dtype=int).reshape(-1)
    encoded = np.zeros((actions.shape[0], n_actions), dtype=float)
    encoded[np.arange(actions.shape[0]), actions] = 1.0
    return encoded


def state_action_features(states: np.ndarray, actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Concatenate state features with a one-hot action indicator."""
    states = np.asarray(states, dtype=float)
    return np.concatenate([states, one_hot(actions, n_actions)], axis=1)


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit standardization parameters."""
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply standardization parameters."""
    return (np.asarray(x, dtype=float) - mean) / std


@dataclass
class MLP:
    """A tiny fully-connected network with ReLU hidden layers."""

    weights: List[np.ndarray]
    biases: List[np.ndarray]
    task: str
    x_mean: np.ndarray
    x_std: np.ndarray

    @classmethod
    def initialize(
        cls,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        rng: np.random.Generator,
        task: str,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> "MLP":
        dims = [input_dim, *hidden_sizes, output_dim]
        weights = []
        biases = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2.0 / max(d_in, 1))
            weights.append(rng.normal(scale=scale, size=(d_in, d_out)))
            biases.append(np.zeros((1, d_out), dtype=float))
        return cls(weights=weights, biases=biases, task=task, x_mean=x_mean, x_std=x_std)

    def _forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        h = standardize_apply(x, self.x_mean, self.x_std)
        activations = [h]
        pre_activations = []
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            pre_activations.append(z)
            if idx + 1 == len(self.weights):
                activations.append(z)
            else:
                activations.append(np.maximum(z, 0.0))
        return activations, pre_activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return raw output."""
        activations, _ = self._forward(x)
        return activations[-1]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for classification tasks."""
        logits = self.predict(x)
        if logits.ndim == 1 or logits.shape[1] == 1:
            probs1 = sigmoid(logits.reshape(-1, 1))
            return np.concatenate([1.0 - probs1, probs1], axis=1)
        return softmax(logits, axis=1)

    def fit_regression(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 5e-3,
        n_epochs: int = 400,
        batch_size: int = 256,
        l2: float = 1e-4,
        verbose: bool = False,
        rng: np.random.Generator | None = None,
    ) -> "MLP":
        """Fit the MLP by mini-batch gradient descent for squared loss."""
        rng = rng or np.random.default_rng(0)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        n = x.shape[0]
        for epoch in range(n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x[idx], y[idx]
                activations, pre_activations = self._forward(xb)
                pred = activations[-1]
                delta = 2.0 * (pred - yb) / max(xb.shape[0], 1)
                for layer in reversed(range(len(self.weights))):
                    a_prev = activations[layer]
                    grad_w = a_prev.T @ delta + l2 * self.weights[layer]
                    grad_b = np.sum(delta, axis=0, keepdims=True)
                    if layer > 0:
                        delta = (delta @ self.weights[layer].T) * (pre_activations[layer - 1] > 0.0)
                    self.weights[layer] -= learning_rate * grad_w
                    self.biases[layer] -= learning_rate * grad_b
            if verbose and (epoch + 1) % 100 == 0:
                loss = np.mean((self.predict(x) - y) ** 2)
                print(f"[MLP regression] epoch={epoch + 1} mse={loss:.6f}")
        return self

    def fit_multiclass(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 5e-3,
        n_epochs: int = 400,
        batch_size: int = 256,
        l2: float = 1e-4,
        verbose: bool = False,
        rng: np.random.Generator | None = None,
    ) -> "MLP":
        """Fit a multiclass classifier by mini-batch cross-entropy."""
        rng = rng or np.random.default_rng(0)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        n = x.shape[0]
        n_classes = self.biases[-1].shape[1]
        for epoch in range(n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x[idx], y[idx]
                activations, pre_activations = self._forward(xb)
                probs = softmax(activations[-1], axis=1)
                probs[np.arange(yb.shape[0]), yb] -= 1.0
                delta = probs / max(yb.shape[0], 1)
                for layer in reversed(range(len(self.weights))):
                    a_prev = activations[layer]
                    grad_w = a_prev.T @ delta + l2 * self.weights[layer]
                    grad_b = np.sum(delta, axis=0, keepdims=True)
                    if layer > 0:
                        delta = (delta @ self.weights[layer].T) * (pre_activations[layer - 1] > 0.0)
                    self.weights[layer] -= learning_rate * grad_w
                    self.biases[layer] -= learning_rate * grad_b
            if verbose and (epoch + 1) % 100 == 0:
                probs = self.predict_proba(x)
                loss = -np.mean(np.log(np.clip(probs[np.arange(n), y], EPS, None)))
                print(f"[MLP classification] epoch={epoch + 1} nll={loss:.6f}")
        return self


@dataclass
class TreeNode:
    """A shallow regression tree node."""

    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    value: float | None = None

    def predict_row(self, row: np.ndarray) -> float:
        """Predict for a single row."""
        if self.value is not None:
            return float(self.value)
        if row[self.feature_index] <= self.threshold:
            return self.left.predict_row(row)
        return self.right.predict_row(row)


def _best_split(x: np.ndarray, residual: np.ndarray, min_leaf: int) -> Tuple[int | None, float | None, float]:
    """Find the best axis-aligned split for squared-error reduction."""
    n_samples, n_features = x.shape
    total_sse = np.sum((residual - residual.mean()) ** 2)
    best_gain = 0.0
    best_feature = None
    best_threshold = None
    for feature in range(n_features):
        values = x[:, feature]
        percentiles = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
        for threshold in percentiles:
            left = values <= threshold
            right = ~left
            if left.sum() < min_leaf or right.sum() < min_leaf:
                continue
            left_resid = residual[left]
            right_resid = residual[right]
            left_sse = np.sum((left_resid - left_resid.mean()) ** 2)
            right_sse = np.sum((right_resid - right_resid.mean()) ** 2)
            gain = total_sse - left_sse - right_sse
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = float(threshold)
    return best_feature, best_threshold, best_gain


def fit_regression_tree(
    x: np.ndarray,
    y: np.ndarray,
    max_depth: int = 2,
    min_leaf: int = 20,
) -> TreeNode:
    """Fit a small regression tree by greedy CART-style splitting."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    def build(x_sub: np.ndarray, y_sub: np.ndarray, depth: int) -> TreeNode:
        if depth >= max_depth or x_sub.shape[0] < 2 * min_leaf:
            return TreeNode(value=float(np.mean(y_sub)))
        feature, threshold, gain = _best_split(x_sub, y_sub, min_leaf=min_leaf)
        if feature is None or gain <= 0.0:
            return TreeNode(value=float(np.mean(y_sub)))
        left = x_sub[:, feature] <= threshold
        return TreeNode(
            feature_index=feature,
            threshold=threshold,
            left=build(x_sub[left], y_sub[left], depth + 1),
            right=build(x_sub[~left], y_sub[~left], depth + 1),
        )

    return build(x, y, 0)


@dataclass
class GradientBoostedRegressor:
    """A lightweight gradient-boosted regressor using shallow trees."""

    base_value: float
    learning_rate: float
    trees: List[TreeNode]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict regression targets."""
        x = np.asarray(x, dtype=float)
        pred = np.full(x.shape[0], self.base_value, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate * np.array([tree.predict_row(row) for row in x])
        return pred


def fit_gradient_boosted_regressor(
    x: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 80,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_leaf: int = 20,
    verbose: bool = False,
) -> GradientBoostedRegressor:
    """Fit a stagewise squared-error boosting model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    base_value = float(np.mean(y))
    pred = np.full(y.shape[0], base_value, dtype=float)
    trees: List[TreeNode] = []
    for idx in range(n_estimators):
        residual = y - pred
        tree = fit_regression_tree(x, residual, max_depth=max_depth, min_leaf=min_leaf)
        update = np.array([tree.predict_row(row) for row in x])
        pred += learning_rate * update
        trees.append(tree)
        if verbose and (idx + 1) % 20 == 0:
            mse = np.mean((y - pred) ** 2)
            print(f"[GBR] stage={idx + 1} mse={mse:.6f}")
    return GradientBoostedRegressor(base_value=base_value, learning_rate=learning_rate, trees=trees)

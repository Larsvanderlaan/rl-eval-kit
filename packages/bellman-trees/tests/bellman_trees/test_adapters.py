from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from bellman_trees import BellmanLeafEnsembleRegressor


def _toy_data(n: int = 220, seed: int = 101):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X_next = X + 0.05 * rng.normal(size=(n, 4))
    reward = np.sin(X[:, 0]) + 0.25 * X[:, 1]
    weights = np.linspace(0.5, 1.5, n)
    return X, reward, X_next, weights


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(max_depth=3, random_state=1),
        RandomForestRegressor(n_estimators=5, max_depth=3, random_state=2),
        GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=3),
        HistGradientBoostingRegressor(max_iter=5, max_leaf_nodes=8, random_state=4),
    ],
)
def test_sklearn_tree_ensemble_adapters_fit_and_predict(model) -> None:
    X, reward, X_next, weights = _toy_data()
    est = BellmanLeafEnsembleRegressor(
        model,
        gamma=0.8,
        ridge=1e-6,
        feature_target="reward",
    ).fit(X, reward, X_next, weights)
    pred = est.predict(X[:20])
    assert pred.shape == (20,)
    assert np.all(np.isfinite(pred))
    assert est.feature_info_["n_features"] > 0


def test_xgboost_adapter_optional() -> None:
    xgb = pytest.importorskip("xgboost")
    X, reward, X_next, weights = _toy_data(n=120)
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, random_state=5, verbosity=0)
    est = BellmanLeafEnsembleRegressor(model, adapter="xgboost", gamma=0.8).fit(X, reward, X_next, weights)
    assert np.all(np.isfinite(est.predict(X[:10])))

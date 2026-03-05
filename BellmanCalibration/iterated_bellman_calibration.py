import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
import numpy as np

import numpy as np
import numpy as np

def iterated_bellman_calibration(
    V,
    V_prime,
    Y,
    is_terminal_outcome,
    discount_factor,
    weights=None,
    min_data_in_leaf=50,
    num_iter=100,
    binning_method="isotonic",   # {"quantile", "isotonic"}
    val_frac=0.2,
    patience=10,
    split_seed=123,
    refit_on_all_data: bool = True,
    verbose: bool = False,
):
    """
    Hybrid (binned) Bellman calibration via tabulated value iteration in value space.

    Steps:
      1) Fit a value-space discretizer on the full data.
      2) Run tabulated Bellman iteration for `num_iter` steps using all samples
         with the provided weights.
    """

    # ----------------------------
    # Basic setup
    # ----------------------------
    V = np.asarray(V, dtype=float)
    V_prime = np.asarray(V_prime, dtype=float)
    Y = np.asarray(Y, dtype=float)
    is_terminal_outcome = np.asarray(is_terminal_outcome, dtype=float)

    n = V.shape[0]
    assert V_prime.shape[0] == n == Y.shape[0] == is_terminal_outcome.shape[0]

    if weights is None:
        weights = np.ones_like(Y, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # Normalize weights safely (preserves zeros)
    w_mean = np.mean(weights)
    if w_mean > 0:
        weights = weights / w_mean

    # Initial Bellman target
    bellman_init = Y + discount_factor * (1.0 - is_terminal_outcome) * V_prime

    # ----------------------------
    # Discretization (value space)
    # ----------------------------
    (
        unique_V,
        inverse_indices,   # bin index for each V
        V_prime_indices,   # bin index for each V_prime
        V,                 # possibly transformed V
        V_prime,           # possibly transformed V_prime
        isotonic_calibrator,
        bin_edges,
    ) = discretizer(
        V=V,
        V_prime=V_prime,
        bellman_init=bellman_init,
        weights=weights,
        binning_method=binning_method,
        min_data_in_leaf=min_data_in_leaf,
    )

    n_bins = len(unique_V)

    # Weighted counts per bin on all data – do NOT fake counts
    weighted_counts = np.bincount(
        inverse_indices,
        weights=weights,
        minlength=n_bins,
    )

    # ----------------------------
    # Tabulated Bellman iteration
    # ----------------------------
    V_new_map = np.zeros(n_bins, dtype=float)  # zero init; zero-weight bins remain unless updated
    bellman_outcome = bellman_init.copy()

    for it in range(num_iter):
        # Weighted Bellman target per sample
        weighted_bellman_outcome = weights * bellman_outcome

        # Numerator per bin
        num = np.bincount(
            inverse_indices,
            weights=weighted_bellman_outcome,
            minlength=n_bins,
        )

        # Only update bins with positive total weight; keep others unchanged
        V_new_map_candidate = V_new_map.copy()
        mask_pos = weighted_counts > 0
        V_new_map_candidate[mask_pos] = num[mask_pos] / weighted_counts[mask_pos]

        # Update Bellman targets for next iteration
        V_prime_new_full = V_new_map_candidate[V_prime_indices]
        bellman_outcome = (
            Y + discount_factor * (1.0 - is_terminal_outcome) * V_prime_new_full
        )
        V_new_map = V_new_map_candidate

        if verbose and (it % 10 == 0 or it == num_iter - 1):
            print(f"[IBC] iter {it+1}/{num_iter}")

    final_V_new_map = V_new_map

    # ----------------------------
    # Final calibrator
    # ----------------------------
    def calibrator(V_in):
        V_in = np.asarray(V_in, dtype=float)

        # Optional isotonic pre-calibration of V_in
        if isotonic_calibrator is not None:
            V_in = isotonic_calibrator(V_in)

        if binning_method == "quantile":
            # Use the learned quantile bin_edges from discretizer
            idx = np.digitize(V_in, bin_edges[1:-1], right=True)
        else:  # "isotonic": bins are sorted unique_V values
            idx = np.clip(np.searchsorted(unique_V, V_in), 0, n_bins - 1)

        return final_V_new_map[idx]

    return calibrator


 
def iterated_bellman_isotonic_calibration(
    V,
    V_prime,
    Y,
    discount_factor,
    weights=None,
    is_terminal_outcome=None,
    min_data_in_leaf: int = 30,
    num_iterations: int = 1000,
    verbose: bool = True,
):
    """
    Iterated Bellman calibration in 1D (value space) using isotonic regression.

    At each iteration k:
      1. Build Bellman targets T_k using the current calibrator (or V_prime at k=0).
      2. Fit an isotonic calibrator g_{k+1} : V -> T_k on the full data.
      3. Update the current calibrator to g_{k+1}.

    Runs for a fixed number of iterations `num_iterations` with no early stopping
    or validation-based tuning, and returns the final calibration map.
    """

    V = np.asarray(V, dtype=float)
    V_prime = np.asarray(V_prime, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n = V.shape[0]

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if is_terminal_outcome is None:
        is_terminal_outcome = np.zeros(n, dtype=float)
    else:
        is_terminal_outcome = np.asarray(is_terminal_outcome, dtype=float)

    if verbose:
        num_segments_approx = int(np.round(n / max(min_data_in_leaf, 1)))
        print(
            f"[Bellman-Cal/iso] Using {n} points, "
            f"min_data_in_leaf = {min_data_in_leaf} "
            f"(≈ {num_segments_approx} constant segments)"
        )

    # --------------------------------------------------
    # Outer FVI loop with isotonic calibrator (no early stopping)
    # --------------------------------------------------
    calibrator_curr = None

    for it in range(num_iterations):
        # 1) Build Bellman targets T_k using current calibrator
        if calibrator_curr is None:
            # First iteration: bootstrap with raw V_prime
            V_next = V_prime
        else:
            V_next = calibrator_curr.predict(V_prime)

        td_target = Y + discount_factor * (1.0 - is_terminal_outcome) * V_next

        # 2) Fit isotonic calibrator g_{k+1} on full data
        calibrator_curr = IsotonicCalibrator(
            min_data_in_leaf=min_data_in_leaf,
        ).fit(V, td_target, weights=weights)

        # Optional logging on train
        if verbose and (it % 10 == 0 or it == num_iterations - 1):
            preds = calibrator_curr.predict(V)
            mse = np.mean((preds - td_target) ** 2)
            print(
                f"[Bellman-Cal/iso] Iter {it+1}/{num_iterations} "
                f"- train TD MSE: {mse:.6f}"
            )

    calibrator_value = calibrator_curr

    def calibrator_map(V_new):
        """
        Apply the learned calibration map to new value estimates V_new.
        """
        V_new = np.asarray(V_new, dtype=float)
        return calibrator_value.predict(V_new)

    return calibrator_map


 
 
class IsotonicCalibrator(BaseEstimator, RegressorMixin):
    """
    Isotonic regression calibrator using XGBoost.

    Parameters:
    max_depth (int): Maximum depth of each tree.
    num_leaves (int): Maximum number of leaves in one tree.
    min_data_in_leaf (int): Minimum number of data points in a leaf.
    """
    def __init__(self, max_depth=50, min_data_in_leaf=100):
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf

    def fit(self, f, y, weights=None):
        """
        Fits the calibrator to the data.

        Parameters:
        f (np.ndarray): Features for calibration.
        y (np.ndarray): Target values.
        weights (np.ndarray, optional): Sample weights.

        Returns:
        self: Fitted calibrator.
        """
        params_xgb = {
            'max_depth': self.max_depth,
            'monotone_constraints': '(1)',
            'learning_rate': 1,
            'min_child_weight': self.min_data_in_leaf,
            'alpha': 0,  # L1 regularization term on weights
            'lambda': 0,  # L2 regularization term on weights
            'verbosity': 0,
            'eta': 1,
            'gamma': 0,
            'objective': 'reg:squarederror'  # Corrected objective parameter
        }

        data = xgb.DMatrix(data=f.reshape(-1, 1), label=y, weight=weights if weights is not None else None)
        self.iso_fit = xgb.train(params=params_xgb, dtrain=data, num_boost_round=1)

        return self

    def predict(self, x):
        """
        Makes predictions using the fitted calibrator.

        Parameters:
        x (np.ndarray): Input features.

        Returns:
        np.ndarray: Calibrated predictions.
        """
        data = xgb.DMatrix(data=x.reshape(-1, 1))
        pred = self.iso_fit.predict(data)
        return pred


      
def calibrate(f: np.ndarray, y: np.ndarray, weights = None, min_data_in_leaf = 200):
        """
        Creates a 1D calibration function based on the specified learner with cross-validation.

        Args:
            f (np.ndarray): Array of uncalibrated predictions (features).
            y (np.ndarray): Array of actual outcomes (labels).
            weights (np.ndarray): Array of sample weights.

        Returns:
            function: A function that takes an array of model predictions and returns calibrated predictions.
        """
        if weights is None:
            weights = np.ones_like(f)
        
        learner = IsotonicCalibrator(min_data_in_leaf = min_data_in_leaf)
        estimator = learner.fit(f.reshape(-1, 1), y, weights = weights)
    
        def transform(x):
            return estimator.predict(x.reshape(-1, 1))
        
        return transform



import numpy as np

def _weighted_quantile(x, w, quantiles):
    """
    Compute weighted quantiles of x with nonnegative weights w.
    quantiles ∈ [0, 1].
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    assert x.shape[0] == w.shape[0]
    assert np.all(w >= 0)

    # Sort by x
    sorter = np.argsort(x)
    x_sorted = x[sorter]
    w_sorted = w[sorter]

    total_w = w_sorted.sum()
    if total_w <= 0:
        # Fallback: unweighted quantiles
        return np.quantile(x, quantiles)

    cum_w = np.cumsum(w_sorted) / total_w
    return np.interp(quantiles, cum_w, x_sorted)


def discretizer(
    V,
    V_prime,
    bellman_init,
    weights,
    binning_method="quantile",   # {"quantile", "isotonic"}
    min_data_in_leaf=50,
):
    """
    Discretize V and V_prime into bins and (optionally) apply isotonic calibration.

    For binning_method == "quantile", bin edges are computed using
    weighted quantiles with the provided weights.
    """

    V = np.asarray(V, dtype=float)
    V_prime = np.asarray(V_prime, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = V.shape[0]

    bin_edges = None  # only used for quantile binning

    # --------------------
    # 1. Optional isotonic
    # --------------------
    if binning_method == "isotonic":
        calibrator_iso = calibrate(
            f=V,
            y=bellman_init,
            weights=weights,
            min_data_in_leaf=min_data_in_leaf,
        )

        def isotonic_calibrator(V_in):
            return calibrator_iso(V_in)

        V = isotonic_calibrator(V)
        V_prime = isotonic_calibrator(V_prime)

        # Exact-value bins after isotonic
        unique_V, inverse_indices = np.unique(V, return_inverse=True)
        n_bins = len(unique_V)

    # --------------------
    # 2. Quantile binning (WEIGHTED)
    # --------------------
    elif binning_method == "quantile":
        K = max(2, int(np.floor(n / min_data_in_leaf)))
        K = min(K, n)

        quantiles = np.linspace(0.0, 1.0, K + 1)
        # Weighted quantile bin edges
        bin_edges = _weighted_quantile(V, weights, quantiles)

        # Bin indices for V
        inverse_indices = np.digitize(V, bin_edges[1:-1], right=True)
        n_bins = K

        # Representative value per bin = weighted mean of V
        unique_V = np.zeros(n_bins, dtype=float)
        for k in range(n_bins):
            mask = inverse_indices == k
            if np.any(mask):
                unique_V[k] = np.average(V[mask], weights=weights[mask])
            else:
                # empty bin: use mid-point of edges
                unique_V[k] = 0.5 * (bin_edges[k] + bin_edges[k + 1])

        isotonic_calibrator = None
    else:
        raise ValueError("binning_method must be 'quantile' or 'isotonic'")

    # --------------------
    # 3. Map V_prime to bins
    # --------------------
    if binning_method == "quantile":
        V_prime_indices = np.digitize(V_prime, bin_edges[1:-1], right=True)
    else:  # isotonic: unique_V is sorted
        V_prime_indices = np.clip(
            np.searchsorted(unique_V, V_prime),
            0,
            n_bins - 1,
        )

    return (
        unique_V,
        inverse_indices,
        V_prime_indices,
        V,
        V_prime,
        isotonic_calibrator if binning_method == "isotonic" else None,
        bin_edges,
    )

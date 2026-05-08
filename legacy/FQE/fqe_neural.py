import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ============================================================
# 5. Neural FQE wrapper  
# ============================================================

def fit_fqe_neural(
    S,
    A,
    S_next,
    A_next,
    rewards,
    discount_factor,
    is_terminal_outcome=None,
    weights=None,
    test_size=0.2,
    train_indices=None,
    seed=42,
    verbose=False,
    n_fvi_iters: int = 30,
    epochs_per_iter: int = 3,
    batch_size: int = 4096,
    lr: float = 1e-4,
    patience: int = 3,
):
    """
    Neural FQE via importance-weighted FVI.

    Arguments
    ---------
    S, A, S_next, A_next : array-like
        State, action, next-state, next-action. A and A_next are kept
        for API compatibility but are not directly used here.
    rewards : array-like, shape (n,)
        Immediate rewards.
    discount_factor : float
        Gamma in [0, 1).
    is_terminal_outcome : array-like, optional
        1 if the transition leads to terminal outcome, 0 otherwise.
        If None, all transitions are treated as non-terminal.
    weights : array-like, optional
        Importance weights w_i (e.g., π(a|s)/b(a|s)). If None, all ones.
    test_size : float
        Fraction of data to use for validation if train_indices is None.
    train_indices : array-like of ints, optional
        Indices to use for training; the complement is used for validation.
    seed : int
        Random seed for numpy/torch.
    verbose : bool
        If True, prints extra info.
    n_fvi_iters : int
        Number of outer FVI (Bellman) iterations.
    epochs_per_iter : int
        Number of epochs of weighted regression per FVI iteration.
    batch_size : int
        Mini-batch size for SGD.
    lr : float
        Learning rate for Adam.
    patience : int
        Early-stopping patience in terms of FVI iterations (no improvement in
        weighted validation Bellman MSE).

    Returns
    -------
    result : dict
        {
          "value_net": fitted ValueNet,
          "state_mean": state_mean tensor,
          "state_std": state_std tensor,
          "train_idx": train_idx (np.ndarray),
          "val_idx": val_idx (np.ndarray),
        }
    """
    # ----------------------------
    # 0. Basic setup & casting
    # ----------------------------
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    S = np.asarray(S, dtype=np.float32)
    S_next = np.asarray(S_next, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)

    n, state_dim = S.shape

    if is_terminal_outcome is None:
        done = np.zeros(n, dtype=np.float32)
    else:
        done = np.asarray(is_terminal_outcome, dtype=np.float32)

    if weights is None:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)

    # Torch tensors
    S_t      = torch.as_tensor(S,      device=DEVICE)
    Snext_t  = torch.as_tensor(S_next, device=DEVICE)
    R_t      = torch.as_tensor(rewards, device=DEVICE)
    done_t   = torch.as_tensor(done,    device=DEVICE)
    w_t      = torch.as_tensor(weights, device=DEVICE)

    # ----------------------------
    # 1. Train/validation split
    # ----------------------------
    if train_indices is None:
        split = make_fvi_train_val_split(
            S_t, R_t, Snext_t, done_t, w_t,
            val_frac=test_size,
            device=DEVICE,
        )
    else:
        # respect provided train_indices
        train_idx = torch.as_tensor(train_indices, device=DEVICE, dtype=torch.long)
        N = S_t.shape[0]
        mask = torch.ones(N, dtype=torch.bool, device=DEVICE)
        mask[train_idx] = False
        val_idx = torch.arange(N, device=DEVICE)[mask]

        split = dict(
            S_train=S_t[train_idx],
            R_train=R_t[train_idx],
            Snext_train=Snext_t[train_idx],
            done_train=done_t[train_idx],
            w_train=w_t[train_idx],
            S_val=S_t[val_idx],
            R_val=R_t[val_idx],
            Snext_val=Snext_t[val_idx],
            done_val=done_t[val_idx],
            w_val=w_t[val_idx],
            train_idx=train_idx,
            val_idx=val_idx,
        )

    S_train     = split["S_train"]
    R_train     = split["R_train"]
    Snext_train = split["Snext_train"]
    done_train  = split["done_train"]
    w_train     = split["w_train"]

    S_val     = split["S_val"]
    R_val     = split["R_val"]
    Snext_val = split["Snext_val"]
    done_val  = split["done_val"]
    w_val     = split["w_val"]

    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]

    if verbose:
        print(f"[fit_fqe_neural] Train size: {train_idx.numel()}, Val size: {val_idx.numel()}")

    # ----------------------------
    # 2. State normalization from train
    # ----------------------------
    state_mean = S_train.mean(dim=0, keepdim=True)
    state_std  = S_train.std(dim=0, keepdim=True) + 1e-6

    # ----------------------------
    # 3. Initialize and fit value network
    # ----------------------------
    V = ValueNet(
        state_dim=state_dim,
        state_mean=state_mean,
        state_std=state_std,
    ).to(DEVICE)

    V = iw_fitted_value_iteration(
        V=V,
        S_train=S_train,
        R_train=R_train,
        Snext_train=Snext_train,
        done_train=done_train,
        w_train=w_train,
        S_val=S_val,
        R_val=R_val,
        Snext_val=Snext_val,
        done_val=done_val,
        w_val=w_val,
        gamma=discount_factor,
        n_fvi_iters=n_fvi_iters,
        epochs_per_iter=epochs_per_iter,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
    )

    V.eval()  # optional but nice for downstream use

    # ----------------------------
    # 5. Package result
    # ----------------------------
    result = dict(
        value_net=V,
        state_mean=state_mean,
        state_std=state_std,
        train_idx=train_idx.detach().cpu().numpy(),
        val_idx=val_idx.detach().cpu().numpy(),
    )
    return result

# ============================================================
# 4. Value network + IW-FVI
# ============================================================

class ValueNet(nn.Module):
    def __init__(self, state_dim, state_mean, state_std, hidden_sizes=(256, 256)):
        super().__init__()
        self.state_mean = state_mean
        self.state_std = state_std

        layers = []
        prev = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        s_norm = (s - self.state_mean) / self.state_std
        return self.net(s_norm).squeeze(-1)


def iw_fitted_value_iteration(
    V: nn.Module,
    S_train, R_train, Snext_train, done_train, w_train,
    S_val,   R_val,   Snext_val,   done_val,   w_val,
    gamma: float = 0.99,
    n_fvi_iters: int = 30,
    epochs_per_iter: int = 3,
    batch_size: int = 4096,
    lr: float = 1e-4,
    patience: int = 3,
    tol_rel: float = 1e-4,
):
    """
    Importance-weighted FVI with early stopping based on whether the regression
    step actually improves the Bellman loss for the *current* targets Y_k.

    At iteration k:
      - Build targets Y_k using V_k.
      - Compute L_old = E[(Y_k - V_k)^2].
      - Train V to get V_{k+1}.
      - Compute L_new = E[(Y_k - V_{k+1})^2].
      - If L_new > L_old (up to a small tolerance), increment bad_count.
        Stop when bad_count >= patience.

    Validation Bellman MSE is logged but not used for stopping.
    """
    opt = torch.optim.Adam(V.parameters(), lr=lr)
    V.train()

    N_train = S_train.shape[0]
    batch_size = min(batch_size, N_train)

    bad_count = 0

    for it in range(n_fvi_iters):
        # ------------------------------------------------
        # 1) Bellman targets and pre-update train loss
        # ------------------------------------------------
        with torch.no_grad():
            V_train_before  = V(S_train)
            V_next_train    = V(Snext_train)
            targets_train   = R_train + gamma * (1.0 - done_train) * V_next_train

            err_before = V_train_before - targets_train
            loss_before = (
                (w_train * err_before ** 2).sum()
                / (w_train.sum() + 1e-8)
            ).item()

        # ------------------------------------------------
        # 2) Weighted regression on Y_k (train)
        # ------------------------------------------------
        for epoch in range(epochs_per_iter):
            perm = torch.randperm(N_train, device=S_train.device)
            for start in range(0, N_train, batch_size):
                end = min(start + batch_size, N_train)
                idx = perm[start:end]

                s_b = S_train[idx]
                y_b = targets_train[idx]
                w_b = w_train[idx]

                v_pred = V(s_b)
                err = v_pred - y_b
                loss = (w_b * err ** 2).sum() / (w_b.sum() + 1e-8)

                opt.zero_grad()
                loss.backward()
                opt.step()

        # ------------------------------------------------
        # 3) Post-update train loss on SAME targets Y_k
        # ------------------------------------------------
        with torch.no_grad():
            V_train_after = V(S_train)
            err_after = V_train_after - targets_train
            loss_after = (
                (w_train * err_after ** 2).sum()
                / (w_train.sum() + 1e-8)
            ).item()

        # ------------------------------------------------
        # 4) Validation Bellman MSE (for monitoring only)
        # ------------------------------------------------
        with torch.no_grad():
            V_next_val  = V(Snext_val)
            targets_val = R_val + gamma * (1.0 - done_val) * V_next_val
            v_val       = V(S_val)
            err_val     = v_val - targets_val
            val_mse = (
                (w_val * err_val ** 2).sum().item()
                / (w_val.sum().item() + 1e-8)
            )

        print(
            f"[IW-FVI] iter {it+1}/{n_fvi_iters} "
            f"train Bellman loss: before={loss_before:.4f}, after={loss_after:.4f}, "
            f"val Bellman MSE={val_mse:.4f}"
        )

        # ------------------------------------------------
        # 5) Early stopping logic:
        #    if we failed to improve train Bellman loss,
        #    count as a "bad" iteration.
        # ------------------------------------------------
        # relative tolerance to avoid floating point noise
        if loss_after > loss_before * (1.0 + tol_rel):
            bad_count += 1
            print(f"[IW-FVI]   No improvement on Y_k; bad_count = {bad_count}.")
        else:
            bad_count = 0  # reset if we improved

        if bad_count >= patience:
            print(
                f"[IW-FVI] Early stopping at iter {it+1}: "
                f"train Bellman loss failed to improve for {patience} iterations."
            )
            break

    V.eval()
    return V




def make_fvi_train_val_split(
    S_t, R_t, Snext_t, done_t, w_t,
    val_frac: float = 0.1,
    device: torch.device = DEVICE,
):
    """
    Build a train/validation split for IW-FVI.
    """
    N = S_t.shape[0]
    perm = torch.randperm(N, device=device)
    n_val = int(val_frac * N)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    split = dict(
        S_train=S_t[train_idx],
        R_train=R_t[train_idx],
        Snext_train=Snext_t[train_idx],
        done_train=done_t[train_idx],
        w_train=w_t[train_idx],
        S_val=S_t[val_idx],
        R_val=R_t[val_idx],
        Snext_val=Snext_t[val_idx],
        done_val=done_t[val_idx],
        w_val=w_t[val_idx],
        train_idx=train_idx,
        val_idx=val_idx,
    )
    return split


 
"""
neural_dualdice.py

Neural DualDICE-style density-ratio estimation for stationary (gamma=1) RL.

This module provides:
    - MLP: a simple fully-connected network for scalar outputs
    - dualdice_nn_saddle: neural saddle-point training for a DualDICE objective
      to estimate stationary density ratios g(s,a) ≈ d_pi(s,a) / d_mu(s,a)

A small simulation + evaluation example is included under
`if __name__ == "__main__":` which assumes two helper functions:
    - simulate_mdp_data(...)
    - build_one_hot_features(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 (kept for potential extensions)
import numpy as np


# ---------------------------------------------------------------------
# Basic MLP builder
# ---------------------------------------------------------------------


class MLP(nn.Module):
    """
    Simple fully-connected network with configurable hidden layers and activation,
    returning a scalar output.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.

    hidden_sizes : int or tuple of int, default=(256, 256)
        Sizes of hidden layers. If an int is provided, a single hidden layer is used.

    activation : {"relu", "tanh", "silu", "gelu"}, default="relu"
        Activation function applied after each hidden linear layer.

    Notes
    -----
    The network architecture is:
        input_dim -> hidden_sizes[0] -> ... -> hidden_sizes[-1] -> 1

    The final output is squeezed to shape (batch,) for convenience in scalar
    regression / value estimation / density-ratio estimation settings.
    """

    def __init__(self, input_dim, hidden_sizes=(256, 256), activation="relu"):
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)

        layers = []
        prev_dim = input_dim

        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        Act = act_map.get(str(activation).lower(), nn.ReLU)

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(Act())
            prev_dim = h

        # Scalar output head
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Input feature batch.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Scalar outputs for each input.
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------
# Neural DualDICE (gamma=1) with Adam
# ---------------------------------------------------------------------


def dualdice_nn_saddle(
    x_sa,
    x_sp_ap,
    num_steps: int = 20_000,
    batch_size: int = 2_048,
    lr_g: float = 3e-4,
    lr_h: float = 3e-4,
    lambda_norm: float = 1.0,
    weight_decay_g: float = 1e-5,
    weight_decay_h: float = 1e-5,
    hidden_sizes_g=(256, 256),
    hidden_sizes_h=(256, 256),
    activation_g: str = "relu",
    activation_h: str = "relu",
    device=None,
    seed=None,
    verbose: bool = False,
):
    """
    Neural DualDICE-style saddle-point optimization (gamma=1, stationary case).

    We model two scalar networks:
        g_theta(s,a) = g_net(x_sa)   ~ stationary density ratio
        h_w(s,a)     = h_net(x_sa)   ~ critic / dual function

    and optimize the saddle-point objective

        L(θ, w) =
            E[ g_θ(S,A) * ( h_w(S,A) - h_w(S',A') ) ]
          - 0.5 * E[ h_w(S,A)^2 ]
          + (λ_norm / 2) * ( E[g_θ(S,A)] - 1 )^2,

    where expectations are over transitions (S,A,S',A') sampled from the
    (approximately) stationary behavior policy. We *minimize* L in θ and
    *maximize* in w. L2 regularization is implemented via Adam weight decay.

    Parameters
    ----------
    x_sa : array-like, shape (n, d_in)
        Feature vectors for current state–action pairs (S,A). For example,
        this can be raw states concatenated with one-hot actions, or a learned
        embedding.

    x_sp_ap : array-like, shape (n, d_in)
        Feature vectors for next state–action pairs (S',A') in the same
        feature space as x_sa.

    num_steps : int, default=20_000
        Number of gradient-descent/ascent iterations.

    batch_size : int, default=2_048
        Minibatch size used for stochastic optimization. Sampling is
        with replacement, so batch_size may exceed n.

    lr_g : float, default=3e-4
        Learning rate for g_net (θ).

    lr_h : float, default=3e-4
        Learning rate for h_net (w).

    lambda_norm : float, default=1.0
        Coefficient of the normalization penalty
        (E[g_θ(S,A)] - 1)^2, encouraging the estimated ratio to integrate
        to 1 under the behavior stationary distribution.

    weight_decay_g : float, default=1e-5
        L2 regularization for g_net parameters via Adam's weight_decay.

    weight_decay_h : float, default=1e-5
        L2 regularization for h_net parameters via Adam's weight_decay.

    hidden_sizes_g : int or tuple of int, default=(256, 256)
        Hidden layer sizes for g_net.

    hidden_sizes_h : int or tuple of int, default=(256, 256)
        Hidden layer sizes for h_net.

    activation_g : {"relu", "tanh", "silu", "gelu"}, default="relu"
        Activation for g_net hidden layers.

    activation_h : {"relu", "tanh", "silu", "gelu"}, default="relu"
        Activation for h_net hidden layers.

    device : str or torch.device, optional
        Device to run the optimization on (e.g. "cpu", "cuda").
        Defaults to "cuda" if available, else "cpu".

    seed : int or None, default=None
        Optional random seed for torch (and CUDA if used).

    verbose : bool, default=False
        If True, prints approximate minibatch objective every ~10% of training.

    Returns
    -------
    g_net : nn.Module
        Trained network approximating the stationary density ratio:
            g_net(x_sa) ≈ d_pi(s,a) / d_mu(s,a).

        At the end of training, g_net is hard-normalized so that
        E[g_net(S,A)] ≈ 1 under the empirical distribution of x_sa.

    h_net : nn.Module
        Trained critic network implementing the dual function h_w(s,a) =
        h_net(x_sa).
    """
    # ----- Device -----
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # ----- Data -----
    X = torch.as_tensor(x_sa, dtype=torch.float32, device=device)
    Xp = torch.as_tensor(x_sp_ap, dtype=torch.float32, device=device)
    n, d_in = X.shape
    assert Xp.shape == (n, d_in)

    if seed is not None:
        torch.manual_seed(seed)

    # ----- Networks -----
    g_net = MLP(d_in, hidden_sizes=hidden_sizes_g, activation=activation_g).to(device)
    h_net = MLP(d_in, hidden_sizes=hidden_sizes_h, activation=activation_h).to(device)

    # ----- Optimizers (L2 via weight_decay) -----
    opt_g = torch.optim.Adam(g_net.parameters(), lr=lr_g, weight_decay=weight_decay_g)
    opt_h = torch.optim.Adam(h_net.parameters(), lr=lr_h, weight_decay=weight_decay_h)

    # ----- Minibatch sampler -----
    def sample_batch():
        idx = torch.randint(n, (batch_size,), device=device)
        return X[idx], Xp[idx]

    # ================= TRAINING LOOP =================
    for t in range(num_steps):
        X_b, Xp_b = sample_batch()

        opt_g.zero_grad(set_to_none=True)
        opt_h.zero_grad(set_to_none=True)

        # Forward: compute g, h, h'
        g_vals = g_net(X_b)        # (batch,)
        h_vals = h_net(X_b)        # (batch,)
        h_p_vals = h_net(Xp_b)     # (batch,)
        diff = h_vals - h_p_vals   # (batch,)

        # Base DualDICE-style term
        L = (g_vals * diff - 0.5 * h_vals.pow(2)).mean()

        # Normalization penalty (E[g] - 1)^2
        if lambda_norm > 0.0:
            L = L + 0.5 * lambda_norm * (g_vals.mean() - 1.0).pow(2)

        # Backprop: this gives dL/dθ and dL/dw
        L.backward()

        # Flip sign of h_net gradients to ascend in w
        for p in h_net.parameters():
            if p.grad is not None:
                p.grad.mul_(-1.0)

        # Adam updates
        opt_g.step()
        opt_h.step()

        # Logging
        if verbose and (t % max(1, num_steps // 10) == 0):
            print(f"step {t:6d}: L_batch ≈ {L.item():.6f}")

    # ----- Final hard normalization on g_net -----
    # Enforce that E[g_net(S,A)] ≈ 1 over the empirical dataset.
    with torch.no_grad():
        g_full = g_net(X).mean()
        if g_full.abs() > 1e-8:
            scale = 1.0 / g_full
            for p in g_net.parameters():
                p.mul_(scale)

    return g_net, h_net

 
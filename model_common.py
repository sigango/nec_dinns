"""Shared core for the DINN / NEC dengue models.

One source of truth, imported by ``dinn_model`` and ``nec_model``:
  - constants (states, parameters, bounds),
  - the feature extractor that conditions the NEC calibration head,
  - the physics: a single ODE right-hand side + its log-space residual,
  - small neural-network helpers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Constants
# ============================================================================

# mu_h is held fixed (not learned); it is injected directly into the RHS.
MU_H_FIXED: float = 4e-4

# Fixed physics-loss weight; not adapted during training.
W_PHYS: float = 3.0

# 8 compartments emitted by the network. Itot is cumulative infections, with
# dItot/dt = theta_h * He.
STATE_NAMES: tuple[str, ...] = ("Hr", "Hs", "He", "Hi", "Me", "Ms", "Mi", "Itot")
STATE_INDEX = {n: i for i, n in enumerate(STATE_NAMES)}
IDX_ITOT = STATE_INDEX["Itot"]

# Backward-compatible alias for CSV / notebook code.
STATE_COLS = list(STATE_NAMES)

# 7 trainable parameters; mu_h is injected separately as MU_H_FIXED.
PARAM_NAMES: tuple[str, ...] = (
    "lambda", "beta_m", "mu_m", "theta_m",
    "beta_h", "theta_h", "gamma_h",
)

# Physical parameter bounds (linear space), centered on the reference DINN
# values (lambda ~ 30, etc.). Wide enough to remain a genuine inverse problem
# while staying in the regime where the full RHS sustains the outbreak.
PHYS_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "lambda":  (5.0, 55.0),
    "beta_m":  (0.01, 0.19),
    "mu_m":    (0.005, 0.095),
    "theta_m": (0.03, 0.57),
    "beta_h":  (0.01, 0.19),
    "theta_h": (0.05, 0.95),
    "gamma_h": (0.02, 0.38),
}


# ============================================================================
# Feature extractor (conditions the NEC calibration head)
# ============================================================================

def baseline_features(itot_series: np.ndarray) -> torch.Tensor:
    """5-D log1p quantiles (min, q25, median, q75, max) of an Itot series."""
    qs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    pts = np.quantile(np.clip(itot_series, 0.0, None), qs)
    return torch.tensor(np.log1p(pts), dtype=torch.float32)


# Backward-compatible alias (older callers used the *_from_cal name).
baseline_features_from_cal = baseline_features


# ============================================================================
# Physics: ODE right-hand side + log-space residual
# ============================================================================

def dengue_rhs(states: torch.Tensor, params: dict) -> torch.Tensor:
    """8-compartment mosquito-human dengue RHS (linear-space derivatives).

    `states` is (T, 8) in linear scale, order [Hr, Hs, He, Hi, Me, Ms, Mi, Itot].
    Frequency-dependent transmission (/H, /M) with susceptible births (mu_h * H).
    """
    Hr, Hs, He, Hi, Me, Ms, Mi, _Itot = states.unbind(dim=-1)
    lam = params["lambda"]
    beta_m = params["beta_m"]
    mu_m = params["mu_m"]
    theta_m = params["theta_m"]
    beta_h = params["beta_h"]
    theta_h = params["theta_h"]
    gamma_h = params["gamma_h"]
    mu_h = MU_H_FIXED  # constant injection

    H = torch.clamp(Hr + Hs + He + Hi, min=1e-8)   # total humans
    M = torch.clamp(Me + Ms + Mi, min=1e-8)        # total mosquitoes

    dHr = gamma_h * Hi - mu_h * Hr
    dHs = mu_h * H - beta_h * Mi * Hs / M - mu_h * Hs
    dHe = beta_h * Mi * Hs / M - (theta_h + mu_h) * He
    dHi = theta_h * He - (gamma_h + mu_h) * Hi
    dMe = beta_m * Hi * Ms / H - (theta_m + mu_m) * Me
    dMs = lam - beta_m * Hi * Ms / H - mu_m * Ms
    dMi = theta_m * Me - mu_m * Mi
    dItot = theta_h * He

    return torch.stack([dHr, dHs, dHe, dHi, dMe, dMs, dMi, dItot], dim=-1)


def compute_residuals(
    t: torch.Tensor,
    log_states: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """Log-space ODE residual r_i = d(log1p s_i)/dt - rhs_i / (1 + s_i).

    `log_states` (T, 8) are the network's log1p outputs. Requires
    `t.requires_grad=True` and `log_states` differentiable in `t`.
    """
    grads = []
    for i in range(log_states.shape[-1]):
        gi = torch.autograd.grad(
            log_states[:, i].sum(),
            t,
            retain_graph=True,
            create_graph=True,
        )[0]
        grads.append(gi.reshape(-1))
    dlog_dt = torch.stack(grads, dim=-1)            # d(log1p s)/dt
    states = torch.expm1(log_states)                # back to linear
    rhs = dengue_rhs(states, params)
    return dlog_dt - rhs / (1.0 + states)


# ============================================================================
# Neural-network helpers shared by every model
# ============================================================================

def sigmoid_bound(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Map an unbounded raw value into [lo, hi] via a sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def xavier_init(module: nn.Module) -> None:
    """Xavier-uniform weights, zero biases, for every Linear in `module`."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


def emulation_layers(hidden: int, n_layers: int) -> nn.Sequential:
    """Time-only emulator trunk: t (1 input) -> 8 raw state outputs, GELU."""
    layers: list[nn.Module] = [nn.Linear(1, hidden)]
    for _ in range(n_layers - 1):
        layers += [nn.GELU(), nn.Linear(hidden, hidden)]
    layers += [nn.GELU(), nn.Linear(hidden, len(STATE_NAMES))]
    return nn.Sequential(*layers)

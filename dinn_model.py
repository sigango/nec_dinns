"""DINN model: single MLP t -> 8 log1p states with 7 trainable ODE params.

Disease-Informed Neural Network (Shaier 2022 style). The ODE parameters are
sigmoid-bounded nn.Parameters; mu_h is a fixed constant injected in the RHS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_common import (
    MU_H_FIXED,
    PARAM_NAMES,
    emulation_layers,
    sigmoid_bound,
    xavier_init,
)


class DINN(nn.Module):
    """t -> 8 log1p states + 7 trainable params (sigmoid-bounded)."""

    def __init__(
        self,
        param_bounds: dict[str, tuple[float, float]],
        hidden: int = 20,
        n_layers: int = 8,
    ):
        super().__init__()
        self.mlp = emulation_layers(hidden, n_layers)
        self.param_raw = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in PARAM_NAMES
        })
        self.register_buffer("param_lo", torch.tensor(
            [param_bounds[n][0] for n in PARAM_NAMES], dtype=torch.float32))
        self.register_buffer("param_hi", torch.tensor(
            [param_bounds[n][1] for n in PARAM_NAMES], dtype=torch.float32))

        # Xavier-init the emulation MLP; params start at range midpoints
        # (param_raw == 0 -> sigmoid 0.5).
        xavier_init(self.mlp)

    def get_params(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for i, name in enumerate(PARAM_NAMES):
            out[name] = sigmoid_bound(
                self.param_raw[name].squeeze(), self.param_lo[i], self.param_hi[i],
            )
        out["mu_h"] = torch.tensor(MU_H_FIXED, device=self.param_lo.device)
        return out

    def log_states(self, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(t.reshape(-1, 1)))

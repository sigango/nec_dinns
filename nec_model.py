"""NEC model: calibration head + emulation MLP.

Net A maps outbreak features -> 7 ODE parameters. Net B is the same
time -> 8 log1p state emulator as DINN and carries the PINN training loss.
The two nets are trained jointly: gradients flow through both via the shared
data + physics loss. Itot comes from Net B's trajectory (no ODE solver).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_common import (
    MU_H_FIXED,
    PARAM_NAMES,
    PHYS_PARAM_BOUNDS,
    emulation_layers,
    sigmoid_bound,
    xavier_init,
)


class NEC(nn.Module):
    """Net A: features -> 7 params; Net B: t -> 8 log1p states."""

    def __init__(
        self,
        n_feat: int = 5,
        param_bounds: dict[str, tuple[float, float]] = PHYS_PARAM_BOUNDS,
        calib_hidden: int = 64,
        emu_hidden: int = 20,
        emu_layers: int = 8,
    ):
        super().__init__()

        # Net A: outbreak features -> 7 param raws. Features are supplied at
        # call time (get_params), never stored, so the head can be queried with
        # calibration features during training and projection features at eval.
        self.calib = nn.Sequential(
            nn.Linear(n_feat, calib_hidden), nn.Tanh(),
            nn.Linear(calib_hidden, calib_hidden), nn.Tanh(),
            nn.Linear(calib_hidden, len(PARAM_NAMES)),
        )

        # Net B: same emulation MLP as DINN.
        self.mlp = emulation_layers(emu_hidden, emu_layers)

        self.register_buffer("param_lo", torch.tensor(
            [param_bounds[n][0] for n in PARAM_NAMES], dtype=torch.float32))
        self.register_buffer("param_hi", torch.tensor(
            [param_bounds[n][1] for n in PARAM_NAMES], dtype=torch.float32))

        # Xavier-init the trunks, then zero the calib output layer so the head
        # starts at the midpoint of every parameter range (neutral prior).
        xavier_init(self.calib)
        xavier_init(self.mlp)
        nn.init.zeros_(self.calib[-1].weight)
        nn.init.zeros_(self.calib[-1].bias)

    def get_params(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map outbreak features -> bounded params. `features` is (n_feat,)."""
        raw = self.calib(features.reshape(1, -1)).squeeze(0)
        out: dict[str, torch.Tensor] = {}
        for i, name in enumerate(PARAM_NAMES):
            out[name] = sigmoid_bound(raw[i], self.param_lo[i], self.param_hi[i])
        out["mu_h"] = torch.tensor(MU_H_FIXED, device=self.param_lo.device)
        return out

    def log_states(self, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mlp(t.reshape(-1, 1)))

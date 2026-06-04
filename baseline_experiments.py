"""DINN + NEC baseline experiments (calibration / projection protocol).

Physics: full mosquito-human dengue RHS with frequency-dependent transmission
(/H, /M) and susceptible births (mu_h * H); Itot is the 8th compartment
(dItot/dt = theta_h * He). The network emits all 8 states in log1p space
(softplus); the residual is in log space (d(log1p s)/dt - rhs/(1+s)).
mu_h = 4e-4 is fixed; loss = MSE(log1p Itot) + w_phys(=3) * residual_MSE; train
on the calibration outbreak only, with the projection outbreak held out.

Models live in dedicated modules (shared core in model_common.py):
  - DINN (dinn_model.py): single MLP t -> 8 log1p states; 7 trainable ODE params.
  - NEC  (nec_model.py) : calibration head (features -> 7 params) + the same
    emulation MLP. Both nets train jointly; projection reuses the emulator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models and shared core live in dedicated modules.
from model_common import (
    MU_H_FIXED,
    W_PHYS,
    STATE_NAMES,
    IDX_ITOT,
    STATE_COLS,
    PARAM_NAMES,
    PHYS_PARAM_BOUNDS,
    baseline_features,
    baseline_features_from_cal,
    dengue_rhs,
    compute_residuals,
)
from dinn_model import DINN
from nec_model import NEC


# --- Region maps ---

NPC_REGION_DIR = {
    "bello": "bello", "cam": "cambodia", "iquitos": "iquitos",
    "phi": "philippines", "sanjuan": "sanjuan", "vn": "vietnam",
}

REGION_LABELS = {
    "bello": "Bello", "cam": "Cambodia", "iquitos": "Iquitos",
    "phi": "Philippines", "sanjuan": "San Juan", "vn": "Vietnam",
}


# --- Path helpers ---

def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def default_output_dir() -> Path:
    return _project_dir() / "outputs"


def default_data_dir() -> Path:
    return _project_dir() / "data_piml" / "npc"


def discover_regions(data_dir: str | Path | None = None) -> list[str]:
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    return [
        key
        for key, sub in NPC_REGION_DIR.items()
        if (root / sub / "data_cali.csv").exists()
        and (root / sub / "data_proj.csv").exists()
    ]


# --- NPC data loader (Itot only) ---

def _read_npc_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "CumulativeHumanInfected" in df.columns:
        col = "CumulativeHumanInfected"
    else:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError(f"no numeric column in {path}")
        col = numeric[0]
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


def load_npc_data(region: str, data_dir: Path | None = None) -> dict:
    """Read data_cali.csv and data_proj.csv for `region`.

    Returns {"cal": {"t": (T,), "itot": (T,)}, "proj": {...}}.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    if root.name != "npc" and (root / "npc").exists():
        root = root / "npc"
    region_dir = root / NPC_REGION_DIR[region]
    cal_path = region_dir / "data_cali.csv"
    proj_path = region_dir / "data_proj.csv"
    if not cal_path.exists() or not proj_path.exists():
        raise FileNotFoundError(
            f"missing NPC data under {region_dir} (need data_cali.csv and data_proj.csv)"
        )
    itot_cal = _read_npc_csv(cal_path)
    itot_proj = _read_npc_csv(proj_path)
    return {
        "cal": {
            "t": np.arange(len(itot_cal), dtype=np.float32),
            "itot": itot_cal.astype(np.float32),
        },
        "proj": {
            "t": np.arange(len(itot_proj), dtype=np.float32),
            "itot": itot_proj.astype(np.float32),
        },
    }


# --- Config ---

@dataclass
class BaselineConfig:
    epochs: int = 4000
    lr: float = 5e-4
    weight_decay: float = 1e-5
    w_phys: float = W_PHYS
    seed: int = 1234
    device: str = "cpu"
    grad_clip: float = 0.1       # gradient-norm clip
    warmup_frac: float = 0.2     # linear warm-up fraction, then linear decay
    log_every: int = 500


# --- Forward helpers ---

def _forward(
    model: nn.Module,
    t: torch.Tensor,
    features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Returns (log_states (T, 8), params).

    NEC requires outbreak features at call time; DINN ignores them.
    """
    if isinstance(model, DINN):
        return model.log_states(t), model.get_params()
    if isinstance(model, NEC):
        if features is None:
            raise ValueError("NEC requires features at forward time")
        # Net B (emulator) is time-only; Net A maps features -> params.
        return model.log_states(t), model.get_params(features)
    raise TypeError(f"unsupported model type {type(model).__name__}")


def _itot_pred(log_states: torch.Tensor) -> torch.Tensor:
    """Linear-scale Itot prediction from the network's log1p Itot output."""
    return torch.expm1(log_states[:, IDX_ITOT])


# --- Training ---

def _make_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_frac: float = 0.2,
) -> optim.lr_scheduler.LambdaLR:
    """Linear warm-up over the first `warmup_frac` of epochs, then linear decay
    (5e-4 peak LR, 20% warm-up by default)."""
    warmup_epochs = max(1, int(total_epochs * warmup_frac))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs            # 0 -> 1
        decay = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.0, 1.0 - decay)                       # 1 -> 0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one(
    model: nn.Module,
    t_cal: torch.Tensor,
    itot_cal: torch.Tensor,
    config: BaselineConfig,
    region: str,
    model_type: str,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Train on calibration only. Returns (summary, history).

    The projection outbreak is never seen here -- not in the gradient and not
    even as a monitored metric. It is used only in evaluate().
    """
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, config.epochs, config.warmup_frac)

    log_target_cal = torch.log1p(itot_cal.clamp(min=0.0))

    # NEC conditions Net A on the calibration outbreak only; gradients never
    # touch projection features (those are used at eval time under no_grad).
    features_cal: torch.Tensor | None = None
    if isinstance(model, NEC):
        features_cal = baseline_features(
            itot_cal.detach().cpu().numpy()
        ).to(itot_cal.device)

    history: list[dict] = []
    last_loss_data = float("nan")
    last_loss_phys = float("nan")

    start = time.perf_counter()
    epoch = 0
    for epoch in range(config.epochs):
        optimizer.zero_grad()

        log_states_cal, params = _forward(model, t_cal, features_cal)
        itot_log_pred = log_states_cal[:, IDX_ITOT]
        loss_data = F.mse_loss(itot_log_pred, log_target_cal)

        residuals = compute_residuals(t_cal, log_states_cal, params)
        loss_phys = (residuals ** 2).mean()

        loss = loss_data + config.w_phys * loss_phys

        if not torch.isfinite(loss):
            print(f"[{region}/{model_type}] non-finite loss at ep {epoch}; abort")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        last_loss_data = float(loss_data.item())
        last_loss_phys = float(loss_phys.item())

        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            row = {
                "region": region,
                "model": model_type,
                "epoch": epoch,
                "loss_data": last_loss_data,
                "loss_phys": last_loss_phys,
                "loss_total": float(loss.item()),
            }
            history.append(row)
            if verbose:
                print(
                    f"  ep {epoch:5d}  data={row['loss_data']:.3e}  "
                    f"phys={row['loss_phys']:.3e}"
                )

    elapsed = time.perf_counter() - start
    summary = {
        "region": region,
        "model": model_type,
        "epochs_run": epoch + 1,
        "train_seconds": elapsed,
        "loss_data_final": last_loss_data,
        "loss_phys_final": last_loss_phys,
    }
    return summary, history


# --- Evaluation ---

def _split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    norm = max(float(np.max(np.abs(y_true))), 1e-8)
    mse_norm = float(np.mean(((y_true - y_pred) / norm) ** 2))
    return {
        "mse": mse,
        "mse_normalized": mse_norm,
    }


def evaluate(
    model: nn.Module,
    t_cal: torch.Tensor,
    itot_cal: torch.Tensor,
    t_proj: torch.Tensor,
    itot_proj: torch.Tensor,
) -> dict:
    model.eval()
    is_nec = isinstance(model, NEC)
    with torch.no_grad():
        # NEC conditions Net A on each outbreak's features; Net B (the emulator)
        # is time-only and shared. Both models project via the emulator MLP on
        # the projection week grid -- no ODE solver, no projection gradient.
        features_cal = (
            baseline_features(itot_cal.detach().cpu().numpy()).to(itot_cal.device)
            if is_nec else None
        )
        log_states_cal = model.log_states(t_cal)
        itot_pred_cal = _itot_pred(log_states_cal)
        params_cal = model.get_params(features_cal) if is_nec else model.get_params()

        log_states_proj = model.log_states(t_proj)
        itot_pred_proj = _itot_pred(log_states_proj)
        params_proj = params_cal

    cal_pred = itot_pred_cal.detach().cpu().numpy()
    proj_pred = itot_pred_proj.detach().cpu().numpy()
    cal_obs = itot_cal.detach().cpu().numpy()
    proj_obs = itot_proj.detach().cpu().numpy()

    def _params_to_float(p: dict) -> dict:
        return {k: float(v.detach().cpu()) for k, v in p.items()}

    return {
        "cal_metrics": _split_metrics(cal_obs, cal_pred),
        "proj_metrics": _split_metrics(proj_obs, proj_pred),
        "cal_pred": cal_pred,
        "proj_pred": proj_pred,
        # `params` kept for backward compatibility (== calibration params).
        "params": _params_to_float(params_cal),
        "params_cal": _params_to_float(params_cal),
        "params_proj": _params_to_float(params_proj),
        # Projection uses the emulator MLP (no IC solve); kept for downstream
        # callers that still read these keys.
        "ic_proj_log1p": None,
        "proj_method": "mlp",
    }


# --- Public API ---

def _build_tensors(
    data: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_cal = torch.tensor(
        data["cal"]["t"], dtype=torch.float32, device=device, requires_grad=True,
    )
    t_proj = torch.tensor(
        data["proj"]["t"], dtype=torch.float32, device=device, requires_grad=True,
    )
    itot_cal = torch.tensor(data["cal"]["itot"], dtype=torch.float32, device=device)
    itot_proj = torch.tensor(data["proj"]["itot"], dtype=torch.float32, device=device)
    return t_cal, t_proj, itot_cal, itot_proj


def _build_model(model_type: str, data: dict, device: torch.device) -> nn.Module:
    if model_type == "dinn":
        return DINN().to(device)
    if model_type == "nec":
        # Features (5-D quantiles) are supplied at call time in train/eval,
        # so the model only needs to know the feature dimension here.
        return NEC(n_feat=5).to(device)
    raise ValueError(f"unknown model_type {model_type!r}")


def run_real_experiment(
    region: str,
    model_type: str,
    config: BaselineConfig | None = None,
    *,
    seed: int | None = None,
    data_dir: str | Path | None = None,
    range7_dir: str | Path | None = None,  # accepted for API compatibility; unused
    verbose: bool = True,
) -> dict:
    """Train one (region, model_type) pair. Returns a result dict.

    `seed` overrides config.seed, letting a caller loop over seeds for the
    multi-seed ensemble.
    """
    config = config or BaselineConfig()
    if seed is not None:
        config = replace(config, seed=seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    data = load_npc_data(region, data_dir)

    t_cal, t_proj, itot_cal, itot_proj = _build_tensors(data, device)
    model = _build_model(model_type, data, device)

    # Train on calibration only; the projection outbreak is held out entirely.
    summary, history = train_one(
        model, t_cal, itot_cal,
        config=config, region=region, model_type=model_type, verbose=verbose,
    )
    # Projection is touched here for the first time, under no_grad in evaluate().
    metrics = evaluate(model, t_cal, itot_cal, t_proj, itot_proj)

    return {
        **summary,
        "seed": config.seed,
        "region_label": REGION_LABELS[region],
        "cal_metrics": metrics["cal_metrics"],
        "proj_metrics": metrics["proj_metrics"],
        "cal_pred": metrics["cal_pred"],
        "proj_pred": metrics["proj_pred"],
        "cal_obs": data["cal"]["itot"],
        "proj_obs": data["proj"]["itot"],
        "cal_t": data["cal"]["t"],
        "proj_t": data["proj"]["t"],
        "params_final": metrics["params"],
        "params_cal_final": metrics["params_cal"],
        "params_proj_final": metrics["params_proj"],
        "ic_proj_log1p": metrics["ic_proj_log1p"],
        "proj_method": metrics["proj_method"],
        "history": history,
        "model_obj": model,
    }


def run_real_experiment_multiseed(
    region: str,
    model_type: str,
    config: BaselineConfig | None = None,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    data_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train one (region, model_type) pair across `seeds`; aggregate metrics."""
    runs = [
        run_real_experiment(
            region, model_type, config,
            seed=s, data_dir=data_dir, verbose=verbose,
        )
        for s in seeds
    ]
    return _aggregate_runs(runs)


def _aggregate_runs(runs: list[dict]) -> dict:
    """Mean/std of metrics over seeds; stack predictions for an uncertainty band."""
    first = runs[0]
    cal_preds = np.stack([r["cal_pred"] for r in runs], axis=0)    # (n_seeds, T_cal)
    proj_preds = np.stack([r["proj_pred"] for r in runs], axis=0)  # (n_seeds, T_proj)

    def stats(key: str, split: str) -> dict:
        vals = [r[f"{split}_metrics"][key] for r in runs]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "region": first["region"],
        "region_label": first["region_label"],
        "model": first["model"],
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "cal_metrics": {k: stats(k, "cal") for k in first["cal_metrics"]},
        "proj_metrics": {k: stats(k, "proj") for k in first["proj_metrics"]},
        "cal_pred_mean": cal_preds.mean(axis=0),
        "cal_pred_p5": np.percentile(cal_preds, 5, axis=0),
        "cal_pred_p95": np.percentile(cal_preds, 95, axis=0),
        "proj_pred_mean": proj_preds.mean(axis=0),
        "proj_pred_p5": np.percentile(proj_preds, 5, axis=0),
        "proj_pred_p95": np.percentile(proj_preds, 95, axis=0),
        "cal_t": first["cal_t"], "proj_t": first["proj_t"],
        "cal_obs": first["cal_obs"], "proj_obs": first["proj_obs"],
        "train_seconds_mean": float(np.mean([r["train_seconds"] for r in runs])),
        "proj_method": first.get("proj_method", "unknown"),
        "runs": runs,
    }


def _result_row(agg: dict) -> dict:
    """Results schema: proj normalized MSE + R^2 as mean/std over seeds.

    Backward-compatible columns (`mse_normalized`, `train_seconds`) hold the
    seed means so existing plots keep working.
    """
    proj = agg["proj_metrics"]
    return {
        "region": agg["region"],
        "model": agg["model"],
        "n_seeds": agg["n_seeds"],
        "mse_normalized": proj["mse_normalized"]["mean"],
        "train_seconds": agg["train_seconds_mean"],
        "mse_normalized_mean": proj["mse_normalized"]["mean"],
        "mse_normalized_std": proj["mse_normalized"]["std"],
        "r2_proj_mean": proj["r2"]["mean"],
        "r2_proj_std": proj["r2"]["std"],
        "train_seconds_mean": agg["train_seconds_mean"],
        "proj_method": agg.get("proj_method", "unknown"),
    }


def _prediction_rows(agg: dict) -> list[dict]:
    rows: list[dict] = []
    splits = (
        ("calibration", agg["cal_t"], agg["cal_obs"],
         agg["cal_pred_mean"], agg["cal_pred_p5"], agg["cal_pred_p95"]),
        ("projection", agg["proj_t"], agg["proj_obs"],
         agg["proj_pred_mean"], agg["proj_pred_p5"], agg["proj_pred_p95"]),
    )
    for split, ts, obs, mean, p5, p95 in splits:
        for ti, oi, mi, lo, hi in zip(ts, obs, mean, p5, p95):
            rows.append({
                "region": agg["region"],
                "region_label": agg["region_label"],
                "model": agg["model"],
                "split": split,
                "target": "Itot",
                "t": float(ti),
                "observed": float(oi),
                "predicted": float(mi),        # backward-compatible (== mean)
                "predicted_mean": float(mi),
                "predicted_p5": float(lo),
                "predicted_p95": float(hi),
            })
    return rows


def _history_rows(agg: dict) -> list[dict]:
    # Keep the first seed's loss history (tagged with its seed) for clean curves.
    first = agg["runs"][0]
    return [{**h, "seed": first["seed"], "region_label": agg["region_label"]}
            for h in first["history"]]


def run_all_baseline_table(
    regions: Iterable[str] | None = None,
    model_types: Iterable[str] = ("dinn", "nec"),
    config: BaselineConfig | None = None,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    data_dir: str | Path | None = None,
    range7_dir: str | Path | None = None,  # accepted for API compatibility; unused
    output_csv: str | Path | None = None,
    prediction_csv: str | Path | None = None,
    loss_history_csv: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Loop over `regions` x `model_types`, each as a multi-seed ensemble.

    Writes CSVs (results with mean/std, predictions with p5/p95 bands) and
    returns the result table.
    """
    config = config or BaselineConfig()
    region_list = list(regions) if regions is not None else discover_regions(data_dir)
    seed_list = list(seeds)

    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    history_rows: list[dict] = []

    for region in region_list:
        for model_type in model_types:
            if verbose:
                print(
                    f"\n=== {REGION_LABELS.get(region, region)} / "
                    f"{model_type.upper()} | {len(seed_list)} seeds ==="
                )
            agg = run_real_experiment_multiseed(
                region, model_type, config,
                seeds=seed_list, data_dir=data_dir, verbose=verbose,
            )
            result_rows.append(_result_row(agg))
            prediction_rows.extend(_prediction_rows(agg))
            history_rows.extend(_history_rows(agg))

    df = pd.DataFrame(result_rows)
    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"wrote results -> {output_csv}")
    if prediction_csv is not None:
        Path(prediction_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(prediction_csv, index=False)
        if verbose:
            print(f"wrote predictions -> {prediction_csv}")
    if loss_history_csv is not None:
        Path(loss_history_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_rows).to_csv(loss_history_csv, index=False)
        if verbose:
            print(f"wrote loss history -> {loss_history_csv}")

    return df


__all__ = [
    "MU_H_FIXED",
    "W_PHYS",
    "STATE_NAMES",
    "STATE_COLS",
    "PARAM_NAMES",
    "PHYS_PARAM_BOUNDS",
    "NPC_REGION_DIR",
    "REGION_LABELS",
    "BaselineConfig",
    "DINN",
    "NEC",
    "load_npc_data",
    "baseline_features",
    "baseline_features_from_cal",
    "dengue_rhs",
    "compute_residuals",
    "train_one",
    "evaluate",
    "run_real_experiment",
    "run_real_experiment_multiseed",
    "run_all_baseline_table",
    "discover_regions",
    "default_output_dir",
    "default_data_dir",
]

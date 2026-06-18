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
    load_param_bounds,
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
    return _project_dir() / "data"


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
    param_log_every: int = 10    # snapshot trainable ODE params every N epochs


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
) -> dict:
    """Train on calibration only. Returns a summary dict.

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

    last_loss_data = float("nan")
    last_loss_phys = float("nan")
    params_trace: list[dict] = []

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

        if epoch % config.param_log_every == 0 or epoch == config.epochs - 1:
            params_trace.append({
                "epoch": epoch,
                **{name: float(params[name].detach().cpu()) for name in PARAM_NAMES},
            })

        if verbose and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(f"  ep {epoch:5d}  data={last_loss_data:.3e}  phys={last_loss_phys:.3e}")

    elapsed = time.perf_counter() - start
    summary = {
        "region": region,
        "model": model_type,
        "epochs_run": epoch + 1,
        "train_seconds": elapsed,
        "loss_data_final": last_loss_data,
        "loss_phys_final": last_loss_phys,
        "params_trace": params_trace,
    }
    return summary


# --- Evaluation ---

def _split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
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


def _build_model(
    model_type: str,
    device: torch.device,
    param_bounds: dict[str, tuple[float, float]],
) -> nn.Module:
    if model_type == "dinn":
        return DINN(param_bounds=param_bounds).to(device)
    if model_type == "nec":
        return NEC(n_feat=5, param_bounds=param_bounds).to(device)
    raise ValueError(f"unknown model_type {model_type!r}")


def run_real_experiment(
    region: str,
    model_type: str,
    config: BaselineConfig | None = None,
    *,
    seed: int | None = None,
    data_dir: str | Path | None = None,
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
    param_bounds = load_param_bounds(NPC_REGION_DIR[region])
    model = _build_model(model_type, device, param_bounds)

    # Train on calibration only; the projection outbreak is held out entirely.
    summary = train_one(
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
    """Mean/std of metrics over seeds; average predictions across seeds."""
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
        "proj_pred_mean": proj_preds.mean(axis=0),
        "cal_t": first["cal_t"], "proj_t": first["proj_t"],
        "cal_obs": first["cal_obs"], "proj_obs": first["proj_obs"],
        "train_seconds_total": float(np.sum([r["train_seconds"] for r in runs])),
        "runs": runs,
    }


def _result_row(agg: dict) -> dict:
    """Results schema: region/model identifiers plus projection normalized MSE
    (mean/std over seeds) and total training time across all seeds."""
    proj = agg["proj_metrics"]
    return {
        "region": agg["region"],
        "model": agg["model"],
        "n_seeds": agg["n_seeds"],
        "mse_normalized": proj["mse_normalized"]["mean"],
        "train_seconds": agg["train_seconds_total"],
        "mse_normalized_mean": proj["mse_normalized"]["mean"],
        "mse_normalized_std": proj["mse_normalized"]["std"],
    }


def _prediction_rows(agg: dict) -> list[dict]:
    rows: list[dict] = []
    splits = (
        ("calibration", agg["cal_t"], agg["cal_obs"], agg["cal_pred_mean"]),
        ("projection", agg["proj_t"], agg["proj_obs"], agg["proj_pred_mean"]),
    )
    for split, ts, obs, mean in splits:
        for ti, oi, mi in zip(ts, obs, mean):
            rows.append({
                "region": agg["region"],
                "region_label": agg["region_label"],
                "model": agg["model"],
                "split": split,
                "target": "Itot",
                "t": float(ti),
                "observed": float(oi),
                "predicted": float(mi),
                "predicted_mean": float(mi),
            })
    return rows


def _seed_result_row(run: dict) -> dict:
    """Per-seed results schema: same shape as `_result_row` but for one run."""
    return {
        "region": run["region"],
        "model": run["model"],
        "seed": run["seed"],
        "mse_normalized": run["proj_metrics"]["mse_normalized"],
        "train_seconds": run["train_seconds"],
    }


def _seed_prediction_rows(run: dict) -> list[dict]:
    """Per-seed prediction schema: same columns as `_prediction_rows` but for one run."""
    rows: list[dict] = []
    splits = (
        ("calibration", run["cal_t"], run["cal_obs"], run["cal_pred"]),
        ("projection", run["proj_t"], run["proj_obs"], run["proj_pred"]),
    )
    for split, ts, obs, pred in splits:
        for ti, oi, pi in zip(ts, obs, pred):
            rows.append({
                "region": run["region"],
                "region_label": run["region_label"],
                "model": run["model"],
                "seed": run["seed"],
                "split": split,
                "target": "Itot",
                "t": float(ti),
                "observed": float(oi),
                "predicted": float(pi),
                "predicted_mean": float(pi),
            })
    return rows


def _seed_param_rows(run: dict) -> list[dict]:
    """Per-seed parameter-trace schema: one row per logged epoch."""
    return [
        {
            "region": run["region"],
            "model": run["model"],
            "seed": run["seed"],
            **snapshot,
        }
        for snapshot in run["params_trace"]
    ]


def run_all_baseline_table(
    regions: Iterable[str] | None = None,
    model_types: Iterable[str] = ("dinn", "nec"),
    config: BaselineConfig | None = None,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    params_dir: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Loop over `regions` x `model_types`, each as a multi-seed ensemble.

    For each region, writes:
    - `<results_dir>/<region>/<region>.csv` and
      `<predictions_dir>/<region>/<region>.csv`: DINN and NEC rows together,
      averaged over seeds (as before).
    - `<results_dir>/<region>/seed<N>/<model>.csv` and
      `<predictions_dir>/<region>/seed<N>/<model>.csv`: per-seed raw results
      and predictions for each model, mirroring the NPC
      `FigureGeneration/<region>/data/uni<NNN>/` layout.

    Returns the combined (seed-averaged) result table for all regions.
    """
    config = config or BaselineConfig()
    region_list = list(regions) if regions is not None else discover_regions(data_dir)
    seed_list = list(seeds)

    all_result_rows: list[dict] = []

    for region in region_list:
        region_result_rows: list[dict] = []
        region_prediction_rows: list[dict] = []
        seed_result_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}
        seed_prediction_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}
        seed_param_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}

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
            region_result_rows.append(_result_row(agg))
            region_prediction_rows.extend(_prediction_rows(agg))

            for run in agg["runs"]:
                seed_result_rows[run["seed"]].append(_seed_result_row(run))
                seed_prediction_rows[run["seed"]].extend(_seed_prediction_rows(run))
                seed_param_rows[run["seed"]].extend(_seed_param_rows(run))

        all_result_rows.extend(region_result_rows)

        if results_dir is not None:
            region_results_dir = Path(results_dir) / region
            region_results_dir.mkdir(parents=True, exist_ok=True)
            out_path = region_results_dir / f"{region}.csv"
            pd.DataFrame(region_result_rows).to_csv(out_path, index=False)
            if verbose:
                print(f"wrote results -> {out_path}")
            for seed in seed_list:
                seed_dir = region_results_dir / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                for model_type in model_types:
                    rows = [r for r in seed_result_rows[seed] if r["model"] == model_type]
                    out_path = seed_dir / f"{model_type}.csv"
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                if verbose:
                    print(f"wrote per-seed results -> {seed_dir}")

        if predictions_dir is not None:
            region_predictions_dir = Path(predictions_dir) / region
            region_predictions_dir.mkdir(parents=True, exist_ok=True)
            out_path = region_predictions_dir / f"{region}.csv"
            pd.DataFrame(region_prediction_rows).to_csv(out_path, index=False)
            if verbose:
                print(f"wrote predictions -> {out_path}")
            for seed in seed_list:
                seed_dir = region_predictions_dir / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                for model_type in model_types:
                    rows = [r for r in seed_prediction_rows[seed] if r["model"] == model_type]
                    out_path = seed_dir / f"{model_type}.csv"
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                if verbose:
                    print(f"wrote per-seed predictions -> {seed_dir}")

        if params_dir is not None:
            region_params_dir = Path(params_dir) / region
            region_params_dir.mkdir(parents=True, exist_ok=True)
            for seed in seed_list:
                seed_dir = region_params_dir / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                for model_type in model_types:
                    rows = [r for r in seed_param_rows[seed] if r["model"] == model_type]
                    out_path = seed_dir / f"{model_type}.csv"
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                if verbose:
                    print(f"wrote per-seed params -> {seed_dir}")

    return pd.DataFrame(all_result_rows)


__all__ = [
    "MU_H_FIXED",
    "W_PHYS",
    "STATE_NAMES",
    "STATE_COLS",
    "PARAM_NAMES",
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

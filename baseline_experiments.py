from __future__ import annotations

import csv
import random
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
from torch.autograd import grad


STATE_COLS = ["Hr", "Hs", "He", "Hi", "Me", "Ms", "Mi", "Itot"]

REGION_LABELS = {
    "bello": "Bello",
    "cam": "Cambodia",
    "iquitos": "Iquitos",
    "phi": "Philippines",
    "sanjuan": "San Juan",
    "vn": "Vietnam",
}

NEC_TO_DINN_PARAMS = {
    "lambda": "lambda",
    "betam": "beta_m",
    "mum": "mu_m",
    "thetam": "theta_m",
    "muh": "mu_h",
    "betah": "beta_h",
    "thetah": "theta_h",
    "gammah": "gamma_h",
}

DINN_PARAM_BOUNDS = {
    "lambda": (0.1, 200.0),
    "beta_m": (0.0, 1.0),
    "mu_m": (1e-5, 0.20),
    "theta_m": (0.001, 1.0),
    "mu_h": (0.0, 0.20),
    "beta_h": (0.0, 1.0),
    "theta_h": (0.01, 3.0),
    "gamma_h": (0.005, 3.0),
}

DEFAULT_PARAM_INIT = {
    "lambda": 30.0,
    "beta_m": 0.05,
    "mu_m": 0.025,
    "theta_m": 0.20,
    "mu_h": 0.0001,
    "beta_h": 0.05,
    "theta_h": 0.30,
    "gamma_h": 1.0,
}

NEC_PARAM_BOUNDS = {
    nec_key: DINN_PARAM_BOUNDS[dinn_key]
    for nec_key, dinn_key in NEC_TO_DINN_PARAMS.items()
}


@dataclass
class _BaseConfig:
    epochs: int = 10000
    lr: float = 5e-4
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.7
    scheduler_patience: int = 3000
    min_lr: float = 1e-7
    seed: int = 1234
    device: str = "cpu"
    w_itot: float = 1.0
    w_phys_real: float = 0.3


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data_piml"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_tanh_bound(value: float, lo: float, hi: float) -> float:
    scaled = (2.0 * value - hi - lo) / max(hi - lo, 1e-12)
    scaled = float(np.clip(scaled, -0.999, 0.999))
    return float(np.arctanh(scaled))


def tanh_bound(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.tanh(raw) * ((hi - lo) / 2.0) + ((hi + lo) / 2.0)


class _BaseDINNBaseline(nn.Module):
    class NetDengue(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(1, 20)] + [nn.Linear(20, 20) for _ in range(7)]
            )
            self.out = nn.Linear(20, len(STATE_COLS))

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            x = t.reshape(-1, 1)
            for layer in self.layers:
                x = torch.tanh(layer(x))
            return self.out(x)

    def __init__(
        self,
        t_train,
        itot_train,
        inc_train=None,
        *,
        full_state: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.t_train = t_train
        self.itot_train = itot_train
        self.inc_train = inc_train
        self.full_state = full_state
        self.losses: list[float] = []
        self.loss_hist: list[float] = []
        self.loss_data_hist: list[float] = []
        self.loss_phys_hist: list[float] = []

        self.lambda_tilda = self._init_tilda("lambda")
        self.beta_m_tilda = self._init_tilda("beta_m")
        self.mu_m_tilda = self._init_tilda("mu_m")
        self.theta_m_tilda = self._init_tilda("theta_m")
        self.mu_h_tilda = self._init_tilda("mu_h")
        self.beta_h_tilda = self._init_tilda("beta_h")
        self.theta_h_tilda = self._init_tilda("theta_h")
        self.gamma_h_tilda = self._init_tilda("gamma_h")

        self.net_dengue = self.NetDengue().to(self.device)
        self.net_params = list(self.net_dengue.parameters())
        self.phys_params = [
            self.lambda_tilda,
            self.beta_m_tilda,
            self.mu_m_tilda,
            self.theta_m_tilda,
            self.mu_h_tilda,
            self.beta_h_tilda,
            self.theta_h_tilda,
            self.gamma_h_tilda,
        ]

    def _init_tilda(self, name: str) -> nn.Parameter:
        lo, hi = DINN_PARAM_BOUNDS[name]
        init = inverse_tanh_bound(DEFAULT_PARAM_INIT[name], lo, hi)
        return nn.Parameter(torch.tensor([init], dtype=torch.float32, device=self.device))

    @property
    def lambda_(self) -> torch.Tensor:
        return tanh_bound(self.lambda_tilda, *DINN_PARAM_BOUNDS["lambda"])

    @property
    def beta_m(self) -> torch.Tensor:
        return tanh_bound(self.beta_m_tilda, *DINN_PARAM_BOUNDS["beta_m"])

    @property
    def mu_m(self) -> torch.Tensor:
        return tanh_bound(self.mu_m_tilda, *DINN_PARAM_BOUNDS["mu_m"])

    @property
    def theta_m(self) -> torch.Tensor:
        return tanh_bound(self.theta_m_tilda, *DINN_PARAM_BOUNDS["theta_m"])

    @property
    def mu_h(self) -> torch.Tensor:
        return tanh_bound(self.mu_h_tilda, *DINN_PARAM_BOUNDS["mu_h"])

    @property
    def beta_h(self) -> torch.Tensor:
        return tanh_bound(self.beta_h_tilda, *DINN_PARAM_BOUNDS["beta_h"])

    @property
    def theta_h(self) -> torch.Tensor:
        return tanh_bound(self.theta_h_tilda, *DINN_PARAM_BOUNDS["theta_h"])

    @property
    def gamma_h(self) -> torch.Tensor:
        return tanh_bound(self.gamma_h_tilda, *DINN_PARAM_BOUNDS["gamma_h"])

    def current_params_dict(self) -> dict[str, float]:
        return {
            "lambda": float(self.lambda_.detach().cpu()),
            "beta_m": float(self.beta_m.detach().cpu()),
            "mu_m": float(self.mu_m.detach().cpu()),
            "theta_m": float(self.theta_m.detach().cpu()),
            "mu_h": float(self.mu_h.detach().cpu()),
            "beta_h": float(self.beta_h.detach().cpu()),
            "theta_h": float(self.theta_h.detach().cpu()),
            "gamma_h": float(self.gamma_h.detach().cpu()),
        }


class CalibNet(nn.Module):
    def __init__(self, param_bounds: dict[str, tuple[float, float]], input_dim: int = 10):
        super().__init__()
        self.param_bounds = dict(param_bounds)
        self.param_names = list(param_bounds.keys())
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(self.param_names))
        self._init_output_to_default_params()

    def _init_output_to_default_params(self) -> None:
        with torch.no_grad():
            self.fc3.weight.zero_()
            for i, name in enumerate(self.param_names):
                dinn_name = NEC_TO_DINN_PARAMS.get(name, name)
                lo, hi = self.param_bounds[name]
                init = DEFAULT_PARAM_INIT[dinn_name]
                self.fc3.bias[i] = inverse_tanh_bound(init, lo, hi)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = torch.tanh(self.fc1(x))
        z = torch.tanh(self.fc2(z))
        raw = self.fc3(z).squeeze(0)
        return {
            name: tanh_bound(raw[i], *self.param_bounds[name])
            for i, name in enumerate(self.param_names)
        }


class EmuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(1, 32)] + [nn.Linear(32, 32) for _ in range(3)]
        )
        self.out = nn.Linear(32, len(STATE_COLS))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.reshape(-1, 1)
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.out(x)


class _BaseNECBaseline(nn.Module):
    def __init__(
        self,
        t_train,
        itot_train,
        inc_train=None,
        *,
        full_state: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.t_train = t_train
        self.itot_train = itot_train
        self.inc_train = inc_train
        self.full_state = full_state
        self.losses: list[float] = []
        self.loss_data_hist: list[float] = []
        self.calib = CalibNet(NEC_PARAM_BOUNDS).to(self.device)
        self.emu = EmuNet().to(self.device)
        self.calib_net = self.calib
        self.emu_net = self.emu
        self.register_buffer("calib_input", torch.zeros(1, 10, dtype=torch.float32, device=self.device))
        self.loss_hist: list[float] = []
        self.loss_phys_hist: list[float] = []

    def current_params_dict(self) -> dict[str, float]:
        params = self.calib(self.calib_input)
        return {k: float(v.detach().cpu()) for k, v in params.items()}


PARAM_BOUNDS = {
    "lambda":  (0.025, 1.7),
    "beta_m":  (0.01,  2.5),
    "mu_m":    (0.005, 1.5),
    "theta_m": (0.01,  3.0),
    "mu_h":    (4e-4,  4e-4),
    "beta_h":  (0.01,  2.0),
    "theta_h": (0.01,  2.0),
    "gamma_h": (0.01,  4.0),
}
MU_H_FIXED: float = 4e-4

DEFAULT_PARAM_INIT = {
    "lambda":  0.5,
    "beta_m":  0.5,
    "mu_m":    0.5,
    "theta_m": 0.5,
    "mu_h":    MU_H_FIXED,
    "beta_h":  0.5,
    "theta_h": 0.5,
    "gamma_h": 1.0,
}

DINN_PARAM_BOUNDS = PARAM_BOUNDS
NEC_PARAM_BOUNDS = {
    nec_key: PARAM_BOUNDS[dinn_key]
    for nec_key, dinn_key in NEC_TO_DINN_PARAMS.items()
}

NPC_REGION_DIR = {
    "bello": "bello",
    "cam": "cambodia",
    "iquitos": "iquitos",
    "phi": "philippines",
    "sanjuan": "sanjuan",
    "vn": "vietnam",
}

WARMUP_PROJ_IC_EPOCHS: int = 500


@dataclass
class BaselineConfig(_BaseConfig):
    epochs: int = 20000
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1500
    min_lr: float = 1e-6

    w_consistency: float = 0.5
    w_anchor: float = 0.1
    w_inc_soft: float = 0.2
    use_inc_soft: bool = True

    use_adaptive_phys: bool = True
    adaptive_alpha: float = 0.5
    adaptive_ema: float = 0.9
    adaptive_update_every: int = 100
    adaptive_w_min: float = 0.1
    adaptive_w_max: float = 10.0
    adaptive_warmup_epochs: int = 200

    t_cal_weeks: int = 48
    t_proj_weeks: int = 48
    projection_ic_policy: str = "A"
    w_ic_prior: float = 1e-3
    protocol: str = "two_window"
    derive_incidence_from_itot: bool = True

    cal_proj_gap_weeks: float = 1.0

    param_bounds_strategy: str = "region"
    range7_dir: str | None = None

    itot_loss_mode: str = "log_mse"
    w_itot_t0_anchor: float = 1.0
    w_itot_linear: float = 0.5

    use_horizon_cap: bool = True
    horizon_cap_ratio: float = 2.5
    w_horizon: float = 0.05

    proj_ic_continuity_w: float = 0.1


def _npc_root(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        return default_data_dir() / "npc"
    path = Path(data_dir)
    if path.name == "npc":
        return path
    if (path / "npc").exists():
        return path / "npc"
    return path


def discover_regions(data_dir: str | Path | None = None) -> list[str]:
    root = _npc_root(data_dir)
    return [
        key
        for key, dirname in NPC_REGION_DIR.items()
        if (root / dirname / "data_cali.csv").exists()
        and (root / dirname / "data_proj.csv").exists()
    ]


def npc_normalized_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    truth = y_true[mask]
    pred = y_pred[mask]
    y_max = max(float(np.max(np.abs(truth))), 1e-8)
    return float(np.mean(((truth - pred) / y_max) ** 2))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"mse_normalized": float("nan")}
    truth = y_true[mask]
    pred = y_pred[mask]
    return {"mse_normalized": npc_normalized_mse(truth, pred)}


def _read_npc_split(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])

    if "CumulativeHumanInfected" in df.columns:
        itot_col = "CumulativeHumanInfected"
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError(f"No numeric cumulative-case column found in {path}")
        itot_col = numeric_cols[0]

    itot = pd.to_numeric(df[itot_col], errors="coerce").to_numpy(dtype=np.float32)
    hi: np.ndarray | None = None
    schema = "country"
    if "HumanInfected" in df.columns:
        hi_series = pd.to_numeric(df["HumanInfected"], errors="coerce")
        hi = hi_series.to_numpy(dtype=np.float32)
        schema = "city"
    return itot, hi, schema


def _has_finite(values: np.ndarray | None) -> bool:
    return values is not None and bool(np.isfinite(values).any())


def _derive_weekly_incidence_from_itot(itot: np.ndarray) -> np.ndarray:
    cumulative = np.asarray(itot, dtype=np.float64)
    incidence = np.full(cumulative.shape, np.nan, dtype=np.float64)
    if cumulative.size == 0:
        return incidence.astype(np.float32)
    if np.isfinite(cumulative[0]):
        incidence[0] = max(cumulative[0], 0.0)
    diffs = np.diff(cumulative)
    finite_diffs = np.isfinite(diffs)
    incidence[1:][finite_diffs] = np.maximum(diffs[finite_diffs], 0.0)
    return incidence.astype(np.float32)


def _resolve_incidence_signal(
    itot: np.ndarray,
    hi: np.ndarray | None,
    *,
    derive_incidence_from_itot: bool,
) -> tuple[np.ndarray | None, str]:
    if _has_finite(hi):
        return hi, "reported"
    if derive_incidence_from_itot:
        return _derive_weekly_incidence_from_itot(itot), "derived_from_cumulative"
    return None, "missing"


def load_real_region(
    region: str,
    data_dir: str | Path | None = None,
    *,
    derive_incidence_from_itot: bool = True,
) -> dict[str, object]:
    if region not in NPC_REGION_DIR:
        known = ", ".join(NPC_REGION_DIR)
        raise ValueError(f"Unknown region={region!r}; expected one of: {known}")

    root = _npc_root(data_dir)
    region_dir = root / NPC_REGION_DIR[region]
    cal_path = region_dir / "data_cali.csv"
    proj_path = region_dir / "data_proj.csv"
    if not cal_path.exists() or not proj_path.exists():
        raise FileNotFoundError(
            f"Missing NPC data for {region!r} under {region_dir}. "
            "Place data_cali.csv and data_proj.csv in that directory."
        )

    itot_cal, hi_cal, schema_cal = _read_npc_split(cal_path)
    itot_proj, hi_proj, schema_proj = _read_npc_split(proj_path)
    schema = "city" if "city" in {schema_cal, schema_proj} else "country"
    inc_cal, inc_cal_source = _resolve_incidence_signal(
        itot_cal,
        hi_cal,
        derive_incidence_from_itot=derive_incidence_from_itot,
    )
    inc_proj, inc_proj_source = _resolve_incidence_signal(
        itot_proj,
        hi_proj,
        derive_incidence_from_itot=derive_incidence_from_itot,
    )

    return {
        "calibration": {
            "Itot": itot_cal,
            "incidence_obs": inc_cal,
            "incidence_source": inc_cal_source,
            "weeks": np.arange(len(itot_cal), dtype=np.float32),
        },
        "projection": {
            "Itot": itot_proj,
            "incidence_obs": inc_proj,
            "incidence_source": inc_proj_source,
            "weeks": np.arange(len(itot_proj), dtype=np.float32),
        },
        "h_total": float("nan"),
        "schema": schema,
    }


NPC_PARAM_ROW_ORDER = (
    "beta_h", "beta_m", "gamma_h", "lambda", "mu_h", "mu_m", "theta_h", "theta_m"
)
NPC_PARAM_ROW_OFFSET = 8

_BOUND_SAFETY_FLOORS = {
    "beta_h": 0.01,
    "beta_m": 0.01,
    "gamma_h": 0.01,
    "lambda": 0.005,
    "mu_m": 0.005,
    "theta_h": 0.01,
    "theta_m": 0.01,
}


def _is_lfs_stub(path: Path) -> bool:
    """Detect git-LFS pointer files (128-ish bytes starting with `version `)."""
    try:
        if path.stat().st_size > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(8) == b"version "
    except OSError:
        return False


def _load_npc_region_bounds_from_csv(
    region: str,
    range7_dir: str | Path,
) -> dict[str, tuple[float, float]] | None:
    """Fallback reader for the Range7 parameter-bound CSV export."""
    if region not in NPC_REGION_DIR:
        return None
    range7_dir = Path(range7_dir)
    here = Path(__file__).resolve().parent
    candidates = [
        range7_dir / "range7_parameter_bounds.csv",
        range7_dir / "data_piml/npc/range7/range7_parameter_bounds.csv",
        here / "data_piml/npc/range7/range7_parameter_bounds.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None

    bounds: dict[str, tuple[float, float]] = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("region") != region or row.get("kind") != "parameter":
                    continue
                name = row.get("project_name") or row.get("npc_name")
                if name not in NPC_PARAM_ROW_ORDER:
                    continue
                lo_txt = row.get("effective_lo") or row.get("lo_raw")
                hi_txt = row.get("effective_hi") or row.get("hi_raw")
                if lo_txt is None or hi_txt is None:
                    return None
                bounds[name] = (float(lo_txt), float(hi_txt))
    except (OSError, ValueError, KeyError):
        return None

    if set(bounds) != set(NPC_PARAM_ROW_ORDER):
        return None
    bounds["mu_h"] = (MU_H_FIXED, MU_H_FIXED)
    return bounds


def load_npc_region_bounds(
    region: str,
    range7_dir: str | Path,
) -> dict[str, tuple[float, float]] | None:
    """Read Range7_<region>.mat and return DINN-keyed parameter bounds.

    Returns None (caller falls back to union) when the file is missing, is an
    LFS pointer stub, and no CSV export is available. Applies
    _BOUND_SAFETY_FLOORS so bounds with NPC lo == 0 don't degenerate the
    tanh reparameterisation. The CSV export is used as a fallback when
    scipy/MAT read support is unavailable.
    """
    if region not in NPC_REGION_DIR:
        return None
    range7_dir = Path(range7_dir)
    candidates = [
        range7_dir / f"Range7_{NPC_REGION_DIR[region]}.mat",
        range7_dir / f"Range7_{region}.mat",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None or _is_lfs_stub(path):
        return _load_npc_region_bounds_from_csv(region, range7_dir)
    try:
        from scipy.io import loadmat
        mat = loadmat(str(path))
    except (ImportError, NotImplementedError, ValueError, OSError):
        return _load_npc_region_bounds_from_csv(region, range7_dir)

    range_arr = None
    for key in ("Range", f"Range7_{NPC_REGION_DIR[region]}", "range"):
        if key in mat:
            range_arr = np.asarray(mat[key])
            break
    if range_arr is None:
        for key, val in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(val)
            if arr.ndim == 2 and arr.shape[0] >= NPC_PARAM_ROW_OFFSET + len(NPC_PARAM_ROW_ORDER):
                range_arr = arr
                break
    if range_arr is None or range_arr.ndim != 2 or range_arr.shape[0] < NPC_PARAM_ROW_OFFSET + len(NPC_PARAM_ROW_ORDER):
        return _load_npc_region_bounds_from_csv(region, range7_dir)

    bounds: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(NPC_PARAM_ROW_ORDER):
        lo = float(range_arr[NPC_PARAM_ROW_OFFSET + i, 0])
        hi = float(range_arr[NPC_PARAM_ROW_OFFSET + i, 1])
        if name in _BOUND_SAFETY_FLOORS:
            lo = max(lo, _BOUND_SAFETY_FLOORS[name])
        if hi <= lo:
            hi = lo + 1e-6
        bounds[name] = (lo, hi)
    bounds["mu_h"] = (MU_H_FIXED, MU_H_FIXED)
    return bounds


def _default_range7_dir() -> Path:
    """Heuristic: workspace root first (where users typically drop the .mat
    files), then nearby NPC clones, then a curated data dir.
    """
    here = Path(__file__).resolve().parent
    for candidate in (
        here,
        here / "../neural-parameter-calibration-dengue/MCMCParameterEstimation/range",
        here.parent / "neural-parameter-calibration-dengue/MCMCParameterEstimation/range",
        here / "data_piml/npc/range",
    ):
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return here


def _resolve_param_bounds(
    region: str,
    config: BaselineConfig,
    *,
    verbose: bool = True,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], str, str]:
    """Pick DINN- and NEC-keyed bounds for `region` per `param_bounds_strategy`.

    Returns (bounds_dinn, bounds_nec, source, source_path) where `source` is one
    of "region", "union (no_range7)", "union (lfs_stub)", "union", and
    `source_path` is the resolved Range7 directory we tried.
    """
    if config.param_bounds_strategy == "union":
        return dict(DINN_PARAM_BOUNDS), dict(NEC_PARAM_BOUNDS), "union", ""
    if config.param_bounds_strategy != "region":
        raise ValueError(
            f"Unknown param_bounds_strategy={config.param_bounds_strategy!r}; "
            "use 'region' or 'union'."
        )
    range7_dir = Path(config.range7_dir) if config.range7_dir else _default_range7_dir()
    region_bounds = load_npc_region_bounds(region, range7_dir)
    if region_bounds is None:
        return (
            dict(DINN_PARAM_BOUNDS),
            dict(NEC_PARAM_BOUNDS),
            "union (no_range7)",
            str(range7_dir),
        )
    bounds_nec = {
        nec_key: region_bounds[dinn_key]
        for nec_key, dinn_key in NEC_TO_DINN_PARAMS.items()
    }
    if verbose:
        lam = region_bounds["lambda"]
        thh = region_bounds["theta_h"]
        bm = region_bounds["beta_m"]
        mum = region_bounds["mu_m"]
        print(
            f"[{region}] Range7 loaded from {range7_dir}: "
            f"lambda=[{lam[0]:.4f}, {lam[1]:.4f}], "
            f"theta_h=[{thh[0]:.4f}, {thh[1]:.4f}], "
            f"beta_m=[{bm[0]:.4f}, {bm[1]:.4f}], "
            f"mu_m=[{mum[0]:.4f}, {mum[1]:.4f}]"
        )
    return region_bounds, bounds_nec, "region", str(range7_dir)


def _attach_shifted_proj_weeks(region_data: dict, cal_proj_gap_weeks: float) -> None:
    """Attach the shifted week axis used for the projection outbreak."""
    cal = region_data["calibration"]
    proj = region_data["projection"]
    cal_len = float(len(cal["weeks"]))
    proj["weeks_shifted"] = (
        np.asarray(proj["weeks"], dtype=np.float32)
        + np.float32(cal_len + float(cal_proj_gap_weeks))
    )


def consistency_loss_a1(
    states_pred: torch.Tensor,
    theta_h: torch.Tensor,
    h_total: float = 1.0,
) -> torch.Tensor:
    """Relative trapezoidal consistency, normalised by h_total."""
    if states_pred.shape[0] < 2:
        return torch.tensor(0.0, device=states_pred.device)
    itot = states_pred[:, 7]
    he = states_pred[:, 2]
    itot_diff = torch.clamp(itot[1:] - itot[:-1], min=0.0)
    he_avg = 0.5 * (he[:-1] + he[1:])
    inc_pred = torch.clamp(theta_h * he_avg, min=0.0)
    denom = max(float(h_total) * 1e-3, 1.0)
    return F.mse_loss(itot_diff / denom, inc_pred / denom)


def mosquito_anchor_loss_a2(
    states_pred: torch.Tensor,
    lam: torch.Tensor,
    mu_m: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    m_total = states_pred[:, 4] + states_pred[:, 5] + states_pred[:, 6]
    m_mean = torch.clamp(m_total.mean(), min=eps)
    m_star = torch.clamp(lam / (mu_m + eps), min=eps)
    return (torch.log(m_mean + eps) - torch.log(m_star + eps)) ** 2


def derived_incidence_loss_b(
    states_pred: torch.Tensor,
    theta_h: torch.Tensor,
    inc_obs: torch.Tensor,
    t_cal: int,
) -> torch.Tensor:
    he_cal = states_pred[:t_cal, 2]
    pred = theta_h * he_cal
    target = inc_obs[:t_cal]
    mask = torch.isfinite(target)
    if int(mask.sum().detach().cpu()) == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean(
        (
            torch.log1p(torch.clamp(pred[mask], min=0.0))
            - torch.log1p(torch.clamp(target[mask], min=0.0))
        )
        ** 2
    )


class GradientBalancer:
    def __init__(self, cfg: BaselineConfig):
        self.alpha = cfg.adaptive_alpha
        self.ema = cfg.adaptive_ema
        self.update_every = cfg.adaptive_update_every
        self.warmup = cfg.adaptive_warmup_epochs
        self.w_min = cfg.adaptive_w_min
        self.w_max = cfg.adaptive_w_max
        self.ratio_ema: float | None = None
        self._floor_streak: int = 0

    def maybe_update(
        self,
        epoch: int,
        w_phys: float,
        loss_data: torch.Tensor,
        loss_phys: torch.Tensor,
        params: Iterable[torch.nn.Parameter],
    ) -> float:
        if epoch < self.warmup or epoch % self.update_every != 0:
            return w_phys
        params = [p for p in params if p.requires_grad]
        if not params:
            return w_phys

        gd = torch.autograd.grad(loss_data, params, retain_graph=True, allow_unused=True)
        gp = torch.autograd.grad(loss_phys, params, retain_graph=True, allow_unused=True)
        device = params[0].device
        nd_sq = torch.zeros((), device=device)
        np_sq = torch.zeros((), device=device)
        for g in gd:
            if g is not None:
                nd_sq = nd_sq + (g.detach() ** 2).sum()
        for g in gp:
            if g is not None:
                np_sq = np_sq + (g.detach() ** 2).sum()
        nd = torch.sqrt(nd_sq + 1e-12)
        np_ = torch.sqrt(np_sq + 1e-12)
        ratio = float((nd / np_).detach().cpu())
        self.ratio_ema = ratio if self.ratio_ema is None else self.ema * self.ratio_ema + (1 - self.ema) * ratio
        w_clipped = float(np.clip(self.alpha * self.ratio_ema, self.w_min, self.w_max))
        if w_clipped <= self.w_min * 1.01:
            self._floor_streak += 1
            if self._floor_streak % 500 == 0:
                print(
                    f"[GradBalancer] w_phys at floor {self.w_min:.2f} "
                    f"for {self._floor_streak} epochs (data/phys grad ratio "
                    f"= {self.ratio_ema:.2e})"
                )
        else:
            self._floor_streak = 0
        return w_clipped


def _param_rows(params: dict[str, float]) -> dict[str, float]:
    return {NEC_TO_DINN_PARAMS.get(k, k): v for k, v in params.items()}


def _inv_softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 1e-6, None)
    return np.where(x > 20.0, x, np.log(np.expm1(x))).astype(np.float32)


def _raw_ic_from_state(state: np.ndarray) -> np.ndarray:
    state_log = np.log1p(np.maximum(np.asarray(state, dtype=np.float64), 0.0))
    return _inv_softplus_np(state_log)


def _first_finite(arr: np.ndarray | None, fallback: float = 0.0) -> float:
    if arr is None:
        return fallback
    a = np.asarray(arr)
    mask = np.isfinite(a)
    return float(a[mask][0]) if mask.any() else fallback


_RANGE7_ROW_FOR_STATE = (3, 4, 5, 1, 2, 6, 7, 0)


def _initial_state_from_range7(
    region: str,
    itot: np.ndarray,
    incidence: np.ndarray | None,
    range7_dir: Path,
) -> np.ndarray:
    """Build the initial state from Range7 medians and observed counts."""
    if region not in NPC_REGION_DIR:
        raise KeyError(f"Unknown region {region!r}; cannot derive Range7 IC.")
    candidates = [
        range7_dir / f"Range7_{NPC_REGION_DIR[region]}.mat",
        range7_dir / f"Range7_{region}.mat",
    ]
    mat_path = next((p for p in candidates if p.exists()), None)
    if mat_path is None or _is_lfs_stub(mat_path):
        raise FileNotFoundError(
            f"Range7 IC source missing or LFS stub for {region!r}; checked {candidates}"
        )
    try:
        from scipy.io import loadmat
        mat = loadmat(str(mat_path))
    except (ImportError, NotImplementedError, ValueError, OSError) as exc:
        raise RuntimeError(f"Failed to read {mat_path}: {exc}") from exc

    arr = None
    for k in ("Range", "range"):
        if k in mat:
            arr = np.asarray(mat[k], dtype=np.float64)
            break
    if arr is None:
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            a = np.asarray(v)
            if a.ndim == 2 and a.shape[0] >= 16 and a.shape[1] >= 2:
                arr = a.astype(np.float64)
                break
    if arr is None or arr.ndim != 2 or arr.shape[0] < 16 or arr.shape[1] < 2:
        raise ValueError(f"Could not find a 16x2 Range array in {mat_path}")

    ic_log1p_median = 0.5 * (arr[:8, 0] + arr[:8, 1])
    ic_log1p_state = np.array(
        [ic_log1p_median[i] for i in _RANGE7_ROW_FOR_STATE],
        dtype=np.float64,
    )
    ic_linear = np.expm1(np.maximum(ic_log1p_state, 0.0))

    itot0_obs = max(_first_finite(itot, fallback=0.0), 0.0)
    if itot0_obs > 0.0:
        ic_linear[7] = itot0_obs

    inc0 = _first_finite(incidence, fallback=0.0)
    if inc0 > 0.0:
        hi0_max_range7 = float(np.expm1(max(arr[1, 1], 0.0)))
        ic_linear[3] = float(np.clip(inc0, 1.0, max(hi0_max_range7, 1.0)))

    if ic_linear[3] > 2.0 * max(ic_linear[7], 1.0):
        capped_hi = max(float(ic_linear[7]), 1.0)
        print(
            f"[Range7 IC audit] {region}: Hi0={ic_linear[3]:.1f} > 2*Itot0="
            f"{ic_linear[7]:.1f}. Capping Hi0 -> {capped_hi:.1f}."
        )
        ic_linear[3] = capped_hi

    non_itot_max = float(np.log1p(np.max(ic_linear[:7])))
    if non_itot_max > 5.0:
        raise ValueError(
            f"[{region}] IC scale check failed: max log1p(state[:7])={non_itot_max:.2f} > 5.0 "
            f"(state={ic_linear.tolist()}). Range7 IC expected log1p < 5 outside Itot."
        )
    itot0_log1p = float(np.log1p(ic_linear[7]))
    if itot0_log1p > 13.0:
        raise ValueError(
            f"[{region}] Itot0 scale check failed: log1p(Itot0)={itot0_log1p:.2f} > 13 "
            f"(value={ic_linear[7]:.1f})."
        )

    return ic_linear.astype(np.float32)


def _dengue_log_residuals(
    t: torch.Tensor,
    state_log: torch.Tensor,
    states: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    hr, hs, he, hi, me, ms, mi, _itot = [states[:, i] for i in range(8)]
    h = torch.clamp(hr + hs + he + hi, min=1e-8)
    m = torch.clamp(me + ms + mi, min=1e-8)

    lam = params["lambda"]
    beta_m = params["beta_m"]
    mu_m = params["mu_m"]
    theta_m = params["theta_m"]
    mu_h = params["mu_h"]
    beta_h = params["beta_h"]
    theta_h = params["theta_h"]
    gamma_h = params["gamma_h"]

    dlog_dt = [
        grad(state_log[:, i].sum(), t, retain_graph=True, create_graph=True)[0]
        for i in range(8)
    ]
    rhs = [
        gamma_h * hi - mu_h * hr,
        mu_h * h - beta_h * mi * hs / m - mu_h * hs,
        beta_h * mi * hs / m - (theta_h + mu_h) * he,
        theta_h * he - (gamma_h + mu_h) * hi,
        beta_m * hi * ms / h - (theta_m + mu_m) * me,
        lam - beta_m * hi * ms / h - mu_m * ms,
        theta_m * me - mu_m * mi,
        theta_h * he,
    ]
    return [dlog_dt[i] - rhs[i] / (1.0 + states[:, i]) for i in range(8)]


def _residual_loss(
    residuals: Iterable[torch.Tensor],
    states: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = sum(torch.mean(r ** 2) for r in residuals)
    if states is not None:
        denom = 1.0 + states.detach().pow(2).mean()
        loss = loss / denom
    return loss


class DINNBaseline(_BaseDINNBaseline):
    def __init__(self, *args, param_bounds_dinn: dict[str, tuple[float, float]] | None = None, **kwargs):
        if param_bounds_dinn is not None:
            _saved = dict(DINN_PARAM_BOUNDS)
            DINN_PARAM_BOUNDS.clear()
            DINN_PARAM_BOUNDS.update(param_bounds_dinn)
            try:
                super().__init__(*args, **kwargs)
            finally:
                DINN_PARAM_BOUNDS.clear()
                DINN_PARAM_BOUNDS.update(_saved)
            self._param_bounds_dinn = dict(param_bounds_dinn)
        else:
            super().__init__(*args, **kwargs)
            self._param_bounds_dinn = dict(DINN_PARAM_BOUNDS)
        self.mu_h_tilda.requires_grad_(False)
        self.ic_head_cal = nn.Parameter(torch.zeros(8, dtype=torch.float32, device=self.device))
        self.ic_head_proj = nn.Parameter(torch.zeros(8, dtype=torch.float32, device=self.device))
        self.register_buffer(
            "ic_head_warmstart_cal",
            torch.zeros(8, dtype=torch.float32, device=self.device),
        )
        self.register_buffer(
            "ic_head_warmstart_proj",
            torch.zeros(8, dtype=torch.float32, device=self.device),
        )
        self.loss_component_hist: list[dict[str, float]] = []
        self.w_phys_hist: list[float] = []

    @property
    def lambda_(self) -> torch.Tensor:
        return tanh_bound(self.lambda_tilda, *self._param_bounds_dinn["lambda"])

    @property
    def beta_m(self) -> torch.Tensor:
        return tanh_bound(self.beta_m_tilda, *self._param_bounds_dinn["beta_m"])

    @property
    def mu_m(self) -> torch.Tensor:
        return tanh_bound(self.mu_m_tilda, *self._param_bounds_dinn["mu_m"])

    @property
    def theta_m(self) -> torch.Tensor:
        return tanh_bound(self.theta_m_tilda, *self._param_bounds_dinn["theta_m"])

    @property
    def beta_h(self) -> torch.Tensor:
        return tanh_bound(self.beta_h_tilda, *self._param_bounds_dinn["beta_h"])

    @property
    def theta_h(self) -> torch.Tensor:
        return tanh_bound(self.theta_h_tilda, *self._param_bounds_dinn["theta_h"])

    @property
    def gamma_h(self) -> torch.Tensor:
        return tanh_bound(self.gamma_h_tilda, *self._param_bounds_dinn["gamma_h"])

    @property
    def mu_h(self) -> torch.Tensor:
        return torch.tensor(MU_H_FIXED, dtype=torch.float32, device=self.device)

    def set_ic_heads(self, cal_state: np.ndarray, proj_state: np.ndarray) -> None:
        cal_raw = torch.tensor(_raw_ic_from_state(cal_state), device=self.device)
        proj_raw = torch.tensor(_raw_ic_from_state(proj_state), device=self.device)
        with torch.no_grad():
            self.ic_head_cal.copy_(cal_raw)
            self.ic_head_proj.copy_(proj_raw)
            self.ic_head_warmstart_cal.copy_(cal_raw)
            self.ic_head_warmstart_proj.copy_(proj_raw)

    def current_params_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "lambda": self.lambda_.squeeze(),
            "beta_m": self.beta_m.squeeze(),
            "mu_m": self.mu_m.squeeze(),
            "theta_m": self.theta_m.squeeze(),
            "mu_h": self.mu_h.squeeze(),
            "beta_h": self.beta_h.squeeze(),
            "theta_h": self.theta_h.squeeze(),
            "gamma_h": self.gamma_h.squeeze(),
        }

    def forward_log_states_with_ic(
        self,
        t: torch.Tensor,
        ic_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_log = self.net_dengue(t.reshape(-1, 1))
        delta = base_log - base_log[:1]
        state_log = F.softplus(ic_head.reshape(1, -1) + delta)
        states = torch.expm1(state_log)
        return state_log, states

    def net_f_with_ic(
        self,
        t: torch.Tensor,
        ic_head: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        params = self.current_params_tensors()
        state_log, states = self.forward_log_states_with_ic(t, ic_head)
        residuals = _dengue_log_residuals(t, state_log, states, params)
        return residuals, state_log, states, params

    def trunk_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.net_dengue.parameters())

    def configure_optimizer(self, config: BaselineConfig) -> None:
        self.optimizer = optim.Adam(
            [
                {"params": self.net_params, "weight_decay": config.weight_decay},
                {"params": self.phys_params, "weight_decay": 0.0},
                {"params": [self.ic_head_cal, self.ic_head_proj], "weight_decay": 0.0},
            ],
            lr=config.lr,
            betas=(0.9, 0.999),
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )

    def predict_states_with_ic(self, t_values: np.ndarray, ic_head: torch.Tensor) -> np.ndarray:
        self.eval()
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _state_log, states = self.forward_log_states_with_ic(t_tensor, ic_head)
        return states.detach().cpu().numpy()


class NECBaseline(_BaseNECBaseline):
    def __init__(self, *args, param_bounds_nec: dict[str, tuple[float, float]] | None = None, **kwargs):
        if param_bounds_nec is not None:
            _saved = dict(NEC_PARAM_BOUNDS)
            NEC_PARAM_BOUNDS.clear()
            NEC_PARAM_BOUNDS.update(param_bounds_nec)
            try:
                super().__init__(*args, **kwargs)
            finally:
                NEC_PARAM_BOUNDS.clear()
                NEC_PARAM_BOUNDS.update(_saved)
            self._param_bounds_nec = dict(param_bounds_nec)
        else:
            super().__init__(*args, **kwargs)
            self._param_bounds_nec = dict(NEC_PARAM_BOUNDS)
        self.ic_head_cal = nn.Parameter(torch.zeros(8, dtype=torch.float32, device=self.device))
        self.ic_head_proj = nn.Parameter(torch.zeros(8, dtype=torch.float32, device=self.device))
        self.register_buffer(
            "ic_head_warmstart_cal",
            torch.zeros(8, dtype=torch.float32, device=self.device),
        )
        self.register_buffer(
            "ic_head_warmstart_proj",
            torch.zeros(8, dtype=torch.float32, device=self.device),
        )
        self.loss_component_hist: list[dict[str, float]] = []
        self.w_phys_hist: list[float] = []

    def set_ic_heads(self, cal_state: np.ndarray, proj_state: np.ndarray) -> None:
        cal_raw = torch.tensor(_raw_ic_from_state(cal_state), device=self.device)
        proj_raw = torch.tensor(_raw_ic_from_state(proj_state), device=self.device)
        with torch.no_grad():
            self.ic_head_cal.copy_(cal_raw)
            self.ic_head_proj.copy_(proj_raw)
            self.ic_head_warmstart_cal.copy_(cal_raw)
            self.ic_head_warmstart_proj.copy_(proj_raw)

    def current_params_tensors(self) -> dict[str, torch.Tensor]:
        calib_input = torch.zeros(1, 10, device=self.device)
        params = self.calib(calib_input)
        return {
            "lambda": params["lambda"].squeeze(),
            "beta_m": params["betam"].squeeze(),
            "mu_m": params["mum"].squeeze(),
            "theta_m": params["thetam"].squeeze(),
            "mu_h": torch.tensor(MU_H_FIXED, dtype=torch.float32, device=self.device),
            "beta_h": params["betah"].squeeze(),
            "theta_h": params["thetah"].squeeze(),
            "gamma_h": params["gammah"].squeeze(),
        }

    def current_params_dict(self) -> dict[str, float]:
        params = super().current_params_dict()
        params["muh"] = float(MU_H_FIXED)
        return params

    def forward_log_states_with_ic(
        self,
        t: torch.Tensor,
        ic_head: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_log = self.emu(t.reshape(-1, 1))
        delta = base_log - base_log[:1]
        state_log = F.softplus(ic_head.reshape(1, -1) + delta)
        states = torch.expm1(state_log)
        return state_log, states

    def net_f_with_ic(
        self,
        t: torch.Tensor,
        ic_head: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        params = self.current_params_tensors()
        state_log, states = self.forward_log_states_with_ic(t, ic_head)
        residuals = _dengue_log_residuals(t, state_log, states, params)
        return residuals, state_log, states, params

    def trunk_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.emu.parameters())

    def configure_optimizer(self, config: BaselineConfig) -> None:
        self.optimizer = optim.Adam(
            [
                {"params": list(self.emu.parameters()), "weight_decay": config.weight_decay},
                {"params": list(self.calib.parameters()), "weight_decay": 0.0},
                {"params": [self.ic_head_cal, self.ic_head_proj], "weight_decay": 0.0},
            ],
            lr=config.lr,
            betas=(0.9, 0.999),
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )

    def predict_states_with_ic(self, t_values: np.ndarray, ic_head: torch.Tensor) -> np.ndarray:
        self.eval()
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _state_log, states = self.forward_log_states_with_ic(t_tensor, ic_head)
        return states.detach().cpu().numpy()

def fit_real_npc(
    model: DINNBaseline | NECBaseline,
    region_data: dict[str, object],
    config: BaselineConfig,
    *,
    verbose: bool = True,
    region: str = "?",
    model_type: str | None = None,
) -> dict[str, float]:
    if config.protocol == "single_window_8020":
        raise NotImplementedError(
            "cfg.protocol='single_window_8020' is reserved but not implemented."
        )
    if config.protocol != "two_window":
        raise ValueError("Only cfg.protocol='two_window' is implemented.")
    if config.projection_ic_policy.upper() != "A":
        raise ValueError("Only projection_ic_policy='A' is implemented.")

    cal = region_data["calibration"]
    proj = region_data["projection"]
    range7_dir = (
        Path(config.range7_dir) if config.range7_dir is not None else _default_range7_dir()
    )
    cal_state0 = _initial_state_from_range7(
        region,
        cal["Itot"],
        cal["incidence_obs"],
        range7_dir,
    )
    proj_state0 = _initial_state_from_range7(
        region,
        proj["Itot"],
        proj["incidence_obs"],
        range7_dir,
    )
    if verbose:
        print(
            f"[{region}] Range7 IC (cal Itot0={cal_state0[7]:.1f}, "
            f"Hs0={cal_state0[1]:.2f}, He0={cal_state0[2]:.2f}, "
            f"Mi0={cal_state0[6]:.2f}; max log1p={float(np.log1p(np.max(cal_state0))):.2f})"
    )
    model.set_ic_heads(cal_state0, proj_state0)
    model.configure_optimizer(config)

    _attach_shifted_proj_weeks(region_data, config.cal_proj_gap_weeks)
    proj_weeks_shifted = region_data["projection"]["weeks_shifted"]

    device = model.device
    t_cal = torch.tensor(cal["weeks"], dtype=torch.float32, requires_grad=True, device=device)
    t_proj = torch.tensor(proj_weeks_shifted, dtype=torch.float32, requires_grad=True, device=device)
    itot_cal = torch.tensor(cal["Itot"], dtype=torch.float32, device=device)
    inc_cal = (
        torch.tensor(cal["incidence_obs"], dtype=torch.float32, device=device)
        if cal["incidence_obs"] is not None
        else None
    )
    t_cal_len = int(t_cal.numel())

    w_phys = float(config.w_phys_real)
    balancer = GradientBalancer(config)
    if model_type is None:
        resolved_model_type = "dinn" if isinstance(model, DINNBaseline) else "nec"
    else:
        resolved_model_type = model_type
    itot_cal_first = float(itot_cal[0].detach().cpu()) if itot_cal.numel() else 0.0
    itot_cal_final = float(itot_cal[-1].detach().cpu()) if itot_cal.numel() else 1.0
    last_good_state: dict[str, torch.Tensor] | None = None
    nan_recoveries = 0
    nan_recovery_max = 3
    current_lr = float(config.lr)
    start = time.perf_counter()
    last: dict[str, float] = {}
    for epoch in range(config.epochs):
        model.optimizer.zero_grad()

        residuals_cal, state_log_cal, states_cal, params_cal = model.net_f_with_ic(
            t_cal,
            model.ic_head_cal,
        )
        residuals_proj, state_log_proj, states_proj, _params_proj = model.net_f_with_ic(
            t_proj,
            model.ic_head_proj,
        )

        if config.itot_loss_mode == "log_mse":
            loss_itot = F.mse_loss(
                state_log_cal[:, 7],
                torch.log1p(torch.clamp(itot_cal, min=0.0)),
            )
        else:
            loss_itot = F.mse_loss(states_cal[:, 7], torch.clamp(itot_cal, min=0.0))
        loss_inc = torch.tensor(0.0, device=device)
        inc_active = bool(
            config.use_inc_soft
            and inc_cal is not None
            and bool(torch.isfinite(inc_cal).any().detach().cpu())
        )
        if inc_active:
            loss_inc = derived_incidence_loss_b(
                states_cal,
                params_cal["theta_h"],
                inc_cal,
                t_cal_len,
            )
        itot_peak = float(torch.clamp(itot_cal, min=0.0).max().item())
        denom_peak = max(itot_peak, 1.0) ** 2
        loss_itot_linear = F.mse_loss(
            states_cal[:, 7], torch.clamp(itot_cal, min=0.0)
        ) / denom_peak
        loss_data = (
            config.w_itot * loss_itot
            + config.w_itot_linear * loss_itot_linear
            + config.w_inc_soft * loss_inc
        )

        loss_residual_cal = _residual_loss(residuals_cal, states_cal)
        loss_residual_proj = _residual_loss(residuals_proj, states_proj)
        loss_residual = loss_residual_cal + loss_residual_proj
        loss_consistency = consistency_loss_a1(
            states_cal, params_cal["theta_h"], h_total=itot_cal_final
        )
        loss_anchor = mosquito_anchor_loss_a2(
            states_cal,
            params_cal["lambda"],
            params_cal["mu_m"],
        )
        loss_phys = (
            loss_residual
            + config.w_consistency * loss_consistency
            + config.w_anchor * loss_anchor
        )
        loss_ic_prior = torch.mean(
            (model.ic_head_cal - model.ic_head_warmstart_cal) ** 2
        ) + torch.mean((model.ic_head_proj - model.ic_head_warmstart_proj) ** 2)

        denom_t0 = max(itot_cal_first, 1.0) ** 2
        loss_itot_t0 = ((states_cal[0, 7] - itot_cal[0]) ** 2) / denom_t0

        if config.use_horizon_cap and itot_cal_final > 0:
            cap = config.horizon_cap_ratio * itot_cal_final
            violation = F.relu(states_proj[-1, 7] - cap)
            loss_horizon = (violation / max(itot_cal_final, 1.0)) ** 2
        else:
            loss_horizon = torch.tensor(0.0, device=device)

        if config.use_adaptive_phys:
            w_phys = balancer.maybe_update(
                epoch,
                w_phys,
                loss_data,
                loss_phys,
                model.trunk_parameters(),
            )

        loss_continuity = F.mse_loss(
            state_log_proj[0, :], state_log_cal[-1, :].detach()
        )

        loss = (
            loss_data
            + w_phys * loss_phys
            + config.w_ic_prior * loss_ic_prior
            + config.w_itot_t0_anchor * loss_itot_t0
            + config.w_horizon * loss_horizon
            + config.proj_ic_continuity_w * loss_continuity
        )
        if not torch.isfinite(loss):
            nan_recoveries += 1
            print(
                f"[{region}/{resolved_model_type}] NaN/Inf total_loss at "
                f"ep {epoch}, recovery #{nan_recoveries}"
            )
            if nan_recoveries > nan_recovery_max or last_good_state is None:
                print(
                    f"[{region}/{resolved_model_type}] ABORT: "
                    f"{nan_recoveries} NaN recoveries exhausted at ep {epoch}"
                )
                break
            model.load_state_dict(last_good_state, strict=False)
            current_lr *= 0.1
            model.configure_optimizer(replace(config, lr=current_lr))
            continue
        loss.backward()
        model.optimizer.step()
        model.scheduler.step(loss.detach())
        if epoch % 500 == 0 and torch.isfinite(loss):
            last_good_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

        if epoch == WARMUP_PROJ_IC_EPOCHS:
            cal_converged = float(loss_itot.detach().cpu()) < 0.1
            if cal_converged:
                with torch.no_grad():
                    end_cal_log = state_log_cal[-1, :].detach().clone()
                    end_cal_raw = torch.log(
                        torch.expm1(end_cal_log).clamp(min=1e-6) + 1e-6
                    )
                    model.ic_head_proj.copy_(end_cal_raw)
                    model.ic_head_warmstart_proj.copy_(end_cal_raw)
                if verbose:
                    print(
                        f"[{region}/{resolved_model_type}] ic_head_proj rewarmstarted "
                        f"from end-of-cal at ep {epoch} "
                        f"(loss_itot={float(loss_itot):.3e})"
                    )
            elif verbose:
                print(
                    f"[{region}/{resolved_model_type}] SKIP warmstart proj IC at "
                    f"ep {epoch}: calibration has not converged "
                    f"(loss_itot={float(loss_itot):.3e})"
                )

        last = {
            "loss_total": float(loss.detach().cpu()),
            "loss_data": float(loss_data.detach().cpu()),
            "loss_itot": float(loss_itot.detach().cpu()),
            "loss_inc": float(loss_inc.detach().cpu()),
            "loss_residual": float(loss_residual.detach().cpu()),
            "loss_phys": float(loss_phys.detach().cpu()),
            "loss_residual_cal": float(loss_residual_cal.detach().cpu()),
            "loss_residual_proj": float(loss_residual_proj.detach().cpu()),
            "loss_consistency": float(loss_consistency.detach().cpu()),
            "loss_anchor": float(loss_anchor.detach().cpu()),
            "loss_ic_prior": float(loss_ic_prior.detach().cpu()),
            "loss_itot_t0": float(loss_itot_t0.detach().cpu()),
            "loss_horizon": float(loss_horizon.detach().cpu()) if isinstance(loss_horizon, torch.Tensor) else 0.0,
            "loss_continuity": float(loss_continuity.detach().cpu()),
            "w_phys": float(w_phys),
            "inc_soft_active": float(inc_active),
        }
        model.losses.append(last["loss_total"])
        model.loss_data_hist.append(last["loss_data"])
        model.loss_phys_hist.append(last["loss_phys"])
        model.loss_component_hist.append(last)
        model.w_phys_hist.append(float(w_phys))

        if verbose and epoch % 200 == 0:
            print(
                f"ep {epoch:5d} total={last['loss_total']:.3e} "
                f"itot={last['loss_itot']:.3e} phys={last['loss_residual']:.3e} "
                f"a1={last['loss_consistency']:.3e} a2={last['loss_anchor']:.3e} "
                f"w_phys={w_phys:.3e}"
            )

    last["train_seconds"] = time.perf_counter() - start
    return last


def evaluate_npc_style(
    model: DINNBaseline | NECBaseline,
    region_data: dict[str, object],
    *,
    region: str,
    model_type: str,
) -> list[dict[str, object]]:
    params = _param_rows(model.current_params_dict())
    theta_h = params["theta_h"]
    split_specs = [
        ("calibration", model.ic_head_cal, region_data["calibration"]),
        ("projection", model.ic_head_proj, region_data["projection"]),
    ]
    rows: list[dict[str, object]] = []
    for split, ic_head, split_data in split_specs:
        forward_weeks = split_data.get("weeks_shifted", split_data["weeks"])
        states = np.maximum(model.predict_states_with_ic(forward_weeks, ic_head), 0.0)
        true_states = {state: np.full(states.shape[0], np.nan, dtype=np.float64) for state in STATE_COLS}
        true_states["Itot"] = np.asarray(split_data["Itot"], dtype=np.float64)

        for idx, state in enumerate(STATE_COLS):
            metrics = metric_dict(true_states[state], states[:, idx])
            rows.append(
                {
                    "region": region,
                    "region_label": REGION_LABELS.get(region, region),
                    "model": model_type,
                    "split": split,
                    "target": state,
                    "state": state,
                    "Hi_status": "not_observed",
                    "incidence_source": split_data["incidence_source"],
                    **metrics,
                }
            )

        infectious_case_true = (
            np.asarray(split_data["incidence_obs"], dtype=np.float64)
            if split_data["incidence_obs"] is not None
            else np.full(states.shape[0], np.nan, dtype=np.float64)
        )
        infectious_case_pred = theta_h * states[:, 2]
        rows.append(
            {
                "region": region,
                "region_label": REGION_LABELS.get(region, region),
                "model": model_type,
                "split": split,
                "target": "infectious_case",
                "state": "infectious_case",
                "Hi_status": "not_observed",
                "incidence_source": split_data["incidence_source"],
                **metric_dict(infectious_case_true, infectious_case_pred),
            }
        )
    return rows


def prediction_rows_for_real_model(
    model: DINNBaseline | NECBaseline,
    region_data: dict[str, object],
    *,
    region: str,
    model_type: str,
) -> list[dict[str, object]]:
    params = _param_rows(model.current_params_dict())
    theta_h = params["theta_h"]
    rows: list[dict[str, object]] = []
    for split, ic_head, split_data in [
        ("calibration", model.ic_head_cal, region_data["calibration"]),
        ("projection", model.ic_head_proj, region_data["projection"]),
    ]:
        forward_weeks = split_data.get("weeks_shifted", split_data["weeks"])
        states = np.maximum(model.predict_states_with_ic(forward_weeks, ic_head), 0.0)
        series = [
            ("Itot", split_data["Itot"], states[:, 7]),
            (
                "infectious_case",
                split_data["incidence_obs"]
                if split_data["incidence_obs"] is not None
                else np.full(states.shape[0], np.nan, dtype=np.float64),
                theta_h * states[:, 2],
            ),
        ]
        for target, observed, predicted in series:
            for t, obs, pred in zip(split_data["weeks"], observed, predicted):
                rows.append(
                    {
                        "region": region,
                        "region_label": REGION_LABELS.get(region, region),
                        "model": model_type,
                        "split": split,
                        "target": target,
                        "t": float(t),
                        "observed": float(obs) if np.isfinite(obs) else np.nan,
                        "predicted": float(pred),
                        "Hi_status": "not_observed",
                        "incidence_source": split_data["incidence_source"],
                    }
                )
    return rows


def train_real_model(
    region: str,
    model_type: str,
    config: BaselineConfig | None = None,
    *,
    data_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[DINNBaseline | NECBaseline, list[dict[str, object]], dict[str, float]]:
    config = config or BaselineConfig()
    data = load_real_region(
        region,
        data_dir,
        derive_incidence_from_itot=config.derive_incidence_from_itot,
    )
    cal = data["calibration"]

    bounds_dinn, bounds_nec, bounds_source, bounds_path = _resolve_param_bounds(
        region, config, verbose=verbose
    )
    if verbose and bounds_source != "region":
        print(
            f"[{REGION_LABELS.get(region, region)}] param_bounds_strategy="
            f"{config.param_bounds_strategy!r} -> using {bounds_source} "
            f"(searched {bounds_path})"
        )

    set_seed(config.seed)
    if model_type == "dinn":
        model = DINNBaseline(
            cal["weeks"],
            cal["Itot"],
            cal["incidence_obs"],
            full_state=False,
            device=config.device,
            param_bounds_dinn=bounds_dinn,
        )
    elif model_type == "nec":
        model = NECBaseline(
            cal["weeks"],
            cal["Itot"],
            cal["incidence_obs"],
            full_state=False,
            device=config.device,
            param_bounds_nec=bounds_nec,
        )
    else:
        raise ValueError(f"Unknown model_type={model_type!r}; use 'dinn' or 'nec'.")

    h_total_value = float("nan")

    if verbose:
        label = REGION_LABELS.get(region, region)
        print(f"[{label}] training {model_type.upper()} baseline")
    summary = fit_real_npc(
        model, data, config,
        verbose=verbose, region=region, model_type=model_type,
    )
    summary.update(
        {
            "protocol": config.protocol,
            "h_total_source": "not_used",
            "h_total_value": h_total_value,
        }
    )
    metric_rows = evaluate_npc_style(
        model,
        data,
        region=region,
        model_type=model_type,
    )
    params = _param_rows(model.current_params_dict())
    bound_columns: dict[str, float] = {}
    for name, (lo, hi) in bounds_dinn.items():
        bound_columns[f"param_bound_lo_{name}"] = float(lo)
        bound_columns[f"param_bound_hi_{name}"] = float(hi)
    for row in metric_rows:
        row.update(
            {
                "schema": data["schema"],
                "epochs": config.epochs,
                "protocol": config.protocol,
                "w_itot": config.w_itot,
                "w_phys_real": config.w_phys_real,
                "w_consistency": config.w_consistency,
                "w_anchor": config.w_anchor,
                "w_inc_soft": config.w_inc_soft,
                "use_adaptive_phys": config.use_adaptive_phys,
                "use_inc_soft": config.use_inc_soft,
                "projection_ic_policy": config.projection_ic_policy,
                "derive_incidence_from_itot": config.derive_incidence_from_itot,
                "incidence_source_calibration": data["calibration"]["incidence_source"],
                "incidence_source_projection": data["projection"]["incidence_source"],
                "h_total_source": "not_used",
                "h_total_value": h_total_value,
                "param_bounds_source": bounds_source,
                "param_bounds_path": bounds_path,
                "param_bounds_strategy": config.param_bounds_strategy,
                "itot_loss_mode": config.itot_loss_mode,
                "use_horizon_cap": config.use_horizon_cap,
                "cal_proj_gap_weeks": config.cal_proj_gap_weeks,
                "train_seconds": summary["train_seconds"],
                "final_loss": summary["loss_total"],
                **bound_columns,
                **summary,
                **params,
            }
        )
    return model, metric_rows, summary


def run_real_experiment(
    region: str,
    model_type: str,
    config: BaselineConfig | None = None,
    *,
    data_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    config = config or BaselineConfig()
    model, metric_rows, summary = train_real_model(
        region,
        model_type,
        config,
        data_dir=data_dir,
        verbose=verbose,
    )
    data = load_real_region(
        region,
        data_dir,
        derive_incidence_from_itot=config.derive_incidence_from_itot,
    )
    _attach_shifted_proj_weeks(data, config.cal_proj_gap_weeks)
    result: dict[str, object] = {
        **summary,
        "model": model,
        "params_final": _param_rows(model.current_params_dict()),
        "metrics": pd.DataFrame(metric_rows),
        "predictions": pd.DataFrame(
            prediction_rows_for_real_model(
                model,
                data,
                region=region,
                model_type=model_type,
            )
        ),
    }
    return result


def run_all_baseline_table(
    regions: Iterable[str] | None = None,
    model_types: Iterable[str] = ("dinn", "nec"),
    config: BaselineConfig | None = None,
    *,
    data_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    prediction_csv: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    config = config or BaselineConfig()
    region_list = list(regions) if regions is not None else discover_regions(data_dir)
    rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for region in region_list:
        for model_type in model_types:
            model, metric_rows, _summary = train_real_model(
                region,
                model_type,
                config,
                data_dir=data_dir,
                verbose=verbose,
            )
            rows.extend(metric_rows)
            if prediction_csv is not None:
                region_data = load_real_region(
                    region,
                    data_dir,
                    derive_incidence_from_itot=config.derive_incidence_from_itot,
                )
                _attach_shifted_proj_weeks(region_data, config.cal_proj_gap_weeks)
                prediction_rows.extend(
                    prediction_rows_for_real_model(
                        model,
                        region_data,
                        region=region,
                        model_type=model_type,
                    )
                )
    full_table = pd.DataFrame(rows)
    table = (
        full_table[
            full_table["split"].eq("projection")
            & full_table["target"].eq("Itot")
        ][["region", "model", "mse_normalized", "train_seconds"]]
        .sort_values(["region", "model"])
        .reset_index(drop=True)
    )
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        if verbose:
            print(f"wrote results table -> {output_path}")
    if prediction_csv is not None:
        prediction_path = Path(prediction_csv)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
        if verbose:
            print(f"wrote prediction table -> {prediction_path}")
    return table

# DINN/NEC Parameter Trace + Marginal Posterior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save the 7 trainable ODE parameter values every 10 epochs for every (region, model, seed) run of DINN/NEC training, write them to per-seed CSVs mirroring the existing results/predictions layout, and extend the existing NPC/MCMC marginal-posterior density plot in `visualisation.ipynb` into a 4-way MCMC/NPC/DINN/NEC overlay.

**Architecture:** `train_one()` already computes the parameter dict every epoch for the physics loss; snapshot it into a `params_trace` list every `param_log_every` epochs and thread it through the existing run -> aggregate -> write pipeline in `baseline_experiments.py`, writing CSVs at `outputs/dinns_nec_params/<region>/seed<N>/<model>.csv`. `visualisation.ipynb`'s `plot_density` function is generalized from a hardcoded 2-series function to an N-series overlay, and the city/country marginal-posterior cells load the new CSVs (final-epoch row per seed) alongside the existing NPC/MCMC loaders.

**Tech Stack:** Python, PyTorch, pandas, matplotlib, scipy, h5py, Jupyter notebooks (`nbformat`/`nbclient` available under the `denguetwin` conda env for notebook-cell verification).

## Global Constraints

- Full source of truth for parameter names: `PARAM_NAMES = ("lambda", "beta_m", "mu_m", "theta_m", "beta_h", "theta_h", "gamma_h")` in `model_common.py` — do not reorder or rename.
- `mu_h` is fixed (not learned) and must NOT appear in the parameter trace CSV.
- New CSV path convention must mirror the existing one exactly: `<dir>/<region>/seed<N>/<model>.csv` (see `outputs/dinns_nec_results/bello/seed0/dinn.csv` for the existing pattern).
- No pytest (or any test framework) is installed in any project conda env (checked: `pinns`, `denguetwin`, `learning`, base). Verification steps in this plan use plain `assert`-based scripts run via `conda run -n <env> python ...`, matching this repo's existing convention of inline smoke-test code (see the "Smoke test" cell in `baseline_all_results.ipynb`). Do not add a `tests/` directory or a new test-framework dependency.
- Use the `pinns` conda env (has working `torch`, `pandas`, `numpy`, no `h5py`) for anything importing `baseline_experiments`/`dinn_model`/`nec_model`.
- Use the `denguetwin` conda env (has `torch`, `pandas`, `h5py`, `scipy`, `matplotlib`, `nbformat`, `nbclient`) for anything touching `FigureGeneration/visualisation.ipynb`.
- `visualisation.ipynb` is too large for the standard Read+NotebookEdit tool path (34k+ tokens, exceeds the 25000-token Read limit). Edit it via a Python script that does `json.load` -> locate the target cell by a unique substring in its joined `source` -> replace `cell["source"]` with the new lines (each line except the last ending in `"\n"`) -> `json.dump(nb, f, indent=1)` followed by a trailing `"\n"`. This round-trips byte-identical for untouched cells (verified: loading and re-dumping the file unchanged reproduces it exactly). Do NOT use the `Edit` tool on `.ipynb` files (it refuses) and do NOT attempt `Read` on `visualisation.ipynb` (it will fail on size).
- `baseline_all_results.ipynb` is small (10 cells, ~65KB) — use the normal `Read` then `NotebookEdit` tool path for it.
- Never execute the real full sweep (6 regions x 2 models x 10 seeds x 2000 epochs) as part of this plan — all verification uses tiny epoch counts (≤25) and 1-2 regions/seeds. The user runs the full sweep separately afterward.

---

### Task 1: Capture a per-epoch parameter trace in `train_one()`

**Files:**
- Modify: `baseline_experiments.py:132-142` (`BaselineConfig` dataclass)
- Modify: `baseline_experiments.py:191-261` (`train_one`)

**Interfaces:**
- Consumes: `model_common.PARAM_NAMES` (already imported at the top of `baseline_experiments.py`); the `params: dict[str, torch.Tensor]` dict already computed each epoch by `_forward(model, t_cal, features_cal)`.
- Produces: `train_one(...)` return dict gains a new key `"params_trace": list[dict]`, where each dict is `{"epoch": int, "lambda": float, "beta_m": float, "mu_m": float, "theta_m": float, "beta_h": float, "theta_h": float, "gamma_h": float}`. Since `run_real_experiment()` already does `**summary` when building its return dict, `params_trace` automatically appears in `run_real_experiment(...)`'s return value and in every `run` dict inside `agg["runs"]` from `_aggregate_runs()` — no changes needed in either of those two functions for this task.

- [ ] **Step 1: Write the failing verification script**

Create `/tmp/check_task1.py`:

```python
import sys
sys.path.insert(0, "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns")
from baseline_experiments import BaselineConfig, run_real_experiment, PARAM_NAMES

cfg = BaselineConfig(epochs=25, param_log_every=10, log_every=100)
result = run_real_experiment("iquitos", "dinn", cfg, seed=0, verbose=False)

trace = result["params_trace"]
assert isinstance(trace, list) and len(trace) > 0, "params_trace missing or empty"
assert [s["epoch"] for s in trace] == [0, 10, 20, 24], [s["epoch"] for s in trace]
for snap in trace:
    assert set(PARAM_NAMES) <= set(snap.keys()), snap.keys()
    assert "mu_h" not in snap, "mu_h must not be in the trace"

first, last = trace[0], trace[-1]
assert any(first[n] != last[n] for n in PARAM_NAMES), "params did not change across training"
print("OK task1", [s["epoch"] for s in trace])
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd /home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns && conda run -n pinns python /tmp/check_task1.py`

Expected: `KeyError: 'params_trace'` (the key doesn't exist yet).

- [ ] **Step 3: Add `param_log_every` to `BaselineConfig`**

In `baseline_experiments.py`, change:

```python
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
```

to:

```python
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
```

- [ ] **Step 4: Snapshot params inside the training loop**

In `train_one()`, change:

```python
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
    }
    return summary
```

to:

```python
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
```

- [ ] **Step 5: Run the verification script again and confirm it passes**

Run: `cd /home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns && conda run -n pinns python /tmp/check_task1.py`

Expected: `OK task1 [0, 10, 20, 24]`

- [ ] **Step 6: Commit**

```bash
git add baseline_experiments.py
git commit -m "Capture per-epoch ODE parameter trace in train_one()"
```

---

### Task 2: Write the parameter trace to per-seed CSVs

**Files:**
- Modify: `baseline_experiments.py:498-531` (add `_seed_param_rows`, near `_seed_result_row`/`_seed_prediction_rows`)
- Modify: `baseline_experiments.py:533-623` (`run_all_baseline_table`)
- Modify: `baseline_experiments.py:626-650` (`__all__`, not required to export the private helper, no change needed there)

**Interfaces:**
- Consumes: `run["params_trace"]` from Task 1 (list of `{"epoch": int, **7 param floats}` dicts), `run["region"]`, `run["model"]`, `run["seed"]`.
- Produces: `_seed_param_rows(run: dict) -> list[dict]`, and a new `params_dir: str | Path | None = None` keyword argument on `run_all_baseline_table(...)`. When set, writes `<params_dir>/<region>/seed<N>/<model>.csv` with columns `region, model, seed, epoch, lambda, beta_m, mu_m, theta_m, beta_h, theta_h, gamma_h`.

- [ ] **Step 1: Write the failing verification script**

Create `/tmp/check_task2.py`:

```python
import sys, tempfile, pathlib
sys.path.insert(0, "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns")
from baseline_experiments import BaselineConfig, run_all_baseline_table
import pandas as pd

with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    cfg = BaselineConfig(epochs=15, param_log_every=5, log_every=100)
    run_all_baseline_table(
        regions=["iquitos"], model_types=["dinn"], config=cfg,
        seeds=[0, 1], params_dir=tmp / "params", verbose=False,
    )
    for seed in (0, 1):
        path = tmp / "params" / "iquitos" / f"seed{seed}" / "dinn.csv"
        assert path.exists(), f"missing {path}"
        df = pd.read_csv(path)
        expected_cols = [
            "region", "model", "seed", "epoch",
            "lambda", "beta_m", "mu_m", "theta_m", "beta_h", "theta_h", "gamma_h",
        ]
        assert list(df.columns) == expected_cols, df.columns.tolist()
        assert df["epoch"].tolist() == [0, 5, 10, 14], df["epoch"].tolist()
        assert (df["seed"] == seed).all()
        assert (df["region"] == "iquitos").all()
        assert (df["model"] == "dinn").all()
print("OK task2")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `cd /home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns && conda run -n pinns python /tmp/check_task2.py`

Expected: `TypeError: run_all_baseline_table() got an unexpected keyword argument 'params_dir'`

- [ ] **Step 3: Add `_seed_param_rows`**

In `baseline_experiments.py`, immediately after the existing `_seed_prediction_rows` function (right before the `# --- ` comment / `def run_all_baseline_table` definition), add:

```python
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
```

- [ ] **Step 4: Thread `params_dir` through `run_all_baseline_table`**

Change the signature:

```python
def run_all_baseline_table(
    regions: Iterable[str] | None = None,
    model_types: Iterable[str] = ("dinn", "nec"),
    config: BaselineConfig | None = None,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    data_dir: str | Path | None = None,
    range7_dir: str | Path | None = None,  # accepted for API compatibility; unused
    results_dir: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
```

to:

```python
def run_all_baseline_table(
    regions: Iterable[str] | None = None,
    model_types: Iterable[str] = ("dinn", "nec"),
    config: BaselineConfig | None = None,
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    data_dir: str | Path | None = None,
    range7_dir: str | Path | None = None,  # accepted for API compatibility; unused
    results_dir: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    params_dir: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
```

Then, inside the function body, change:

```python
        seed_result_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}
        seed_prediction_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}
```

to:

```python
        seed_result_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}
        seed_prediction_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}
        seed_param_rows: dict[int, list[dict]] = {seed: [] for seed in seed_list}
```

and, in the loop over `model_types` where `seed_result_rows`/`seed_prediction_rows` are populated:

```python
            for run in agg["runs"]:
                seed_result_rows[run["seed"]].append(_seed_result_row(run))
                seed_prediction_rows[run["seed"]].extend(_seed_prediction_rows(run))
```

to:

```python
            for run in agg["runs"]:
                seed_result_rows[run["seed"]].append(_seed_result_row(run))
                seed_prediction_rows[run["seed"]].extend(_seed_prediction_rows(run))
                seed_param_rows[run["seed"]].extend(_seed_param_rows(run))
```

Finally, after the existing `if predictions_dir is not None:` block (which ends right before `return pd.DataFrame(all_result_rows)`), add a parallel block:

```python
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
```

- [ ] **Step 5: Run the verification script again and confirm it passes**

Run: `cd /home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns && conda run -n pinns python /tmp/check_task2.py`

Expected: `OK task2`

- [ ] **Step 6: Commit**

```bash
git add baseline_experiments.py
git commit -m "Write per-seed DINN/NEC parameter traces to CSV via run_all_baseline_table"
```

---

### Task 3: Add a `--params-output` CLI flag to `run_baseline_table.py`

**Files:**
- Modify: `run_baseline_table.py:29-99` (`parse_args`)
- Modify: `run_baseline_table.py:413-440` (`main`)

**Interfaces:**
- Consumes: `run_all_baseline_table(..., params_dir=...)` from Task 2.
- Produces: `args.params_output` (a `Path`), passed through to `run_all_baseline_table`.

- [ ] **Step 1: Write the failing verification script**

Create `/tmp/check_task3.sh`:

```bash
#!/bin/bash
set -e
cd /home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns
rm -rf /tmp/res_t3 /tmp/pred_t3 /tmp/params_t3
conda run -n pinns python run_baseline_table.py --help | grep -q -- "--params-output"
conda run -n pinns python run_baseline_table.py \
    --epochs 12 --regions iquitos --models dinn --quiet --no-plots \
    --output /tmp/res_t3 --predictions-output /tmp/pred_t3 --params-output /tmp/params_t3
test -f /tmp/params_t3/iquitos/seed0/dinn.csv
echo "OK task3"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `chmod +x /tmp/check_task3.sh && /tmp/check_task3.sh`

Expected: the `grep -q -- "--params-output"` line fails (flag doesn't exist yet), script exits non-zero because of `set -e`.

- [ ] **Step 3: Add the CLI flag**

In `run_baseline_table.py`, change:

```python
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=default_output_dir() / "dinns_nec_predictions",
        help=(
            "Directory for per-region calibration/projection prediction CSVs, "
            "with per-seed CSVs under <region>/seed<N>/<model>.csv."
        ),
    )
```

to:

```python
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=default_output_dir() / "dinns_nec_predictions",
        help=(
            "Directory for per-region calibration/projection prediction CSVs, "
            "with per-seed CSVs under <region>/seed<N>/<model>.csv."
        ),
    )
    parser.add_argument(
        "--params-output",
        type=Path,
        default=default_output_dir() / "dinns_nec_params",
        help=(
            "Directory for per-seed ODE parameter trace CSVs "
            "(<region>/seed<N>/<model>.csv, logged every --param-log-every epochs)."
        ),
    )
    parser.add_argument(
        "--param-log-every",
        type=int,
        default=BaselineConfig().param_log_every,
        help="Snapshot trainable ODE params every N epochs.",
    )
```

- [ ] **Step 4: Pass the new args through in `main()`**

In `run_baseline_table.py`, change:

```python
    table = pd.DataFrame()
    if not args.plot_only:
        config = BaselineConfig(
            epochs=args.epochs,
            w_phys=args.w_phys,
        )
        table = run_all_baseline_table(
            regions=regions,
            model_types=args.models,
            config=config,
            results_dir=args.output,
            predictions_dir=args.predictions_output,
            verbose=not args.quiet,
        )
```

to:

```python
    table = pd.DataFrame()
    if not args.plot_only:
        config = BaselineConfig(
            epochs=args.epochs,
            w_phys=args.w_phys,
            param_log_every=args.param_log_every,
        )
        table = run_all_baseline_table(
            regions=regions,
            model_types=args.models,
            config=config,
            results_dir=args.output,
            predictions_dir=args.predictions_output,
            params_dir=args.params_output,
            verbose=not args.quiet,
        )
```

- [ ] **Step 5: Run the verification script again and confirm it passes**

Run: `/tmp/check_task3.sh`

Expected: `OK task3`

- [ ] **Step 6: Commit**

```bash
git add run_baseline_table.py
git commit -m "Add --params-output / --param-log-every CLI flags for parameter-trace CSVs"
```

---

### Task 4: Wire `PARAMS_DIR` into `baseline_all_results.ipynb`

**Files:**
- Modify: `baseline_all_results.ipynb` (cell containing `RESULTS_DIR = default_output_dir() / "dinns_nec_results"`, currently cell index 4)
- Modify: `baseline_all_results.ipynb` (cell containing `RUN_FULL_SWEEP = True`, currently cell index 8)

**Interfaces:**
- Consumes: `run_all_baseline_table(..., params_dir=...)` from Task 2.
- Produces: a `PARAMS_DIR` notebook variable, used the same way `RESULTS_DIR`/`PREDICTIONS_DIR` already are.

- [ ] **Step 1: Read the notebook**

Use the Read tool on `/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/baseline_all_results.ipynb` (it is small enough; `NotebookEdit` requires a prior successful `Read` of the file in the conversation). The two target cells have ids `0dc5d58f` (constants) and `d1f04d7c` (full sweep) — confirm these still match after reading; if the notebook has since changed, locate the cells by content instead (`RESULTS_DIR = default_output_dir()` and `RUN_FULL_SWEEP = True`).

- [ ] **Step 2: Write the failing verification check**

Create `/tmp/check_task4.py`:

```python
import json
path = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/baseline_all_results.ipynb"
nb = json.load(open(path))
sources = ["".join(c["source"]) for c in nb["cells"]]
assert any("PARAMS_DIR" in s for s in sources), "PARAMS_DIR not defined anywhere"
assert any("params_dir=PARAMS_DIR" in s for s in sources), "params_dir=PARAMS_DIR not passed to run_all_baseline_table"
print("OK task4")
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `conda run -n pinns python /tmp/check_task4.py`

Expected: `AssertionError: PARAMS_DIR not defined anywhere`

- [ ] **Step 4: Edit the constants cell**

Using `NotebookEdit` with `edit_mode="replace"` and `cell_id="0dc5d58f"`, the cell's current full source is:

```python
REGIONS = discover_regions()
MODEL_TYPES = ("dinn", "nec")
EPOCHS = 2000       # match the NPC paper training budget
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # multi-seed ensemble

RESULTS_DIR = default_output_dir() / "dinns_nec_results"
PREDICTIONS_DIR = default_output_dir() / "dinns_nec_predictions"

if not REGIONS:
    raise RuntimeError(
        "No NPC data found under data. Place data_cali.csv and "
        "data_proj.csv under data/<region>/."
    )

print("regions:", REGIONS)
print("seeds  :", SEEDS)
print("results ->", RESULTS_DIR)
```

Set `new_source` to the same content with one addition, `PARAMS_DIR`, placed next to `PREDICTIONS_DIR`, plus a matching print line:

```python
REGIONS = discover_regions()
MODEL_TYPES = ("dinn", "nec")
EPOCHS = 2000       # match the NPC paper training budget
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # multi-seed ensemble

RESULTS_DIR = default_output_dir() / "dinns_nec_results"
PREDICTIONS_DIR = default_output_dir() / "dinns_nec_predictions"
PARAMS_DIR = default_output_dir() / "dinns_nec_params"

if not REGIONS:
    raise RuntimeError(
        "No NPC data found under data. Place data_cali.csv and "
        "data_proj.csv under data/<region>/."
    )

print("regions:", REGIONS)
print("seeds  :", SEEDS)
print("results ->", RESULTS_DIR)
print("params  ->", PARAMS_DIR)
```

- [ ] **Step 5: Edit the full-sweep cell**

Using `NotebookEdit` with `edit_mode="replace"` and `cell_id="d1f04d7c"`, the cell's current full source is:

```python
RUN_FULL_SWEEP = True

full_cfg = BaselineConfig(epochs=EPOCHS, log_every=500)

if RUN_FULL_SWEEP:
    results = run_all_baseline_table(
        regions=REGIONS,
        model_types=MODEL_TYPES,
        config=full_cfg,
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        predictions_dir=PREDICTIONS_DIR,
        verbose=True,
    )
elif RESULTS_DIR.exists():
    results = pd.concat(
        (pd.read_csv(p) for p in sorted(RESULTS_DIR.glob("*.csv"))),
        ignore_index=True,
    )
else:
    results = pd.DataFrame()
    print("No saved results yet. Set RUN_FULL_SWEEP = True to train.")

display(results)
```

Set `new_source` to the same content, adding `params_dir=PARAMS_DIR` to the `run_all_baseline_table(...)` call (the `elif`/`else` branches are unchanged):

```python
RUN_FULL_SWEEP = True

full_cfg = BaselineConfig(epochs=EPOCHS, log_every=500)

if RUN_FULL_SWEEP:
    results = run_all_baseline_table(
        regions=REGIONS,
        model_types=MODEL_TYPES,
        config=full_cfg,
        seeds=SEEDS,
        results_dir=RESULTS_DIR,
        predictions_dir=PREDICTIONS_DIR,
        params_dir=PARAMS_DIR,
        verbose=True,
    )
elif RESULTS_DIR.exists():
    results = pd.concat(
        (pd.read_csv(p) for p in sorted(RESULTS_DIR.glob("*.csv"))),
        ignore_index=True,
    )
else:
    results = pd.DataFrame()
    print("No saved results yet. Set RUN_FULL_SWEEP = True to train.")

display(results)
```

- [ ] **Step 6: Run the verification script again and confirm it passes**

Run: `conda run -n pinns python /tmp/check_task4.py`

Expected: `OK task4`

- [ ] **Step 7: Commit**

```bash
git add baseline_all_results.ipynb
git commit -m "Wire PARAMS_DIR through the full DINN/NEC sweep notebook"
```

---

### Task 5: Generalize `plot_density` to overlay N series

**Files:**
- Modify: `FigureGeneration/visualisation.ipynb` (the cell defining `plot_density`, found by the unique marker `"def plot_density(ax, data_npc, data_mcmc"`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `plot_density(ax, series, param, min_scale, max_scale, quantile_threshold=0.99)` where `series` is a list of `(data: np.ndarray, label: str, color: str)` tuples. This replaces the old 2-positional-array signature `plot_density(ax, data_npc, data_mcmc, param, min_scale, max_scale, quantile_threshold=0.99)`. Task 6 is the only caller and is updated in lockstep.

- [ ] **Step 1: Write the failing verification script**

Create `/tmp/check_task5.py`:

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_density(ax, series, param, min_scale, max_scale, quantile_threshold=0.99):
    x_range = np.linspace(min_scale, max_scale, 200)
    for data, label, color in series:
        try:
            kde = gaussian_kde(data, bw_method=1)
            y = kde(x_range)
        except np.linalg.LinAlgError:
            y, bin_edges = np.histogram(data, bins=50, density=True)
            x_range = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.plot(x_range, y, color=color, lw=2, label=label)
        ax.fill_between(x_range, y, alpha=0.3, color=color)

        mean_val = np.mean(data)
        mean_y = np.interp(mean_val, x_range, y)
        ax.scatter(mean_val, mean_y, color='black', s=30, marker='o', zorder=5)

        mode_idx = np.argmax(y)
        ax.scatter(x_range[mode_idx], y[mode_idx], color='grey', s=30, marker='D', zorder=5)


rng = np.random.default_rng(0)
series = [
    (rng.normal(1.0, 0.1, 50), "NPC", "#E08E45"),
    (rng.normal(1.1, 0.1, 200), "MCMC", "#4E79A7"),
    (rng.normal(0.9, 0.1, 10), "DINN", "#8175AA"),
    (rng.normal(1.05, 0.1, 10), "NEC", "#C0504D"),
]
fig, ax = plt.subplots()
plot_density(ax, series, "beta_h", 0.5, 1.5)
assert len(ax.lines) == 4, f"expected 4 line series, got {len(ax.lines)}"
assert len(ax.collections) >= 4, "expected at least 4 fill_between/scatter collections"
plt.close(fig)
print("OK task5")
```

- [ ] **Step 2: Run it to confirm the *standalone copy* passes (sanity check before touching the notebook)**

Run: `conda run -n denguetwin python /tmp/check_task5.py`

Expected: `OK task5`. (This validates the new function body in isolation before it's transplanted into the notebook cell — the notebook edit in Step 3 must use this exact function body.)

- [ ] **Step 3: Replace the `plot_density` cell in the notebook**

Create `/tmp/edit_task5.py`:

```python
import json

path = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb"
with open(path) as f:
    nb = json.load(f)

marker = "def plot_density(ax, data_npc, data_mcmc"
target = None
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if marker in src:
        target = cell
        break
assert target is not None, "plot_density cell not found"

new_source = '''import h5py
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_density(ax, series, param, min_scale, max_scale, quantile_threshold=0.99):
    """Overlay KDE density curves (+ mean/mode markers) for each (data, label, color) in `series`."""
    x_range = np.linspace(min_scale, max_scale, 200)
    for data, label, color in series:
        try:
            kde = gaussian_kde(data, bw_method=1)
            y = kde(x_range)
        except np.linalg.LinAlgError:
            # Fallback to histogram-based density estimation
            y, bin_edges = np.histogram(data, bins=50, density=True)
            x_range = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.plot(x_range, y, color=color, lw=2, label=label)
        ax.fill_between(x_range, y, alpha=0.3, color=color)

        mean_val = np.mean(data)
        mean_y = np.interp(mean_val, x_range, y)
        ax.scatter(mean_val, mean_y, color='black', s=30, marker='o', zorder=5)

        mode_idx = np.argmax(y)
        ax.scatter(x_range[mode_idx], y[mode_idx], color='grey', s=30, marker='D', zorder=5)
'''

lines = new_source.splitlines(keepends=True)
target["source"] = lines
target["outputs"] = []
target["execution_count"] = None

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("rewrote plot_density cell")
```

Run: `conda run -n denguetwin python /tmp/edit_task5.py`

Expected: `rewrote plot_density cell`

- [ ] **Step 4: Confirm the notebook file is still valid JSON and the new signature landed**

Run:

```bash
python3 -c "
import json
nb = json.load(open('/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb'))
print('valid JSON, cells:', len(nb['cells']))
src = ''.join(c for cell in nb['cells'] for c in cell['source'] if 'def plot_density' in ''.join(cell['source']))
"
grep -n "def plot_density(ax, series" /home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb
```

Expected: `valid JSON, cells: <some number>` and one matching line for the new signature.

- [ ] **Step 5: Commit**

```bash
git add FigureGeneration/visualisation.ipynb
git commit -m "Generalize plot_density to overlay an arbitrary number of series"
```

---

### Task 6: Load DINN/NEC parameters into the marginal-posterior cells

**Files:**
- Modify: `FigureGeneration/visualisation.ipynb` (cities cell, found by unique marker `"cities = ['bello', 'iquitos', 'sanjuan']"`)
- Modify: `FigureGeneration/visualisation.ipynb` (countries cell, found by unique marker `"countries = ['vietnam', 'philippines', 'cambodia']"`)

**Interfaces:**
- Consumes: `plot_density(ax, series, param, min_scale, max_scale)` from Task 5; CSVs at `outputs/dinns_nec_params/<short_region_key>/seed*/<model>.csv` from Task 2 (columns `region, model, seed, epoch, lambda, beta_m, mu_m, theta_m, beta_h, theta_h, gamma_h`).
- Produces: the cities/countries marginal-posterior grids now overlay MCMC, NPC, DINN, and NEC per parameter, using colors `{"MCMC": "#4E79A7", "NPC": "#E08E45", "DINN": "#8175AA", "NEC": "#C0504D"}`.

Note: the original cells contained a dead branch (`if param in initial_conditions: ... * population`) that can never execute, because this loop only iterates over `param_names` (the 7 rate parameters) and `initial_conditions` is a disjoint list never referenced inside the loop. It is dropped below since the loop body is being rewritten anyway to support N series.

- [ ] **Step 1: Write the failing verification harness**

Create `/tmp/check_task6_setup.py` — this builds a tiny fixture tree of fake DINN/NEC parameter CSVs for `iquitos`, mirroring Task 2's real output format, so the test doesn't depend on having run real training:

```python
import pathlib
import pandas as pd
import numpy as np

FIXTURE_DIR = pathlib.Path("/tmp/check_task6_params")
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

param_cols = ["lambda", "beta_m", "mu_m", "theta_m", "beta_h", "theta_h", "gamma_h"]
rng = np.random.default_rng(0)

# 10 seeds, matching the real sweep's SEEDS = [0..9]. Fewer seeds (e.g. 3) make the
# 1%/99% quantile trim in the cities/countries cell collapse to a single point
# (np.quantile on n=3 with q=0.01/0.99 trims to the middle element only), which is
# not representative of the real 10-seed case this cell is built for.
for model in ("dinn", "nec"):
    for seed in range(10):
        seed_dir = FIXTURE_DIR / "iquitos" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for epoch in (0, 5, 10):
            row = {"region": "iquitos", "model": model, "seed": seed, "epoch": epoch}
            row.update({c: float(rng.uniform(0.01, 0.5)) for c in param_cols})
            rows.append(row)
        pd.DataFrame(rows).to_csv(seed_dir / f"{model}.csv", index=False)

print("fixture written to", FIXTURE_DIR)
```

Run: `conda run -n denguetwin python /tmp/check_task6_setup.py`

Expected: `fixture written to /tmp/check_task6_params`

- [ ] **Step 2: Write the notebook-execution verification script**

Create `/tmp/check_task6.py`. This extracts the *actual* (already-edited-by-Step-3) cities cell source from the real notebook, patches only the `DINNS_NEC_PARAMS_DIR` constant to point at the Step 1 fixture (so it doesn't require a real training run), and executes it for real via `nbclient` against the real `iquitos` MCMC/NPC data files (which already exist under `FigureGeneration/iquitos/`):

```python
import json
import pathlib
import nbformat
from nbclient import NotebookClient

NB_PATH = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb"
FIG_DIR = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration"

nb = json.load(open(NB_PATH))


def cell_source(marker):
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if marker in src:
            return src
    raise AssertionError(f"no cell found containing {marker!r}")


plot_density_src = cell_source("def plot_density(ax, series")
cities_src = cell_source("cities = ['bello', 'iquitos', 'sanjuan']")

# Point at the Step 1 fixture instead of the real (not-yet-generated) training output,
# and restrict to a single region (iquitos) so the smoke run only needs files that
# already exist in this repo.
cities_src = cities_src.replace(
    'DINNS_NEC_PARAMS_DIR = "../outputs/dinns_nec_params"',
    'DINNS_NEC_PARAMS_DIR = "/tmp/check_task6_params"',
)
cities_src = cities_src.replace(
    "cities = ['bello', 'iquitos', 'sanjuan']",
    "cities = ['iquitos']",
)
cities_src = cities_src.replace(
    "city_titles = ['Bello', 'Iquitos', 'San Juan']",
    "city_titles = ['Iquitos']",
)

test_nb = nbformat.v4.new_notebook()
test_nb.cells = [
    nbformat.v4.new_code_cell("import matplotlib\nmatplotlib.use('Agg')"),
    nbformat.v4.new_code_cell(plot_density_src),
    nbformat.v4.new_code_cell(cities_src),
]

client = NotebookClient(test_nb, timeout=120, kernel_name="python3",
                         resources={"metadata": {"path": FIG_DIR}})
client.execute()

errors = [
    out for cell in test_nb.cells for out in cell.get("outputs", [])
    if out.get("output_type") == "error"
]
if errors:
    for e in errors:
        print("".join(e.get("traceback", [])))
    raise SystemExit("notebook cell raised an error")

print("OK task6")
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `conda run -n denguetwin python /tmp/check_task6.py`

Expected: `AssertionError: no cell found containing "def plot_density(ax, series"` (Task 5 not yet applied) — if Task 5 is already committed, expect instead some exception raised while executing the not-yet-updated cities cell (it still calls the old 2-array `plot_density(ax, data_npc, data_mcmc, ...)` form, which no longer matches Task 5's `plot_density(ax, series, param, min_scale, max_scale)` signature, and it has no `load_dinns_nec_final_params`/`all_params_dinn` yet either) — the exact exception type doesn't matter, it just must fail before Step 4.

- [ ] **Step 4: Replace the cities cell**

Create `/tmp/edit_task6_cities.py`:

```python
import json

path = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb"
with open(path) as f:
    nb = json.load(f)

marker = "cities = ['bello', 'iquitos', 'sanjuan']"
target = None
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if marker in src:
        target = cell
        break
assert target is not None, "cities cell not found"

new_source = '''import glob
import pandas as pd

param_names = ['beta_h', 'beta_m', 'gamma_h', 'lambda',
               'mu_m', 'theta_h', 'theta_m']
population = 544549

# Define cities to plot
cities = ['bello', 'iquitos', 'sanjuan']
city_titles = ['Bello', 'Iquitos', 'San Juan']

DINNS_NEC_PARAMS_DIR = "../outputs/dinns_nec_params"
REGION_KEY = {
    'bello': 'bello', 'iquitos': 'iquitos', 'sanjuan': 'sanjuan',
    'vietnam': 'vn', 'philippines': 'phi', 'cambodia': 'cam',
}


def load_dinns_nec_final_params(region, model):
    """Final-epoch parameter vector per seed: (n_seeds, len(param_names)), in `param_names` order."""
    region_key = REGION_KEY[region]
    seed_files = sorted(glob.glob(f"{DINNS_NEC_PARAMS_DIR}/{region_key}/seed*/{model}.csv"))
    rows = []
    for path in seed_files:
        df = pd.read_csv(path).sort_values("epoch")
        final = df.iloc[-1]
        rows.append([final[p] for p in param_names])
    return np.array(rows)


METHOD_COLORS = {"MCMC": "#4E79A7", "NPC": "#E08E45", "DINN": "#8175AA", "NEC": "#C0504D"}

# Load data for all cities
all_params_mcmc = []
all_params_npc = []
all_params_dinn = []
all_params_nec = []

for city in cities:
    # Load MCMC parameters
    mat = scipy.io.loadmat(f"{city}/{city}_MCMC_all_params.mat")
    params_mcmc = mat[f'{city}_mcmc']
    all_params_mcmc.append(params_mcmc)

    # Load NPC parameters
    param_list = []
    folders = sorted([f for f in os.listdir(f"{city}/data") if f.startswith("uni")])
    for folder in folders:
        h5_path = os.path.join(f"{city}/data", folder, "data.h5")
        if not os.path.exists(h5_path):
            print(f"File not found: {h5_path}")
            continue
        with h5py.File(h5_path, "r") as hf:
            params = hf["Dengue"]["parameters"][-1, :]
            param_list.append(params)
    params_npc = np.vstack(param_list)
    all_params_npc.append(params_npc)

    # Load DINN / NEC parameters (final epoch per seed)
    all_params_dinn.append(load_dinns_nec_final_params(city, "dinn"))
    all_params_nec.append(load_dinns_nec_final_params(city, "nec"))

# Define parameter names for axis titles
math_param_names = {
    "beta_h": r"$\\beta_{h}$", "beta_m": r"$\\beta_{m}$",
    "gamma_h": r"$\\gamma_{h}$", "lambda": r"$\\lambda$", "mu_m": r"$\\mu_{m}$",
    "theta_h": r"$\\theta_{h}$", "theta_m": r"$\\theta_{m}$"
}

# Set up plot grid (3 rows, 8 columns)
nrows, ncols = 3, 8
fig, axs = plt.subplots(nrows, ncols, figsize=(40, 18))

# Plot densities for each city and parameter
for city_idx, (city, city_title) in enumerate(zip(cities, city_titles)):
    params_mcmc = all_params_mcmc[city_idx]
    params_npc = all_params_npc[city_idx]
    params_dinn = all_params_dinn[city_idx]
    params_nec = all_params_nec[city_idx]

    for param_idx, param in enumerate(param_names):
        ax = axs[city_idx, param_idx]
        all_data = {
            "NPC": params_npc[:, param_idx],
            "MCMC": params_mcmc[param_idx, :],
            "DINN": params_dinn[:, param_idx],
            "NEC": params_nec[:, param_idx],
        }

        # Trim each method's own samples to its central 99% interval
        trimmed = {}
        for label, data in all_data.items():
            q_low, q_high = np.quantile(data.ravel(), [0.01, 0.99])
            trimmed[label] = data[(data >= q_low) & (data <= q_high)]

        min_scale = min(d.min() for d in trimmed.values())
        max_scale = max(d.max() for d in trimmed.values())
        series = [(trimmed[label], label, METHOD_COLORS[label]) for label in ["NPC", "MCMC", "DINN", "NEC"]]
        plot_density(ax, series, param, min_scale, max_scale)

        # Set titles only for top row
        if city_idx == 0:
            ax.set_title(math_param_names[param], fontsize=30)

        # Set y-label only for first column
        if param_idx == 0:
            ax.set_ylabel(f"{city_title}\\nDensity", fontsize=30)

        # Increase tick label size
        ax.tick_params(axis='y', labelsize=25)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)  # no marks, no labels


# Add legend only once in the top right corner
legend_ax = axs[0, -1]
legend_ax.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["NPC"], alpha=0.6),
           plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["MCMC"], alpha=0.6),
           plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["DINN"], alpha=0.6),
           plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["NEC"], alpha=0.6),
           plt.scatter([0], [0], color='black', marker='o'),
           plt.scatter([0], [0], color='grey', marker='D')]
labels = ['NPC', 'MCMC', 'DINN', 'NEC', 'Mean', 'Mode']
legend_ax.legend(handles, labels, loc='center', fontsize=26)

# Hide the legend positions for other rows
for city_idx in [1, 2]:
    axs[city_idx, -1].axis("off")

# Adjust layout with more spacing
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
# plt.savefig('marginal_densities_cities.tiff', dpi=200, format='tiff')
plt.show()
'''

lines = new_source.splitlines(keepends=True)
target["source"] = lines
target["outputs"] = []
target["execution_count"] = None

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("rewrote cities cell")
```

Run: `conda run -n denguetwin python /tmp/edit_task6_cities.py`

Expected: `rewrote cities cell`

- [ ] **Step 5: Replace the countries cell the same way**

Create `/tmp/edit_task6_countries.py`:

```python
import json

path = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb"
with open(path) as f:
    nb = json.load(f)

marker = "countries = ['vietnam', 'philippines', 'cambodia']"
target = None
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if marker in src:
        target = cell
        break
assert target is not None, "countries cell not found"

new_source = '''import glob
import pandas as pd

param_names = ['beta_h', 'beta_m', 'gamma_h', 'lambda',
               'mu_m', 'theta_h', 'theta_m']
population = 544549

# Define countries to plot
countries = ['vietnam', 'philippines', 'cambodia']
country_titles = ['Vietnam', 'Philippines', 'Cambodia']

DINNS_NEC_PARAMS_DIR = "../outputs/dinns_nec_params"
REGION_KEY = {
    'bello': 'bello', 'iquitos': 'iquitos', 'sanjuan': 'sanjuan',
    'vietnam': 'vn', 'philippines': 'phi', 'cambodia': 'cam',
}


def load_dinns_nec_final_params(region, model):
    """Final-epoch parameter vector per seed: (n_seeds, len(param_names)), in `param_names` order."""
    region_key = REGION_KEY[region]
    seed_files = sorted(glob.glob(f"{DINNS_NEC_PARAMS_DIR}/{region_key}/seed*/{model}.csv"))
    rows = []
    for path in seed_files:
        df = pd.read_csv(path).sort_values("epoch")
        final = df.iloc[-1]
        rows.append([final[p] for p in param_names])
    return np.array(rows)


METHOD_COLORS = {"MCMC": "#4E79A7", "NPC": "#E08E45", "DINN": "#8175AA", "NEC": "#C0504D"}

# Load data for all countries
all_params_mcmc = []
all_params_npc = []
all_params_dinn = []
all_params_nec = []

for country in countries:
    # Load MCMC parameters
    mat = scipy.io.loadmat(f"{country}/{country}_MCMC_all_params.mat")
    params_mcmc = mat[f'{country}_mcmc']
    all_params_mcmc.append(params_mcmc)

    # Load NPC parameters
    param_list = []
    folders = sorted([f for f in os.listdir(f"{country}/data") if f.startswith("uni")])
    for folder in folders:
        h5_path = os.path.join(f"{country}/data", folder, "data.h5")
        if not os.path.exists(h5_path):
            print(f"File not found: {h5_path}")
            continue
        with h5py.File(h5_path, "r") as hf:
            params = hf["Dengue"]["parameters"][-1, :]
            param_list.append(params)
    params_npc = np.vstack(param_list)
    all_params_npc.append(params_npc)

    # Load DINN / NEC parameters (final epoch per seed)
    all_params_dinn.append(load_dinns_nec_final_params(country, "dinn"))
    all_params_nec.append(load_dinns_nec_final_params(country, "nec"))

# Define parameter names for axis titles
math_param_names = {
    "beta_h": r"$\\beta_{h}$", "beta_m": r"$\\beta_{m}$",
    "gamma_h": r"$\\gamma_{h}$", "lambda": r"$\\lambda$", "mu_m": r"$\\mu_{m}$",
    "theta_h": r"$\\theta_{h}$", "theta_m": r"$\\theta_{m}$"
}

# Set up plot grid (3 rows, 8 columns)
nrows, ncols = 3, 8
fig, axs = plt.subplots(nrows, ncols, figsize=(40, 18))

# Plot densities for each country and parameter
for country_idx, (country, country_title) in enumerate(zip(countries, country_titles)):
    params_mcmc = all_params_mcmc[country_idx]
    params_npc = all_params_npc[country_idx]
    params_dinn = all_params_dinn[country_idx]
    params_nec = all_params_nec[country_idx]

    for param_idx, param in enumerate(param_names):
        ax = axs[country_idx, param_idx]
        all_data = {
            "NPC": params_npc[:, param_idx],
            "MCMC": params_mcmc[param_idx, :],
            "DINN": params_dinn[:, param_idx],
            "NEC": params_nec[:, param_idx],
        }

        # Trim each method's own samples to its central 99% interval
        trimmed = {}
        for label, data in all_data.items():
            q_low, q_high = np.quantile(data.ravel(), [0.01, 0.99])
            trimmed[label] = data[(data >= q_low) & (data <= q_high)]

        min_scale = min(d.min() for d in trimmed.values())
        max_scale = max(d.max() for d in trimmed.values())
        series = [(trimmed[label], label, METHOD_COLORS[label]) for label in ["NPC", "MCMC", "DINN", "NEC"]]
        plot_density(ax, series, param, min_scale, max_scale)

        # Set titles only for top row
        if country_idx == 0:
            ax.set_title(math_param_names[param], fontsize=30)

        # Set y-label only for first column
        if param_idx == 0:
            ax.set_ylabel(f"{country_title}\\nDensity", fontsize=30)

        # Increase tick label size
        ax.tick_params(axis='y', labelsize=25)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)  # no marks, no labels

# Add legend only once in the top right corner
legend_ax = axs[0, -1]
legend_ax.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["NPC"], alpha=0.6),
           plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["MCMC"], alpha=0.6),
           plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["DINN"], alpha=0.6),
           plt.Rectangle((0, 0), 1, 1, fc=METHOD_COLORS["NEC"], alpha=0.6),
           plt.scatter([0], [0], color='black', marker='o'),
           plt.scatter([0], [0], color='grey', marker='D')]
labels = ['NPC', 'MCMC', 'DINN', 'NEC', 'Mean', 'Mode']
legend_ax.legend(handles, labels, loc='center', fontsize=26)

# Hide the legend positions for other rows
for country_idx in [1, 2]:
    axs[country_idx, -1].axis("off")

# Adjust layout with more spacing
plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
# plt.savefig('marginal_densities_countries.tiff', dpi=200, format='tiff')
plt.show()
'''

lines = new_source.splitlines(keepends=True)
target["source"] = lines
target["outputs"] = []
target["execution_count"] = None

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("rewrote countries cell")
```

Run: `conda run -n denguetwin python /tmp/edit_task6_countries.py`

Expected: `rewrote countries cell`

- [ ] **Step 6: Confirm the notebook is still valid JSON**

Run:

```bash
python3 -c "
import json
nb = json.load(open('/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb'))
print('valid JSON, cells:', len(nb['cells']))
"
```

Expected: `valid JSON, cells: <some number>` (no exception).

- [ ] **Step 7: Run the verification script again and confirm it passes**

Run: `conda run -n denguetwin python /tmp/check_task6.py`

Expected: `OK task6`

- [ ] **Step 8: Repeat the same execution check for the countries cell**

`vietnam/vietnam_MCMC_all_params.mat` and `vietnam/data/uni015/data.h5` already exist under `FigureGeneration/`, confirmed during planning — so this step runs unconditionally, no fallback needed.

Create `/tmp/check_task6_countries_setup.py` (mirrors Step 1, but writes the fixture under the `vn` short key, since `REGION_KEY['vietnam'] == 'vn'`):

```python
import pathlib
import pandas as pd
import numpy as np

FIXTURE_DIR = pathlib.Path("/tmp/check_task6_params")
param_cols = ["lambda", "beta_m", "mu_m", "theta_m", "beta_h", "theta_h", "gamma_h"]
rng = np.random.default_rng(1)

for model in ("dinn", "nec"):
    for seed in range(10):
        seed_dir = FIXTURE_DIR / "vn" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for epoch in (0, 5, 10):
            row = {"region": "vn", "model": model, "seed": seed, "epoch": epoch}
            row.update({c: float(rng.uniform(0.01, 0.5)) for c in param_cols})
            rows.append(row)
        pd.DataFrame(rows).to_csv(seed_dir / f"{model}.csv", index=False)

print("fixture written for vn")
```

Run: `conda run -n denguetwin python /tmp/check_task6_countries_setup.py`

Expected: `fixture written for vn`

Create `/tmp/check_task6_countries.py`:

```python
import json
import nbformat
from nbclient import NotebookClient

NB_PATH = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration/visualisation.ipynb"
FIG_DIR = "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns/FigureGeneration"

nb = json.load(open(NB_PATH))


def cell_source(marker):
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if marker in src:
            return src
    raise AssertionError(f"no cell found containing {marker!r}")


plot_density_src = cell_source("def plot_density(ax, series")
countries_src = cell_source("countries = ['vietnam', 'philippines', 'cambodia']")

countries_src = countries_src.replace(
    'DINNS_NEC_PARAMS_DIR = "../outputs/dinns_nec_params"',
    'DINNS_NEC_PARAMS_DIR = "/tmp/check_task6_params"',
)
countries_src = countries_src.replace(
    "countries = ['vietnam', 'philippines', 'cambodia']",
    "countries = ['vietnam']",
)
countries_src = countries_src.replace(
    "country_titles = ['Vietnam', 'Philippines', 'Cambodia']",
    "country_titles = ['Vietnam']",
)

test_nb = nbformat.v4.new_notebook()
test_nb.cells = [
    nbformat.v4.new_code_cell("import matplotlib\nmatplotlib.use('Agg')"),
    nbformat.v4.new_code_cell(plot_density_src),
    nbformat.v4.new_code_cell(countries_src),
]

client = NotebookClient(test_nb, timeout=120, kernel_name="python3",
                         resources={"metadata": {"path": FIG_DIR}})
client.execute()

errors = [
    out for cell in test_nb.cells for out in cell.get("outputs", [])
    if out.get("output_type") == "error"
]
if errors:
    for e in errors:
        print("".join(e.get("traceback", [])))
    raise SystemExit("notebook cell raised an error")

print("OK task6 countries")
```

Run: `conda run -n denguetwin python /tmp/check_task6_countries.py`

Expected: `OK task6 countries`

- [ ] **Step 9: Commit**

```bash
git add FigureGeneration/visualisation.ipynb
git commit -m "Overlay DINN/NEC parameter posteriors in the marginal-posterior density grids"
```

---

## Final check

After all 6 tasks are committed, run `git log --oneline -8` and confirm 6 new commits exist on top of the spec commit. Tell the user the code changes are in place and that the next step is for them to run the real full sweep (`baseline_all_results.ipynb`'s `RUN_FULL_SWEEP` cell, or `run_baseline_table.py` with the default 2000 epochs / 10 seeds) to populate `outputs/dinns_nec_params/`, then re-run the `visualisation.ipynb` marginal-posterior cells to see the real DINN/NEC overlays.

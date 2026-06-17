# DINN / NEC parameter trace + marginal posterior design

## Problem

`train_one()` in [baseline_experiments.py](../../../baseline_experiments.py) computes the 7 trainable
ODE parameters (`PARAM_NAMES`) every epoch but only the final value ever leaves the training loop
(via `evaluate()` -> `params_final`), and that final value is never written to disk by
`run_all_baseline_table()`. NPC, by contrast, persists a full per-iteration parameter trace per seed
(`FigureGeneration/<region>/data/uni<NNN>/data.h5`, dataset `Dengue/parameters`), which
`visualisation.ipynb` uses to build marginal posterior density plots (NPC vs MCMC, per parameter,
per region). DINN and NEC have no equivalent, so they're absent from that plot.

## Goal

For every (region, model, seed) run in the full 2000-epoch sweep, save the 7 ODE parameter values
every 10 epochs, in a layout that mirrors the existing per-seed results/predictions CSVs. Then
extend the existing NPC/MCMC marginal-posterior density cells in `visualisation.ipynb` into a
4-way MCMC/NPC/DINN/NEC overlay.

## Non-goals

- No changes to NPC/MCMC data loading or to the `.h5` format.
- No changes to model architecture, loss, or training hyperparameters.
- Not re-running the full sweep as part of this change (smoke-tested only; the user runs the full
  sweep separately).

## Design

### 1. Config

`BaselineConfig` ([baseline_experiments.py](../../../baseline_experiments.py)) gets a new field:

```python
param_log_every: int = 10
```

### 2. Capture the trace during training

In `train_one()`, the `params` dict is already computed once per epoch (used for the physics
residual loss) before `backward()`/`optimizer.step()`. No extra forward pass is needed.

Add a `params_trace: list[dict] = []` accumulator. Each epoch, if
`epoch % config.param_log_every == 0 or epoch == config.epochs - 1`, append:

```python
{"epoch": epoch, **{name: float(params[name].detach().cpu()) for name in PARAM_NAMES}}
```

`mu_h` is excluded (fixed constant, not learned). Add `"params_trace": params_trace` to the
summary dict `train_one()` returns. `run_real_experiment()` already spreads `**summary` into its
return dict, and `_aggregate_runs()` already keeps full run dicts in `agg["runs"]`, so the trace
reaches `run_all_baseline_table()` with no other changes to the aggregation path.

### 3. Write to disk

New helper, parallel to the existing `_seed_prediction_rows`:

```python
def _seed_param_rows(run: dict) -> list[dict]:
    return [
        {"region": run["region"], "model": run["model"], "seed": run["seed"], **snap}
        for snap in run["params_trace"]
    ]
```

`run_all_baseline_table()` gains an optional `params_dir: str | Path | None = None` argument.
When set, for each seed it writes:

```
<params_dir>/<region>/seed<N>/<model>.csv
```

with columns `region, model, seed, epoch, lambda, beta_m, mu_m, theta_m, beta_h, theta_h, gamma_h`
— exactly mirroring the existing `<results_dir>/<region>/seed<N>/<model>.csv` and
`<predictions_dir>/<region>/seed<N>/<model>.csv` layout.

### 4. Plumbing

- `run_baseline_table.py`: new `--params-output` CLI flag, default
  `default_output_dir() / "dinns_nec_params"`, passed to `run_all_baseline_table(params_dir=...)`.
- `baseline_all_results.ipynb`: new `PARAMS_DIR = default_output_dir() / "dinns_nec_params"`
  alongside the existing `RESULTS_DIR` / `PREDICTIONS_DIR`, passed into the full-sweep
  `run_all_baseline_table(...)` call.

### 5. Visualization

In `FigureGeneration/visualisation.ipynb`, generalize `plot_density` (currently hardcoded to two
series, `npc` and `MCMC`, with fixed colors) into a function that accepts a list of
`(data, label, color)` tuples and overlays all of them (KDE fill + mean marker + mode marker),
preserving the existing quantile-trimming and combined-scale logic.

In the city/country marginal-posterior cells, after loading `params_mcmc`/`params_npc` as today,
also load DINN and NEC: for each region, glob
`outputs/dinns_nec_params/<short_region_key>/seed*/<model>.csv`, take the last row (final epoch)
of each seed's trace, and stack across seeds into a `(n_seeds, 7)` array — the same pattern NPC
already uses by taking `hf["Dengue"]["parameters"][-1, :]` per `uni` folder. The short region key
mapping (`vietnam` -> `vn`, `philippines` -> `phi`, `cambodia` -> `cam`, identity for the three
cities) already exists in this notebook (the `REGION_KEY` dict in the Hi-grid cell) and is reused
here.

Colors reuse the palette already established in this notebook for MCMC/NPC/DINN/NEC:
`#4E79A7` / `#E08E45` / `#8175AA` / `#C0504D`.

## Output schema

`outputs/dinns_nec_params/<region>/seed<N>/<model>.csv`:

| column | meaning |
|---|---|
| region | short region key (`bello`, `cam`, `iquitos`, `phi`, `sanjuan`, `vn`) |
| model | `dinn` or `nec` |
| seed | seed integer |
| epoch | 0, 10, 20, ..., plus the final epoch |
| lambda, beta_m, mu_m, theta_m, beta_h, theta_h, gamma_h | parameter value at that epoch |

For a 2000-epoch run with `param_log_every=10`, that's 201 rows per `(region, model, seed)` file.

## Verification

- Smoke test: run one region, one model, a handful of epochs with `param_log_every` small enough
  to produce multiple snapshots; confirm the CSV is written with the expected columns/row count
  and that values change across epochs (i.e. the trace isn't just one constant row).
- Confirm `run_baseline_table.py --help` shows the new flag and the notebook cell runs without
  error on the smoke config.
- Visually confirm the extended density plot renders 4 overlaid series per parameter without
  crashing on a small synthetic/smoke `outputs/dinns_nec_params` tree.
- Full 2000-epoch x 10-seed x 6-region sweep is left to the user to run; not executed as part of
  this change.

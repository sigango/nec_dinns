# DINN / NEC Dengue Baselines

This repository implements two physics-informed neural baselines for
calibrating and projecting dengue outbreaks from a mosquito-human
compartmental ODE model:

- **DINN** (Disease-Informed Neural Network, Shaier 2022 style): a single
  MLP maps time `t` to the 8 log1p-transformed ODE states, with the 7
  epidemiological parameters learned as free (sigmoid-bounded) network
  parameters fit to one outbreak at a time.
- **NEC** (Neural Embedding Calibration): a calibration head (Net A) maps
  outbreak-level features to the 7 ODE parameters, and an emulation MLP
  (Net B, architecturally identical to DINN's) maps time to the 8 log1p
  states. Both nets train jointly on the calibration outbreak; the fitted
  emulator is then reused to project a held-out outbreak.

Both models share the dengue ODE right-hand side, parameter bounds, and training utilities in `model_common.py`, and are trained/evaluated under
the same calibration/projection protocol per region.

These baselines are compared against **NPC** (Neural Parameter
Calibration) and a classical **MCMC** calibration, both implemented in a companion repository: [neural-parameter-calibration-dengue](https://github.com/Hoangpham13579/neural-parameter-calibration-dengue).
The per-region calibration/projection CSVs under `data/<region>/` is sourced from
that repository.

## Main files

- `model_common.py` — shared constants (states, parameters, bounds), the
  dengue ODE right-hand side and its log-space residual, the NEC feature
  extractor, and small neural-network helpers.
- `dinn_model.py` — the `DINN` model.
- `nec_model.py` — the `NEC` model (calibration head + emulation MLP).
- `baseline_experiments.py` — data loading (NPC-format CSVs), the
  training loop, metrics, and prediction export for both models.
- `run_baseline_table.py` — CLI entry point that runs experiments across
  regions/seeds and writes result tables and plots.
- `baseline_all_results.ipynb` — notebook interface for full sweeps and
  reporting.
- `FigureGeneration/` — MCMC reference outputs and plotting notebook used
  to compare against DINN/NEC results.
- `boundaries/` — parameter/initial-condition bounds
  used to constrain training.

## Running the baselines

1. **Compute** — open `baseline_all_results.ipynb` and run it top to
   bottom. The "Smoke test" section runs a 2-epoch sanity check across
   regions; the "Full DINN / NEC sweep" section (`RUN_FULL_SWEEP = True`)
   trains both models for every region/seed and writes the result,
   prediction, and parameter-trace CSVs under `outputs/`.
2. **Report / visualise** — open `FigureGeneration/visualisation.ipynb`
   to load those outputs alongside the MCMC and NPC reference results,
   compute calibration/projection MSE, and produce the comparison plots
   (including the marginal posterior figures).

## Default outputs

The full sweep in `baseline_all_results.ipynb` trains both models for
every region across 10 seeds (`SEEDS = [0..9]`, 2000 epochs each) and
writes three parallel directory trees under `outputs/`, one subfolder
per region using the short keys from `NPC_REGION_DIR` (`bello`,
`iquitos`, `sanjuan`, `vn`, `phi`, `cam`):

- `outputs/dinns_nec_results/<region>/seed<N>/<model>.csv` — the same
  `mse_normalized` and training time for a single seed `N`.
- `outputs/dinns_nec_predictions/<region>/<region>.csv` — predicted vs.
  observed `Itot` trajectories (seed-averaged) for both the
  `calibration` and `projection` splits, one row per time step.
- `outputs/dinns_nec_predictions/<region>/seed<N>/<model>.csv` — the same
  predicted/observed trajectories for a single seed.
- `outputs/dinns_nec_params/<region>/seed<N>/<model>.csv` — the 7 trained
  ODE parameters (`beta_h`, `beta_m`, `gamma_h`, `lambda`, `mu_m`,
  `theta_h`, `theta_m`), snapshotted every `param_log_every` epochs
  (default 10) of training for that seed; there is no seed-averaged
  version of this file.

`FigureGeneration/visualisation.ipynb` reads these three trees (plus the
MCMC/NPC reference files alongside it) to compute calibration/projection
MSE and render the comparison and marginal-posterior plots inline — it
does not write the figures back into `outputs/`.

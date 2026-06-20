# Structural identifiability analysis design

## Problem

`FigureGeneration/visualisation.ipynb` already has a `# Structural Identifiability Analysis`
section header (with an empty code cell right after it) sitting between the Wilcoxon
comparison section and an equally empty `# Practical Identifiability Analysis` placeholder.
Neither placeholder has any code yet. This design covers only the structural one.

The dengue ODE (`model_common.dengue_rhs`) has 8 states (`Hr, Hs, He, Hi, Me, Ms, Mi, Itot`), 7
trainable parameters (`PARAM_NAMES`: `lambda, beta_m, mu_m, theta_m, beta_h, theta_h, gamma_h`),
and one fixed parameter (`mu_h`). Across all four calibration methods compared in this notebook
(MCMC, NPC, DINN, NEC), only `Itot` is ever fit against real data — the other 7 states are latent.
The MCMC/NPC baselines (notebook cells `apollo_model_7` / `nn_calibrate`) also treat the 7 initial
conditions `Hi0, Me0, Hr0, Hs0, He0, Ms0, Mi0` as unknowns to be estimated, with only `Itot0` read
directly from data. There is no existing check in this repo for whether this 14-unknown system is
even *capable* of being uniquely recovered from a single observed `Itot(t)` trajectory, regardless
of which calibration method is used.

## Goal

Add a numeric local-structural-identifiability check, per region, that answers: at generic points
in this model's actual parameter/IC search space (the same bounds DINN/NEC/MCMC/NPC are all
constrained to), is the map from the 14 unknowns to the observed `Itot(t)` trajectory locally
injective? Report a per-region verdict, and — if any region is rank-deficient — which combination
of unknowns is structurally entangled.

## Non-goals

- Practical identifiability (the next placeholder section, `# Practical Indentifiability
  Analysis`) — not touched by this change.
- Global structural identifiability via exact differential algebra (e.g. SIAN /
  `StructuralIdentifiability.jl`) — out of scope; this is a numeric generic-rank test, not a
  symbolic proof. Noted as a known limitation below.
- No changes to `dinn_model.py`, `nec_model.py`, training, or any existing notebook cell other
  than filling in the one empty cell.

## Design

### 1. New shared bounds loader — `model_common.py`

`load_param_bounds` already reads the 7 parameter bounds (rows 8-15 of `BC_<region>.mat`, natural
units). The same file's rows 0-7 hold IC bounds in **log1p space** — confirmed numerically: row 0
(`Itot0`) bounds are exactly `log1p` of each region's real `Hitotal[0]` (e.g. iquitos:
`1.79176 == log1p(5)`). Add, mirroring the existing loader:

```python
IC_NAMES: tuple[str, ...] = ("Hi0", "Me0", "Hr0", "Hs0", "He0", "Ms0", "Mi0")

_MAT_IC_ROW: dict[str, int] = {
    "Itot0": 0, "Hi0": 1, "Me0": 2, "Hr0": 3, "Hs0": 4, "He0": 5, "Ms0": 6, "Mi0": 7,
}

def load_ic_bounds(region_name, boundaries_dir=None) -> dict[str, tuple[float, float]]:
    ...  # same shape as load_param_bounds, returns bounds in log1p space, includes "Itot0"
```

`Itot0`'s bounds always satisfy `lo == hi` (it's read from data, not estimated) — callers use it
as the fixed known initial condition; `IC_NAMES` (excluding `Itot0`) are the 7 unknown ICs sampled
in step 3.

### 2. ODE simulation for the sensitivity test — notebook cell

A small `torchdiffeq.odeint` wrapper around `model_common.dengue_rhs`, structurally identical to
the existing `run_dengue_ode` already in this notebook (cell 20) but built on `dengue_rhs` (single
source of truth) instead of the notebook's separate `apollo_model_7` copy:

```python
def simulate_itot(theta: torch.Tensor, itot0_linear: torch.Tensor, t) -> torch.Tensor:
    """theta = 14-vector [7 params (natural units), 7 log1p ICs, in IC_NAMES order].

    Returns log1p(Itot(t)), shape (len(t),).
    """
    params = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
    params["mu_h"] = MU_H_FIXED
    ic_raw = dict(zip(IC_NAMES, torch.expm1(theta[len(PARAM_NAMES):])))  # log1p -> linear
    # Assemble y0 in model_common.STATE_NAMES order: (Hr, Hs, He, Hi, Me, Ms, Mi, Itot).
    y0 = torch.stack([
        ic_raw["Hr0"], ic_raw["Hs0"], ic_raw["He0"], ic_raw["Hi0"],
        ic_raw["Me0"], ic_raw["Ms0"], ic_raw["Mi0"], itot0_linear,
    ])
    sol = odeint(lambda t, y: dengue_rhs(y.unsqueeze(0), params).squeeze(0), y0, t, method="dopri5", rtol=1e-3, atol=1e-6)
    return torch.log1p(sol[:, IDX_ITOT])
```

State ordering follows `model_common.STATE_NAMES` = `(Hr, Hs, He, Hi, Me, Ms, Mi, Itot)`, not the
notebook's older `apollo_model_7` ordering — the wrapper assembles `y0` explicitly in that order
rather than relying on `IC_NAMES`' order, which is `(Hi0, Me0, Hr0, Hs0, He0, Ms0, Mi0)`.

### 3. Sampling + sensitivity-rank test — notebook cell

For each region in `REGION_KEY`:

1. `param_bounds = load_param_bounds(region)`, `ic_bounds = load_ic_bounds(region)`.
2. Fixed known IC: `itot0_linear = expm1(ic_bounds["Itot0"][0])`.
3. Draw `N_SAMPLES = 25` random `theta` vectors: each of the 7 params uniform in its natural-unit
   `(lo, hi)`; each of the 7 unknown ICs uniform in its log1p `(lo, hi)` (left in log1p space —
   `simulate_itot` converts to linear internally).
4. For each sampled `theta`, compute `J = torch.autograd.functional.jacobian(lambda th: simulate_itot(th, ...), theta)`
   — shape `(49, 14)` (49 weekly points, `t = torch.arange(0, 49)`, matching the calibration
   horizon used everywhere else in this notebook).
5. Scale column `j` of `J` by that unknown's bound width `(hi_j - lo_j)` — a semi-relative
   sensitivity: "does moving this unknown across its plausible range change the observed
   trajectory." This puts params (natural units, ranges like `0.04-1.6`) and ICs (log1p units,
   ranges like `0-3.4`) on a comparable footing for rank thresholding, and avoids the zero-scale
   problem of scaling by the (possibly-zero) sampled value itself.
6. `rank = numpy.linalg.matrix_rank(J_scaled)`. Full rank = 14 means the 14 unknowns are locally
   distinguishable from the `Itot(t)` trajectory at that sampled point.
7. Repeat over all 25 samples; record `min_rank`, `pct_full_rank` for the region.

A genuine algebraic relation between unknowns causes rank deficiency at *every* point in
parameter space, not just some; an isolated low-rank sample is therefore a numerical artifact
(e.g. landing near a sampled value that makes some term in `dengue_rhs` vanish), not evidence of
non-identifiability. So the verdict is threshold-based on `pct_full_rank`, not `min_rank`:

- `pct_full_rank >= 0.8` -> **structurally identifiable** (full rank is the generic case; a few
  sampled points dropping rank are treated as numerical edge cases).
- `pct_full_rank == 0` (deficient at *every* sample) -> **not structurally identifiable**; step 4
  below explains which unknowns are entangled.
- `0 < pct_full_rank < 0.8` -> inconclusive; print a warning and the raw per-sample rank list
  instead of a verdict, since this pattern matches neither the generic-identifiable nor the
  always-deficient signature and warrants manual inspection rather than an automatic label.

### 4. Null-space diagnosis (only runs if any region is rank-deficient)

At one representative rank-deficient sample, take the SVD of `J_scaled`, take right-singular
vectors whose singular value is below the rank tolerance, and report each as a named combination:

```
0.71 * beta_h  -  0.69 * theta_h   (singular value 2e-9)
```

i.e. round small coefficients to 0, sort by magnitude, print the top 2-3 terms — this is exactly
the standard "which parameters trade off" reading of a sensitivity-matrix null vector.

### 5. Summary table

One `pandas.DataFrame`, printed/displayed, columns:
`region, n_unknowns (14), n_obs (49), n_samples (25), min_rank, pct_full_rank, verdict`, where
`verdict` is one of `"identifiable"`, `"not identifiable"`, `"inconclusive"` per the thresholds in
step 3.

## Known limitation

This is a **numeric, sampling-based generic-rank test**, not an exact symbolic proof. It can in
principle miss a non-identifiability that only manifests on a measure-zero subset of parameter
space larger than what 25 random samples would catch (vanishingly unlikely in practice for this
kind of rational ODE system, but not impossible). An exact symbolic cross-check (e.g.
`StructuralIdentifiability.jl`'s SIAN algorithm) would close this gap but requires introducing a
Julia dependency into an otherwise all-Python repo; left as a possible future follow-up, not part
of this change.

## Verification

- `model_common.load_ic_bounds("iquitos")["Itot0"]` returns `(1.79176, 1.79176)` (sanity check
  against the manually-inspected `.mat` values above) and the tuple's two bounds are equal (it's
  fixed/known, not sampled).
- Run the new cell top to bottom for all 6 regions; confirm it completes without error and prints
  a 6-row summary table.
- Sanity check: temporarily fix one parameter's sampled value to a single constant (e.g. force
  `beta_h` to a single value instead of sampling its range) and re-run for one region — `min_rank`
  should drop by exactly 1 (collapsing the search range for an unknown is expected to make it
  invisible to the test), confirming the rank test is actually sensitive to the thing it's
  supposed to measure. Revert after checking.

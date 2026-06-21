# Structural Identifiability Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill in the empty `# Structural Identifiability Analysis` cell in
`FigureGeneration/visualisation.ipynb` with a numeric local-structural-identifiability check: for
each region, can the 7 ODE parameters + 7 unobserved initial conditions be locally distinguished
from the observed `Itot(t)` trajectory?

**Architecture:** One small shared bounds-loading addition to `model_common.py` (mirrors the
existing `load_param_bounds`), plus a self-contained analysis cell in the notebook that samples
random points from each region's own calibration bounds, computes the Jacobian of the simulated
`log1p(Itot(t))` trajectory w.r.t. all 14 unknowns via `torch.autograd.functional.jacobian`, and
checks the rank of that (bound-width-scaled) Jacobian.

**Tech Stack:** Python (`denguetwin` conda env), PyTorch, `torchdiffeq.odeint`, NumPy, pandas — all
already in use elsewhere in this notebook/repo. No new dependencies.

## Global Constraints

- Do not run any `git` commands (no `git add`, `git commit`, branches, or worktrees) —
  explicit user instruction. Implementers edit files directly and leave changes uncommitted;
  the user commits them on their own.
- Full design: [docs/superpowers/specs/2026-06-20-structural-identifiability-design.md](../specs/2026-06-20-structural-identifiability-design.md).
- This repo has no test framework (no `pytest`/`tests/` anywhere) — do not introduce one. Verify
  with small throwaway verification scripts instead (run, confirm, discard — never commit them).
- `model_common.py` is the declared single source of truth for shared model physics/bounds
  (per `README.md`); only **add** new names to it (`IC_NAMES`, `load_ic_bounds`) — do not modify
  `PARAM_NAMES`, `load_param_bounds`, or any other existing name, since `dinn_model.py` and
  `nec_model.py` import from this file.
- `FigureGeneration/visualisation.ipynb` currently has uncommitted changes (a completed Wilcoxon
  test section) — do not run `jupyter nbconvert --execute` (or anything else that re-executes and
  saves the whole notebook) against the real file. Only touch the one empty cell via the
  `NotebookEdit` tool. Validate the new cell's logic with a standalone script first; final
  in-kernel execution of the new cell is left to the user.
- New analysis code (sampling, Jacobian/rank, null-space diagnosis, summary table) lives in the
  notebook cell, not in `model_common.py` — matches this notebook's existing pattern (e.g.
  `apollo_model_7`, `dinns_nec_log1p_mse` already live in notebook cells, not shared modules).
- Use `float32` throughout (matches the rest of the notebook); do not call
  `torch.set_default_dtype` (a global side effect that would affect any earlier cell re-run later
  in the same kernel session).

---

### Task 1: Add `load_ic_bounds` to `model_common.py`

**Files:**
- Modify: `model_common.py:91` (insert new block immediately after the closing `}` of
  `load_param_bounds`, before the `# Feature extractor` section header at line 94).

**Interfaces:**
- Consumes: nothing new — reuses `_BOUNDARIES_DIR` (`model_common.py:18`), already-imported
  `numpy as np` and `Path`.
- Produces: `IC_NAMES: tuple[str, ...]` (7 names: `Hi0, Me0, Hr0, Hs0, He0, Ms0, Mi0`) and
  `load_ic_bounds(region_name: str, boundaries_dir: str | Path | None = None) -> dict[str, tuple[float, float]]`
  — same shape as the existing `load_param_bounds`, but for IC bounds (log1p space), and its
  returned dict additionally includes `"Itot0"` (always `lo == hi` — fixed/known from data).
  Task 2 imports both directly: `from model_common import IC_NAMES, load_ic_bounds`.

- [ ] **Step 1: Write the failing verification script**

Create `/tmp/verify_ic_bounds.py`:

```python
import sys

sys.path.insert(0, "/home/hoang/Documents/VGU_projects/dengue_vgu/code/nec_dinns")

from model_common import IC_NAMES, load_ic_bounds

assert IC_NAMES == ("Hi0", "Me0", "Hr0", "Hs0", "He0", "Ms0", "Mi0"), IC_NAMES

bounds = load_ic_bounds("iquitos")
assert set(bounds) == {"Itot0", "Hi0", "Me0", "Hr0", "Hs0", "He0", "Ms0", "Mi0"}, bounds.keys()

lo, hi = bounds["Itot0"]
assert lo == hi, "Itot0 should be fixed (known from data), not a range"
assert abs(lo - 1.791759) < 1e-4, f"expected log1p(5) ~= 1.791759, got {lo}"  # iquitos Hitotal[0] == 5

lo, hi = bounds["He0"]
assert abs(lo - 0.226878) < 1e-4 and abs(hi - 3.19427) < 1e-4, bounds["He0"]

print("OK")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/home/hoang/anaconda3/envs/denguetwin/bin/python /tmp/verify_ic_bounds.py`
Expected: `ImportError: cannot import name 'IC_NAMES' from 'model_common'`

- [ ] **Step 3: Implement `IC_NAMES` and `load_ic_bounds`**

In `model_common.py`, insert immediately after line 91 (the `}` closing `load_param_bounds`,
before the blank lines and `# Feature extractor` section header):

```python

# Row index of each *initial condition* inside the BC .mat array (shape 16x2),
# in log1p space (confirmed numerically: row 0 == log1p of each region's real
# Hitotal[0]). Itot0 is always a fixed point (lo == hi) -- it is read directly
# from data, never sampled/estimated; the other 7 rows are genuinely unknown.
_MAT_IC_ROW: dict[str, int] = {
    "Itot0": 0,
    "Hi0":   1,
    "Me0":   2,
    "Hr0":   3,
    "Hs0":   4,
    "He0":   5,
    "Ms0":   6,
    "Mi0":   7,
}

# The 7 unobserved initial conditions. Itot0 is excluded here (it's observed
# directly from data) but is still present in load_ic_bounds()'s return value.
IC_NAMES: tuple[str, ...] = ("Hi0", "Me0", "Hr0", "Hs0", "He0", "Ms0", "Mi0")


def load_ic_bounds(
    region_name: str,
    boundaries_dir: str | Path | None = None,
) -> dict[str, tuple[float, float]]:
    """Return per-region initial-condition bounds from BC_{region_name}.mat, in log1p space.

    Includes "Itot0", whose bounds always satisfy lo == hi (read directly from
    data, not estimated). Raises FileNotFoundError if the .mat file is missing.
    """
    import scipy.io

    mat_path = (
        Path(boundaries_dir) if boundaries_dir is not None else _BOUNDARIES_DIR
    ) / f"BC_{region_name}.mat"

    if not mat_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {mat_path}")

    arr = scipy.io.loadmat(str(mat_path), squeeze_me=True)
    data_key = next(k for k in arr if not k.startswith("__"))
    mat = np.asarray(arr[data_key], dtype=np.float64)  # shape (16, 2)

    return {
        name: (float(mat[row_idx, 0]), float(mat[row_idx, 1]))
        for name, row_idx in _MAT_IC_ROW.items()
    }
```

- [ ] **Step 4: Run the verification script again to confirm it passes**

Run: `/home/hoang/anaconda3/envs/denguetwin/bin/python /tmp/verify_ic_bounds.py`
Expected: `OK`

- [ ] **Step 5: Discard the scratch script**

```bash
rm /tmp/verify_ic_bounds.py
```

Do not run any `git` commands (no `git add`, `git commit`, etc.) — per explicit user
instruction, leave the change uncommitted in the working tree; committing is the user's own
action to take, not the implementer's or controller's.

---

### Task 2: Structural identifiability cell

**Files:**
- Create (scratch, never committed): `/tmp/si_check.py` — used only to validate the logic before
  it goes in the notebook.
- Modify: `FigureGeneration/visualisation.ipynb`, cell with `id` matching the currently-empty code
  cell that immediately follows the `# Structural Identifiability Analysis` markdown header (cell
  index 61 as of this plan — confirm by reading the notebook; the markdown header cell right
  before it has source `"# Structural Identifiability Analysis"`).

**Interfaces:**
- Consumes: `IC_NAMES`, `load_ic_bounds` (Task 1); `model_common.PARAM_NAMES`, `IDX_ITOT`,
  `dengue_rhs` (already exist); `model_common.load_param_bounds` (already exists); the notebook's
  existing global `REGIONS_ORDERED` (defined in the Wilcoxon section, cell 58 — the list
  `["bello", "iquitos", "sanjuan", "vietnam", "philippines", "cambodia"]`, matching the
  `boundaries/BC_<region>.mat` filenames exactly).
- Produces: nothing consumed by later cells — this is the last new logic before the (separately
  out-of-scope) Practical Identifiability placeholder. Prints a `pandas.DataFrame` named
  `si_summary` and, for any region whose verdict isn't `"identifiable"`, prints its null-space
  diagnosis.

- [ ] **Step 1: Write the analysis logic as a standalone script**

Create `/tmp/si_check.py` (run from inside `FigureGeneration/` so `sys.path.insert(0, "..")`
resolves the same way it will inside the notebook):

```python
import sys

sys.path.insert(0, "..")

import time

import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint

from model_common import IC_NAMES, IDX_ITOT, PARAM_NAMES, dengue_rhs, load_ic_bounds, load_param_bounds

REGIONS_ORDERED = ["bello", "iquitos", "sanjuan", "vietnam", "philippines", "cambodia"]
N_SI_SAMPLES = 25
SI_T = torch.arange(0, 49, dtype=torch.float32)
SI_UNKNOWN_NAMES = list(PARAM_NAMES) + list(IC_NAMES)
N_SI_UNKNOWNS = len(SI_UNKNOWN_NAMES)


def simulate_log_itot(theta: torch.Tensor, itot0_linear: torch.Tensor) -> torch.Tensor:
    """theta = [7 params (natural units), 7 log1p ICs in IC_NAMES order] -> log1p(Itot(t))."""
    params = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
    ic = dict(zip(IC_NAMES, torch.expm1(theta[len(PARAM_NAMES):])))
    y0 = torch.stack([
        ic["Hr0"], ic["Hs0"], ic["He0"], ic["Hi0"],
        ic["Me0"], ic["Ms0"], ic["Mi0"], itot0_linear,
    ])
    sol = odeint(
        lambda t, y: dengue_rhs(y.unsqueeze(0), params).squeeze(0),
        y0, SI_T, method="dopri5", rtol=1e-3, atol=1e-6,
    )
    return torch.log1p(sol[:, IDX_ITOT])


def describe_null_space(J_scaled: np.ndarray, names: list, tol: float = 1e-6) -> list:
    """Named unknown-combinations spanning J_scaled's (near-)null space."""
    _, S, Vt = np.linalg.svd(J_scaled, full_matrices=False)
    out = []
    for sv, v in zip(S, Vt):
        if sv > tol * S[0]:
            continue
        order = np.argsort(-np.abs(v))[:3]
        terms = [f"{v[i]:+.2f}*{names[i]}" for i in order if abs(v[i]) > 0.05]
        out.append(f"{'  '.join(terms)}   (singular value {sv:.1e})")
    return out


def region_rank_test(region: str, rng: np.random.Generator) -> dict:
    param_bounds = load_param_bounds(region)
    ic_bounds = load_ic_bounds(region)

    itot0_lo, itot0_hi = ic_bounds["Itot0"]
    assert itot0_lo == itot0_hi, f"{region}: Itot0 bound should be fixed (known from data)"
    itot0_linear = torch.tensor(float(np.expm1(itot0_lo)), dtype=torch.float32)

    lo = torch.tensor(
        [param_bounds[n][0] for n in PARAM_NAMES] + [ic_bounds[n][0] for n in IC_NAMES],
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [param_bounds[n][1] for n in PARAM_NAMES] + [ic_bounds[n][1] for n in IC_NAMES],
        dtype=torch.float32,
    )
    scale = (hi - lo).clamp(min=1e-8)  # bound width -- the sensitivity normalization (see design doc)

    ranks = []
    last_J_scaled = None
    for _ in range(N_SI_SAMPLES):
        theta0 = torch.tensor(rng.uniform(lo.numpy(), hi.numpy()), dtype=torch.float32)
        J = torch.autograd.functional.jacobian(
            lambda theta: simulate_log_itot(theta, itot0_linear), theta0,
        )
        J_scaled = (J * scale).numpy()
        ranks.append(np.linalg.matrix_rank(J_scaled))
        last_J_scaled = J_scaled

    ranks = np.array(ranks)
    pct_full_rank = float((ranks == N_SI_UNKNOWNS).mean())
    if pct_full_rank >= 0.8:
        verdict = "identifiable"
    elif pct_full_rank == 0.0:
        verdict = "not identifiable"
    else:
        verdict = "inconclusive"

    return {
        "region": region, "n_unknowns": N_SI_UNKNOWNS, "n_obs": len(SI_T),
        "n_samples": N_SI_SAMPLES, "min_rank": int(ranks.min()),
        "pct_full_rank": pct_full_rank, "verdict": verdict,
        "last_J_scaled": last_J_scaled,
    }


if __name__ == "__main__":
    t0 = time.time()
    _si_rng = np.random.default_rng(0)
    si_rows = []
    for region in REGIONS_ORDERED:
        result = region_rank_test(region, _si_rng)
        si_rows.append({k: v for k, v in result.items() if k != "last_J_scaled"})
        if result["verdict"] != "identifiable":
            print(f"-- {region}: rank-deficient sample(s), null-space directions --")
            for line in describe_null_space(result["last_J_scaled"], SI_UNKNOWN_NAMES):
                print("  ", line)

    si_summary = pd.DataFrame(si_rows)
    print(si_summary)
    print(f"elapsed: {time.time() - t0:.1f}s")
```

- [ ] **Step 2: Run it and inspect timing + the summary table**

Run: `cd FigureGeneration && /home/hoang/anaconda3/envs/denguetwin/bin/python /tmp/si_check.py`

Expected: prints a 6-row table with columns `region, n_unknowns, n_obs, n_samples, min_rank,
pct_full_rank, verdict`, `n_unknowns == 14` and `n_obs == 49` for every row, and an `elapsed`
under ~2 minutes. If `elapsed` is much higher than that, add `vectorize=True` to the
`torch.autograd.functional.jacobian(...)` call (re-run to confirm it still produces the same
ranks) before moving on — do not silently reduce `N_SI_SAMPLES` or the time grid without noting
why in the cell's comments, since both were explicit choices in the approved design.

- [ ] **Step 3: Sanity-check the rank test is actually sensitive to what it measures**

Append a temporary check to the bottom of `/tmp/si_check.py` (this block is throwaway — it exists
only to validate the test method, and is never copied into the notebook):

```python
    # Sanity check: collapsing one unknown's range to a point should drop
    # min_rank by exactly 1 for that region (it becomes invisible to the test).
    bounds = load_param_bounds("iquitos")
    fixed_value = sum(bounds["beta_h"]) / 2
    bounds["beta_h"] = (fixed_value, fixed_value)
    import model_common
    model_common.load_param_bounds = lambda region, boundaries_dir=None: bounds if region == "iquitos" else load_param_bounds(region, boundaries_dir)
    sanity_result = region_rank_test("iquitos", np.random.default_rng(0))
    print("sanity min_rank with beta_h fixed:", sanity_result["min_rank"], "(expect original_min_rank - 1)")
```

Run: `cd FigureGeneration && /home/hoang/anaconda3/envs/denguetwin/bin/python /tmp/si_check.py`
Expected: the printed `sanity min_rank with beta_h fixed` is exactly 1 less than iquitos's
`min_rank` from Step 2's table. If it isn't, the scaling or sampling logic has a bug — fix it in
the function definitions above and re-run Steps 2-3 before proceeding.

- [ ] **Step 4: Read the notebook and locate the target cell**

Use the Read tool on `FigureGeneration/visualisation.ipynb`. Find the markdown cell whose source
is `# Structural Identifiability Analysis`; note the `id` of the empty code cell immediately
after it (the design's placeholder — currently empty source, around index 61).

- [ ] **Step 5: Insert the validated code into that cell**

Use `NotebookEdit` with `edit_mode="replace"`, the `cell_id` found in Step 4, and `new_source` set
to the **finalized** version of the code from Step 1 (drop the `if __name__ == "__main__":` guard
and the `time` import/timing prints — those were only for the standalone script — and drop the
Step 3 sanity-check block entirely, since it's a one-off validation, not part of the deliverable).
The final cell source:

```python
# Numeric local-structural-identifiability check: at random points drawn from
# each region's own calibration bounds, is the map from the 14 unknowns (7 ODE
# parameters + 7 unobserved initial conditions) to the observed Itot(t)
# trajectory locally injective? Full rank at (nearly) every sampled point means
# yes, generically; rank-deficient at *every* sampled point would indicate a
# genuine structural relation between unknowns.
import sys

sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint

from model_common import IC_NAMES, IDX_ITOT, PARAM_NAMES, dengue_rhs, load_ic_bounds, load_param_bounds

N_SI_SAMPLES = 25
SI_T = torch.arange(0, 49, dtype=torch.float32)
SI_UNKNOWN_NAMES = list(PARAM_NAMES) + list(IC_NAMES)
N_SI_UNKNOWNS = len(SI_UNKNOWN_NAMES)


def simulate_log_itot(theta: torch.Tensor, itot0_linear: torch.Tensor) -> torch.Tensor:
    """theta = [7 params (natural units), 7 log1p ICs in IC_NAMES order] -> log1p(Itot(t))."""
    params = {name: theta[i] for i, name in enumerate(PARAM_NAMES)}
    ic = dict(zip(IC_NAMES, torch.expm1(theta[len(PARAM_NAMES):])))
    y0 = torch.stack([
        ic["Hr0"], ic["Hs0"], ic["He0"], ic["Hi0"],
        ic["Me0"], ic["Ms0"], ic["Mi0"], itot0_linear,
    ])
    sol = odeint(
        lambda t, y: dengue_rhs(y.unsqueeze(0), params).squeeze(0),
        y0, SI_T, method="dopri5", rtol=1e-3, atol=1e-6,
    )
    return torch.log1p(sol[:, IDX_ITOT])


def describe_null_space(J_scaled: np.ndarray, names: list, tol: float = 1e-6) -> list:
    """Named unknown-combinations spanning J_scaled's (near-)null space."""
    _, S, Vt = np.linalg.svd(J_scaled, full_matrices=False)
    out = []
    for sv, v in zip(S, Vt):
        if sv > tol * S[0]:
            continue
        order = np.argsort(-np.abs(v))[:3]
        terms = [f"{v[i]:+.2f}*{names[i]}" for i in order if abs(v[i]) > 0.05]
        out.append(f"{'  '.join(terms)}   (singular value {sv:.1e})")
    return out


def region_rank_test(region: str, rng: np.random.Generator) -> dict:
    param_bounds = load_param_bounds(region)
    ic_bounds = load_ic_bounds(region)

    itot0_lo, itot0_hi = ic_bounds["Itot0"]
    assert itot0_lo == itot0_hi, f"{region}: Itot0 bound should be fixed (known from data)"
    itot0_linear = torch.tensor(float(np.expm1(itot0_lo)), dtype=torch.float32)

    lo = torch.tensor(
        [param_bounds[n][0] for n in PARAM_NAMES] + [ic_bounds[n][0] for n in IC_NAMES],
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [param_bounds[n][1] for n in PARAM_NAMES] + [ic_bounds[n][1] for n in IC_NAMES],
        dtype=torch.float32,
    )
    scale = (hi - lo).clamp(min=1e-8)  # bound width -- the sensitivity normalization (see design doc)

    ranks = []
    last_J_scaled = None
    for _ in range(N_SI_SAMPLES):
        theta0 = torch.tensor(rng.uniform(lo.numpy(), hi.numpy()), dtype=torch.float32)
        J = torch.autograd.functional.jacobian(
            lambda theta: simulate_log_itot(theta, itot0_linear), theta0,
        )
        J_scaled = (J * scale).numpy()
        ranks.append(np.linalg.matrix_rank(J_scaled))
        last_J_scaled = J_scaled

    ranks = np.array(ranks)
    pct_full_rank = float((ranks == N_SI_UNKNOWNS).mean())
    if pct_full_rank >= 0.8:
        verdict = "identifiable"
    elif pct_full_rank == 0.0:
        verdict = "not identifiable"
    else:
        verdict = "inconclusive"

    return {
        "region": region, "n_unknowns": N_SI_UNKNOWNS, "n_obs": len(SI_T),
        "n_samples": N_SI_SAMPLES, "min_rank": int(ranks.min()),
        "pct_full_rank": pct_full_rank, "verdict": verdict,
        "last_J_scaled": last_J_scaled,
    }


_si_rng = np.random.default_rng(0)
si_rows = []
for region in REGIONS_ORDERED:
    result = region_rank_test(region, _si_rng)
    si_rows.append({k: v for k, v in result.items() if k != "last_J_scaled"})
    if result["verdict"] != "identifiable":
        print(f"-- {region}: rank-deficient sample(s), null-space directions --")
        for line in describe_null_space(result["last_J_scaled"], SI_UNKNOWN_NAMES):
            print("  ", line)

si_summary = pd.DataFrame(si_rows)
si_summary
```

(If Step 2 required `vectorize=True` for acceptable runtime, include it here too:
`torch.autograd.functional.jacobian(..., theta0, vectorize=True)`.)

- [ ] **Step 6: Discard the scratch script**

```bash
rm /tmp/si_check.py
```

- [ ] **Step 7: Confirm the inserted cell's source matches**

Use Read on `FigureGeneration/visualisation.ipynb` to display the cell you just edited; confirm
its source matches Step 5 exactly (no leftover `__main__` guard, no sanity-check block, no timing
code). Do not run any `git` commands — per explicit user instruction, leave the change
uncommitted; committing is the user's own action to take.

- [ ] **Step 8: Hand off to the user for in-kernel verification**

Tell the user: open `FigureGeneration/visualisation.ipynb` in their live kernel (where
`REGIONS_ORDERED` from the Wilcoxon section above is already defined) and run the new cell;
confirm it produces the same kind of summary table seen in Step 2's standalone run. Do not
execute the full notebook from the agent side — it has other uncommitted, already-computed
sections (e.g. the Wilcoxon results) that must not be disturbed.

## Self-Review Notes

- Spec coverage: shared bounds loader (Task 1, design section 1); ODE wrapper + sampling +
  rank/verdict logic (Task 2, design sections 2-3); null-space diagnosis (Task 2, design section
  4); summary table with `verdict` column (Task 2, design section 5, matches the corrected
  3-way verdict from the spec self-review). Known-limitation note from the design doc is preserved
  implicitly (no symbolic cross-check is implemented) — nothing left to do for it, it's
  documentation only.
- No placeholders: every step has runnable code; the only "TBD"-shaped item (exact cell `id` and
  cell index) is explicitly resolved by a Read-the-notebook step before the edit, not deferred.
- Type/name consistency checked: `simulate_log_itot`, `describe_null_space`, `region_rank_test`,
  `N_SI_SAMPLES`, `SI_T`, `SI_UNKNOWN_NAMES`, `N_SI_UNKNOWNS` are spelled identically between the
  standalone script (Step 1) and the final cell (Step 5); `IC_NAMES`/`load_ic_bounds` names match
  exactly what Task 1 produces.

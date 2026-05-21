# NEC-DINNs Dengue Baseline

Main files:

- `baseline_experiments.py` contains the DINN/NEC models, NPC data loader,
  training loop, metrics, and prediction export.
- `run_baseline_table.py` runs experiments and writes all plots.
- `baseline_all_results.ipynb` is the notebook interface for full sweeps and
  reporting.

Example smoke run:

```bash
conda run -n dinn python run_baseline_table.py \
  --epochs 2 --regions iquitos --models dinn nec --quiet
```

Default outputs:

- `outputs/baseline_results.csv`
- `outputs/baseline_predictions.csv`
- `outputs/plots/*.png`

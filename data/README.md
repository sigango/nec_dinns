# NPC Dengue Data

These CSVs are downloaded from the public repository for Pham et al. 2026,
*Neural Parameter Calibration for dengue*:

https://github.com/Hoangpham13579/neural-parameter-calibration-dengue

This project expects the 12 calibration/projection CSV files already present
under `data/<region>/`. They are the files consumed by
`baseline_experiments.py`.

The NPC CSVs used here contain only `CumulativeHumanInfected` in practice. City
files have a `HumanInfected` column, but it is blank in this clone. The loader
therefore derives weekly incidence as:

```text
incidence[0] = CumulativeHumanInfected[0]
incidence[t] = max(CumulativeHumanInfected[t] - CumulativeHumanInfected[t-1], 0)
```

This derived signal is used for the soft incidence loss and incidence reporting;
it is not an independent observation of the ODE `Hi` compartment.

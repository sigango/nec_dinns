from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from baseline_experiments import (
    BaselineConfig,
    default_output_dir,
    discover_regions,
    run_all_baseline_table,
)


CITY_REGIONS = ["bello", "iquitos", "sanjuan"]
COUNTRY_REGIONS = ["cam", "phi", "vn"]
TARGET_TITLES = {
    "Itot": "Cumulative Infectious Cases (Hitotal)",
    "infectious_case": "Weekly Infectious Cases",
}


def parse_args() -> argparse.Namespace:
    defaults = BaselineConfig()
    parser = argparse.ArgumentParser(
        description="Run DINN/NEC dengue baselines and create result plots."
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Training epochs.")
    parser.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Optional region subset, e.g. bello phi vn. Defaults to discovered NPC data.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["dinn", "nec"],
        choices=["dinn", "nec"],
        help="Model types to run or plot.",
    )
    parser.add_argument(
        "--w-phys",
        type=float,
        default=defaults.w_phys,
        help="Fixed physics-loss weight (V32: 3.0).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir() / "baseline_results.csv",
        help="CSV path for the combined metric table.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=default_output_dir() / "baseline_predictions.csv",
        help="CSV path for calibration/projection prediction curves.",
    )
    parser.add_argument(
        "--loss-history-output",
        type=Path,
        default=default_output_dir() / "baseline_loss_history.csv",
        help="CSV path for per-epoch training losses.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=default_output_dir() / "plots",
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--plot-style",
        choices=["group", "split", "infectious", "score", "all"],
        default="all",
        help="Which plots to write after training or in --plot-only mode.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training and create plots from existing CSV files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run training but skip figure generation.",
    )
    return parser.parse_args()


def _normalize_prediction_targets(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.copy()


def _region_order(predictions: pd.DataFrame) -> list[str]:
    discovered = discover_regions()
    present = list(dict.fromkeys(predictions["region"].tolist()))
    return [r for r in discovered if r in present] + [r for r in present if r not in discovered]


def plot_prediction_groups(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    models: list[str],
) -> list[Path]:
    predictions = _normalize_prediction_targets(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    model_styles = {
        "dinn": {"label": "DINN", "color": "C0", "marker": "s"},
        "nec": {"label": "NEC", "color": "C1", "marker": "^"},
    }
    region_groups = [
        ("cities", [r for r in CITY_REGIONS if r in set(predictions["region"])]),
        ("countries", [r for r in COUNTRY_REGIONS if r in set(predictions["region"])]),
    ]

    for split in ["calibration", "projection"]:
        for group_name, group_regions in region_groups:
            if not group_regions:
                continue
            fig, axes = plt.subplots(
                len(TARGET_TITLES),
                len(group_regions),
                figsize=(5.8 * len(group_regions), 8.2),
                squeeze=False,
            )
            for row_idx, target in enumerate(TARGET_TITLES):
                for col_idx, region in enumerate(group_regions):
                    ax = axes[row_idx, col_idx]
                    obs = predictions[
                        predictions["region"].eq(region)
                        & predictions["split"].eq(split)
                        & predictions["target"].eq(target)
                    ].copy()
                    if obs.empty:
                        ax.axis("off")
                        continue

                    label = obs["region_label"].iloc[0]
                    truth = obs.sort_values(["t", "model"]).drop_duplicates("t").sort_values("t")
                    ax.plot(
                        truth["t"],
                        truth["observed"],
                        color="C3",
                        marker="o",
                        markersize=3.3,
                        lw=1.6,
                        label="Ground Truth",
                    )
                    for model in models:
                        sub = obs[obs["model"].eq(model)].sort_values("t")
                        if sub.empty:
                            continue
                        style = model_styles.get(
                            model,
                            {"label": model.upper(), "color": None, "marker": None},
                        )
                        ax.plot(
                            sub["t"],
                            sub["predicted"],
                            color=style["color"],
                            marker=style["marker"],
                            markersize=3.0,
                            lw=1.7,
                            label=style["label"],
                        )
                    ax.set_title(label)
                    ax.set_xlabel("Week")
                    if col_idx == 0:
                        ax.set_ylabel(TARGET_TITLES[target])
                    ax.minorticks_on()
                    ax.grid(which="major", linestyle="--", alpha=0.32)
                    ax.grid(which="minor", linestyle=":", alpha=0.14)

            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.suptitle(f"{split.title()} prediction curves - {group_name}", y=0.99)
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.945),
                ncol=max(len(labels), 1),
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            out_path = output_dir / f"{split}_{group_name}_dinn_nec.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written.append(out_path)
    return written


def plot_split_grids(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    models: list[str],
) -> list[Path]:
    predictions = _normalize_prediction_targets(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in models:
        for target in TARGET_TITLES:
            for split in ["calibration", "projection"]:
                subset = predictions[
                    predictions["model"].eq(model)
                    & predictions["target"].eq(target)
                    & predictions["split"].eq(split)
                ].copy()
                if subset.empty:
                    continue
                regions = _region_order(subset)
                fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
                for ax, region in zip(axes.ravel(), regions):
                    obs = subset[subset["region"].eq(region)].sort_values("t")
                    if obs.empty:
                        ax.axis("off")
                        continue
                    ax.plot(
                        obs["t"],
                        obs["observed"],
                        color="C3",
                        marker="o",
                        markersize=3.0,
                        lw=1.5,
                        label="Ground Truth",
                    )
                    ax.plot(
                        obs["t"],
                        obs["predicted"],
                        color="C0",
                        lw=2.0,
                        label="Predicted",
                    )
                    ax.set_title(obs["region_label"].iloc[0])
                    ax.set_xlabel("Week")
                    ax.grid(alpha=0.25)
                for ax in axes.ravel()[len(regions):]:
                    ax.axis("off")

                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.suptitle(f"{model.upper()} {split} - {TARGET_TITLES[target]}", y=0.99)
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.945),
                    ncol=2,
                    frameon=False,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.88))
                out_path = output_dir / f"{model}_{split}_{target}.png"
                fig.savefig(out_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                written.append(out_path)
    return written


def plot_infectious_case_comparison(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    models: list[str],
) -> list[Path]:
    predictions = _normalize_prediction_targets(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in models:
        incidence = predictions[
            predictions["target"].eq("infectious_case") & predictions["model"].eq(model)
        ].copy()
        if incidence.empty:
            continue
        regions = _region_order(incidence)
        ncols = 3
        nrows = (len(regions) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.5 * ncols, 3.5 * nrows),
            squeeze=False,
        )
        for ax, region in zip(axes.ravel(), regions):
            df = incidence[incidence["region"].eq(region)]
            cal = df[df["split"].eq("calibration")].sort_values("t")
            proj = df[df["split"].eq("projection")].sort_values("t")
            label = df["region_label"].iloc[0]
            ax.plot(cal["t"], cal["observed"], color="C3", lw=1.5, label="Ground Truth cal")
            ax.plot(cal["t"], cal["predicted"], color="C0", lw=1.8, label="Predicted cal")
            ax.plot(proj["t"], proj["observed"], color="0.35", lw=1.5, label="Ground Truth proj")
            ax.plot(proj["t"], proj["predicted"], color="C1", lw=1.8, label="Predicted proj")
            ax.set_title(f"{label} - infectious case")
            ax.set_xlabel("Week")
            ax.set_ylabel("Weekly cases")
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(regions):]:
            ax.axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f"Infectious case comparison ({model.upper()})", y=0.99)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=4,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        out_path = output_dir / f"infectious_case_comparison_{model}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def plot_scores_and_runtime(results: pd.DataFrame, *, output_dir: Path) -> list[Path]:
    if results.empty:
        return []
    results = results.copy()
    runtime_df = results.drop_duplicates(["region", "model"]).copy()
    if "mse_normalized" not in results.columns or "train_seconds" not in runtime_df.columns:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    score_pivot = results.pivot_table(
        index="region",
        columns="model",
        values="mse_normalized",
        aggfunc="mean",
    ).reindex(_region_order(results))
    score_pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("projection Itot")
    axes[0].set_ylabel("Normalized MSE")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", alpha=0.25)

    runtime_pivot = runtime_df.pivot_table(
        index="region",
        columns="model",
        values="train_seconds",
        aggfunc="mean",
    ).reindex(_region_order(runtime_df))
    runtime_pivot.plot(kind="bar", ax=axes[1])
    axes[1].set_title("runtime")
    axes[1].set_ylabel("training seconds")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Projection scores and runtime", y=1.03)
    fig.tight_layout()

    out_path = output_dir / "projection_scores_runtime.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def write_plots(
    *,
    results_csv: Path,
    predictions_csv: Path,
    output_dir: Path,
    plot_style: str,
    models: list[str],
) -> list[Path]:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {predictions_csv}")
    predictions = pd.read_csv(predictions_csv)
    written: list[Path] = []
    if plot_style in {"group", "all"}:
        written.extend(plot_prediction_groups(predictions, output_dir=output_dir, models=models))
    if plot_style in {"split", "all"}:
        written.extend(plot_split_grids(predictions, output_dir=output_dir, models=models))
    if plot_style in {"infectious", "all"}:
        written.extend(plot_infectious_case_comparison(predictions, output_dir=output_dir, models=models))
    if plot_style in {"score", "all"} and results_csv.exists():
        written.extend(plot_scores_and_runtime(pd.read_csv(results_csv), output_dir=output_dir))
    return written


def simple_score_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["region", "model", "mse_normalized", "train_seconds"])
    simple = results[["region", "model", "mse_normalized", "train_seconds"]].copy()
    return simple.sort_values(["region", "model"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    regions = args.regions if args.regions is not None else discover_regions()
    if not regions:
        raise SystemExit("No NPC data found under data_piml/npc.")

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
            output_csv=args.output,
            prediction_csv=args.predictions_output,
            loss_history_csv=args.loss_history_output,
            verbose=not args.quiet,
        )
        if args.quiet:
            print(f"wrote results table -> {args.output}")
            print(f"wrote prediction table -> {args.predictions_output}")
            print(f"wrote loss history -> {args.loss_history_output}")
        else:
            print(simple_score_table(table).to_string(
                index=False,
                float_format=lambda x: f"{x:.4e}",
            ))

    if not args.no_plots:
        written = write_plots(
            results_csv=args.output,
            predictions_csv=args.predictions_output,
            output_dir=args.plot_dir,
            plot_style=args.plot_style,
            models=args.models,
        )
        if written:
            for path in written:
                print(f"wrote {path}")
        elif args.plot_only:
            raise SystemExit("No plots were written; check the CSV filters.")


if __name__ == "__main__":
    main()

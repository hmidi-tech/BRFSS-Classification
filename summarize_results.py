"""Derive ablations, uncertainty diagnostics, and report figures from saved runs."""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from data_io import write_json
from evaluation import metrics, stratified_split


def recompute(results_dir, data_dir, artifact_dir, output_dir, config):
    """Recompute into a fresh directory using local, untracked score arrays."""
    results_dir, artifact_dir, output_dir = map(
        Path, [results_dir, artifact_dir, output_dir]
    )
    if output_dir.exists():
        raise FileExistsError("Choose a fresh aggregate output directory")
    rows = json.loads((results_dir / "leaderboard.json").read_text())
    by_name = {r["config"]["name"]: r for r in rows}
    y = np.loadtxt(
        Path(data_dir) / "y_train.csv", delimiter=",", skiprows=1, usecols=1, dtype=int
    )
    _, tuning, _ = stratified_split(y, config["seed"])
    tuning_y = y[tuning]
    fixed = {}
    for name, row in by_name.items():
        stem = artifact_dir / "candidates" / Path(row["artifact_stem"]).name
        scores = np.load(str(stem) + ".npz", allow_pickle=False)["tuning_scores"]
        fixed[name] = metrics(
            tuning_y, scores, 0.2, row["config"]["family"] == "logistic"
        )
    ablations = []
    for previous, following, change in [
        ("A0_mean", "A1_median", "mean to median imputation"),
        ("A1_median", "A2_missing", "add missingness indicators"),
        (
            "A2_missing",
            "A3_categories",
            "encode integer features with at most 12 observed levels",
        ),
        ("A3_categories", "A3_l2_0.0001", "reduce L2 lambda from 0.001 to 0.0001"),
        ("A3_l2_0.0001", "A3_l2_0", "remove L2 regularization"),
    ]:
        ablations.append(
            {
                "change": change,
                "before": previous,
                "after": following,
                "fixed_threshold": 0.2,
                "f1_before": fixed[previous]["f1"],
                "f1_after": fixed[following]["f1"],
                "f1_delta": fixed[following]["f1"] - fixed[previous]["f1"],
                "accuracy_delta": fixed[following]["accuracy"]
                - fixed[previous]["accuracy"],
                "scope": "tuning; same rows, optimizer budget, threshold and all unrelated settings",
            }
        )
    output_dir.mkdir(parents=True)
    with (output_dir / "ablation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablations[0]))
        writer.writeheader()
        writer.writerows(ablations)
    held = json.loads((results_dir / "holdout.json").read_text())
    selected = held["selected"]
    counts = np.array([selected[k] for k in ["tp", "fp", "fn", "tn"]])
    rng = np.random.default_rng(433)
    draws = rng.multinomial(counts.sum(), counts / counts.sum(), size=2000)
    bootstrap_f1 = 2 * draws[:, 0] / (2 * draws[:, 0] + draws[:, 1] + draws[:, 2])
    cv = json.loads((results_dir / "cross_validation.json").read_text())
    cv_f1 = np.array([fold["metrics"]["f1"] for fold in cv["folds"]])
    summary = {
        "holdout": held,
        "holdout_f1_iid_bootstrap_95_percent_interval": np.quantile(
            bootstrap_f1, [0.025, 0.975]
        ).tolist(),
        "uncertainty_scope": "2000 multinomial resamples of the fixed classifier's confusion counts; assumes independent survey rows; ignores complex survey design and model-selection variability",
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_sample_std": float(cv_f1.std(ddof=1)),
        "cv_fold_f1": cv_f1.tolist(),
        "cv_scope": cv["scope"],
        "fixed_threshold_metrics": fixed,
        "threshold_delta_holdout_f1": selected["f1"] - held["selected_at_0_5"]["f1"],
        "threshold_delta_holdout_accuracy": selected["accuracy"]
        - held["selected_at_0_5"]["accuracy"],
        "ablations": ablations,
    }
    write_json(output_dir / "validation_summary.json", summary)
    candidate = artifact_dir / "candidates" / Path(config["candidate_artifact"]).name
    candidate_data = json.loads(Path(str(candidate) + ".json").read_text())
    curve = {
        "experiment_id": config["experiment_id"],
        "rows": candidate_data["threshold_curve"],
    }
    write_json(output_dir / "selected_threshold_curve.json", curve)
    return summary, curve


def plot_figures(summary, curve, config, results_dir, figure_dir):
    """Produce figures using tracked aggregate evidence, without row data."""
    results_dir, figure_dir = Path(results_dir), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if curve["experiment_id"] != config["experiment_id"]:
        raise ValueError("Threshold curve and selected configuration disagree")
    # Matplotlib is used only to produce figures, never for model fitting.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cs433-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    plt.rcParams.update({"font.size": 11})
    curve = curve["rows"]
    fig, ax = plt.subplots(figsize=(4.6, 2.35), layout="constrained")
    ax.plot(
        [r["threshold"] for r in curve],
        [r["f1"] for r in curve],
        label="Tuning F1",
        color="#146b8c",
    )
    ax.plot(
        [r["threshold"] for r in curve],
        [r["accuracy"] for r in curve],
        label="Tuning accuracy",
        color="#b45929",
    )
    ax.axvline(config["threshold"], color="0.4", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Probability threshold", ylabel="Score", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(figure_dir / "threshold_tradeoff.pdf")
    fig.savefig(figure_dir / "threshold_tradeoff.png", dpi=180)
    plt.close(fig)
    with (results_dir / "feature_audit.csv").open() as handle:
        missing = list(csv.DictReader(handle))
    top = sorted(
        missing, key=lambda r: float(r["train_missing_fraction"]), reverse=True
    )[:12]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.barh(
        [r["feature"] for r in reversed(top)],
        [float(r["train_missing_fraction"]) * 100 for r in reversed(top)],
        color="#146b8c",
    )
    ax.set(
        xlabel="Missing training entries (%)",
        title="Raw missingness; no global survey-code replacement",
    )
    fig.savefig(figure_dir / "missingness.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "cv_f1_mean",
                    "cv_f1_sample_std",
                    "holdout_f1_iid_bootstrap_95_percent_interval",
                    "threshold_delta_holdout_f1",
                ]
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="config/selected_model.json")
    parser.add_argument("--figure-dir", default="report/figures")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output-dir", help="Fresh directory required for --recompute")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.recompute:
        if not args.output_dir:
            parser.error("--recompute requires a fresh --output-dir")
        summary, curve = recompute(
            args.results_dir, args.data_dir, args.artifact_dir, args.output_dir, config
        )
    else:
        summary = json.loads(
            (Path(args.results_dir) / "validation_summary.json").read_text()
        )
        curve = json.loads(
            (Path(args.results_dir) / "selected_threshold_curve.json").read_text()
        )
    plot_figures(summary, curve, config, args.results_dir, args.figure_dir)


if __name__ == "__main__":
    main()

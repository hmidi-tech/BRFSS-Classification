"""Derive ablations, uncertainty diagnostics, and report figures from saved runs."""

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from evaluation import metrics
from experiments import write_json


def main():
    rows = json.loads(Path("results/leaderboard.json").read_text())
    by_name = {r["config"]["name"]: r for r in rows}
    split = np.load("artifacts/split_indices.npz")
    y = np.loadtxt(
        "dataset/y_train.csv", delimiter=",", skiprows=1, usecols=1, dtype=int
    )
    tuning_y = y[split["tuning"]]
    fixed = {}
    for name, row in by_name.items():
        scores = np.load(row["artifact_stem"] + ".npz")["tuning_scores"]
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
    with Path("results/ablation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablations[0]))
        writer.writeheader()
        writer.writerows(ablations)
    held = json.loads(Path("results/holdout.json").read_text())
    selected = held["selected"]
    counts = np.array([selected[k] for k in ["tp", "fp", "fn", "tn"]])
    rng = np.random.default_rng(433)
    draws = rng.multinomial(counts.sum(), counts / counts.sum(), size=2000)
    bootstrap_f1 = 2 * draws[:, 0] / (2 * draws[:, 0] + draws[:, 1] + draws[:, 2])
    cv = json.loads(Path("results/cross_validation.json").read_text())
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
    write_json("results/validation_summary.json", summary)
    # Matplotlib is used only to produce figures, never for model fitting.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cs433-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    selected_config = json.loads(Path("config/selected_model.json").read_text())
    candidate = json.loads(
        Path(selected_config["candidate_artifact"] + ".json").read_text()
    )
    curve = candidate["threshold_curve"]
    fig, ax = plt.subplots(figsize=(4.6, 2.25), layout="constrained")
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
    ax.axvline(selected_config["threshold"], color="0.4", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Probability threshold", ylabel="Score", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    Path("report/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("report/figures/threshold_tradeoff.pdf")
    fig.savefig("report/figures/threshold_tradeoff.png", dpi=180)
    plt.close(fig)
    missing = list(csv.DictReader(Path("results/feature_audit.csv").open()))
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
    fig.savefig("report/figures/missingness.png", dpi=160)
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


if __name__ == "__main__":
    main()

"""Cross-check saved measurements and bind the report's numeric claims to them."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def _read(path):
    return json.loads(Path(path).read_text())


def _same(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"Inconsistent measured value: {label}")


def validate_measurements(
    results_dir="results", config_path="config/selected_model.json"
):
    """Reject disagreement between independent saved result views."""
    root = Path(results_dir)
    summary = _read(root / "validation_summary.json")
    held = _read(root / "holdout.json")
    cv = _read(root / "cross_validation.json")
    selection = _read(config_path)
    rows = {r["config"]["name"]: r for r in _read(root / "leaderboard.json")}
    chosen = next(
        r for r in rows.values() if r["experiment_id"] == selection["experiment_id"]
    )
    if (
        held != summary["holdout"]
        or held["experiment_id"] != selection["experiment_id"]
        or chosen["config"] != selection["model"]
    ):
        raise ValueError("Holdout, summary and selected configuration disagree")
    _same(selection["threshold"], chosen["tuning"]["threshold"], "selected threshold")
    _same(selection["feature_count"], chosen["feature_count"], "selected dimension")
    for field in ["selected", "selected_at_0_5", "majority_baseline"]:
        m = held[field]
        tp, fp, fn, tn = (m[k] for k in ["tp", "fp", "fn", "tn"])
        total = tp + fp + fn + tn
        for key, expected in {
            "accuracy": (tp + tn) / total,
            "f1": 2 * tp / (2 * tp + fp + fn),
            "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp),
        }.items():
            _same(m[key], expected, f"{field}.{key}")
    f1 = [f["metrics"]["f1"] for f in cv["folds"]]
    if len(f1) != 3:
        raise ValueError("Expected the measured three-fold diagnostic")
    _same(summary["cv_f1_mean"], statistics.mean(f1), "CV mean")
    _same(summary["cv_f1_sample_std"], statistics.stdev(f1), "CV sample SD")
    with (root / "experiment_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in [
                "f1",
                "accuracy",
                "threshold",
                "recall",
                "precision",
                "balanced_accuracy",
            ]:
                _same(
                    row[key], rows[row["name"]]["tuning"][key], f"{row['name']}.{key}"
                )
    with (root / "ablation_results.csv").open(newline="") as handle:
        ablations = list(csv.DictReader(handle))
    if len(ablations) != len(summary["ablations"]):
        raise ValueError("Ablation row counts disagree")
    for row, reference in zip(ablations, summary["ablations"]):
        for key in [
            "f1_before",
            "f1_after",
            "f1_delta",
            "accuracy_delta",
            "fixed_threshold",
        ]:
            _same(row[key], reference[key], f"ablation.{key}")
        _same(
            reference["f1_delta"],
            reference["f1_after"] - reference["f1_before"],
            "ablation delta",
        )
    return summary, rows, chosen, selection


def generate(
    results_dir="results",
    config_path="config/selected_model.json",
    model_metadata="artifacts/final_model.json",
):
    summary, rows, chosen, selection = validate_measurements(results_dir, config_path)
    root = Path(results_dir)
    data, protocol, final = (
        _read(root / "data_audit.json"),
        _read(root / "protocol.json"),
        _read(model_metadata),
    )
    if final["selection"] != selection:
        raise ValueError("Saved final model does not match the selected configuration")
    h, half = summary["holdout"]["selected"], summary["holdout"]["selected_at_0_5"]
    values = {}

    def number(name, value, digits=4):
        values[name] = f"{value:.{digits}f}"

    def integer(name, value):
        values[name] = f"{int(value):,}"

    for name, value in [
        ("NTrain", data["shape_train"][0]),
        ("NTest", data["shape_test"][0]),
        ("NFeatures", data["shape_train"][1]),
        ("NPositive", data["positive"]),
        ("SelectedDimension", selection["feature_count"]),
        ("FinalDimension", len(final["preprocessor"]["feature_names"])),
        ("ProxyObserved", data["HAREHAB1_available"]),
        ("Seed", selection["seed"]),
        ("MaxIterations", selection["model"]["max_iters"]),
    ]:
        integer(name, value)
    for name, key in [
        ("NDevelopment", "development"),
        ("NTuning", "tuning"),
        ("NHoldout", "holdout"),
    ]:
        integer(name, protocol["sizes"][key])
    for name, key in [
        ("HoldFone", "f1"),
        ("HoldAccuracy", "accuracy"),
        ("HoldRecall", "recall"),
        ("HoldBalanced", "balanced_accuracy"),
        ("HoldPrecision", "precision"),
        ("HoldAUC", "roc_auc"),
        ("HoldAP", "average_precision"),
    ]:
        number(name, h[key])
    for key in ["tp", "fp", "fn", "tn"]:
        integer("Hold" + key.upper(), h[key])
    for name, value in [
        ("Prevalence", 100 * data["positive_fraction"]),
        ("TrainMissing", 100 * data["train_missing_fraction"]),
        ("TestMissing", 100 * data["test_missing_fraction"]),
        ("CVMean", summary["cv_f1_mean"]),
        ("CVStd", summary["cv_f1_sample_std"]),
        ("TuningFone", chosen["tuning"]["f1"]),
        ("TrainingFone", chosen["training_at_selected_threshold"]["f1"]),
        ("UnregularizedFone", rows["A3_l2_0"]["tuning"]["f1"]),
        ("StrongLtwoFone", rows["A3_l2_0.01"]["tuning"]["f1"]),
        ("HalfFone", half["f1"]),
        ("HalfAccuracy", half["accuracy"]),
        ("HalfRecall", half["recall"]),
        ("ThresholdFoneGain", h["f1"] - half["f1"]),
        ("ThresholdAccuracyLoss", half["accuracy"] - h["accuracy"]),
        ("CategoryGain", summary["ablations"][2]["f1_delta"]),
        ("AccuracyTunedScore", chosen["tuning_accuracy_optimal"]["accuracy"]),
        ("AccuracyTunedFone", chosen["tuning_accuracy_optimal"]["f1"]),
    ]:
        number(name, value)
    for name, value in [
        ("SelectedThreshold", selection["threshold"]),
        ("AccuracyTunedThreshold", chosen["tuning_accuracy_optimal"]["threshold"]),
        ("HoldFNRPercent", 100 * h["false_negative_rate"]),
    ]:
        number(name, value, 2)
    for name, value in zip(
        ["HoldCILower", "HoldCIUpper"],
        summary["holdout_f1_iid_bootstrap_95_percent_interval"],
    ):
        number(name, value)
    number("FinalGradient", final["fit"]["gradient_inf_norm"], 7)
    number("MajorityAccuracy", summary["holdout"]["majority_baseline"]["accuracy"])
    number("SelectedLambda", selection["model"]["lambda"], 4)
    diagnostics = _read(root / "review_20261029/diagnostics.json")
    if (
        diagnostics.get("status") != "completed"
        or len(diagnostics["categorical_folds"]) != 3
    ):
        raise ValueError(
            "Report robustness claims require three completed diagnostic folds"
        )
    deltas = [fold["f1_delta"] for fold in diagnostics["categorical_folds"]]
    number("CategoryRobustMin", min(deltas))
    number("CategoryRobustMax", max(deltas))
    lines = [
        "% Generated and cross-checked by build_report.py; do not edit numbers manually."
    ]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    baseline = summary["holdout"]["majority_baseline"]
    table = [
        f"Majority & --- & {baseline['accuracy']:.4f} & {baseline['f1']:.4f} & {baseline['recall']:.4f}\\\\"
    ]
    for name, label in [
        ("linear_0", "Linear regression"),
        ("linear_0.001", "Ridge ($10^{-3}$)"),
        ("A0_mean", "Logistic, mean fill"),
        ("A3_l2_0", "Categorical, no L2"),
        ("A3_l2_0.0001", "Selected L2 logistic"),
    ]:
        m = rows[name]["tuning"]
        table.append(
            f"{label} & {m['threshold']:.2f} & {m['accuracy']:.4f} & {m['f1']:.4f} & {m['recall']:.4f}\\\\"
        )
    lines.append("\\newcommand{\\ModelRows}{\n" + "\n".join(table) + "\n}")
    ablation = [
        f"Mean fill + scaling & {summary['fixed_threshold_metrics']['A0_mean']['f1']:.4f} & ---\\\\"
    ]
    for label, entry in zip(
        [
            "Median fill",
            "+ Missing indicators",
            "+ Categorical encoding",
            "L2 $10^{-3}\\to10^{-4}$",
        ],
        summary["ablations"][:4],
    ):
        ablation.append(
            f"{label} & {entry['f1_after']:.4f} & {entry['f1_delta']:+.4f}\\\\"
        )
    lines.append("\\newcommand{\\AblationRows}{\n" + "\n".join(ablation) + "\n}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="config/selected_model.json")
    parser.add_argument("--model-metadata", default="artifacts/final_model.json")
    parser.add_argument("--output", default="report/results.tex")
    args = parser.parse_args()
    Path(args.output).write_text(
        generate(args.results_dir, args.config, args.model_metadata)
    )
    print(
        f"Validated saved measurements and generated {args.output}; local-only candidates and course examples are not required."
    )


if __name__ == "__main__":
    main()

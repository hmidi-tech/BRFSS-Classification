"""Bind report tables and macros to saved measurements; preserve author metadata."""

import json
import shutil
from pathlib import Path


def main():
    summary = json.loads(Path("results/validation_summary.json").read_text())
    rows = {
        r["config"]["name"]: r
        for r in json.loads(Path("results/leaderboard.json").read_text())
    }
    held = summary["holdout"]["selected"]
    macros = {
        "HoldFone": held["f1"],
        "HoldAccuracy": held["accuracy"],
        "HoldRecall": held["recall"],
        "HoldBalanced": held["balanced_accuracy"],
        "HoldPrecision": held["precision"],
        "CVMean": summary["cv_f1_mean"],
        "CVStd": summary["cv_f1_sample_std"],
    }
    lines = [
        f"\\newcommand{{\\{name}}}{{{value:.4f}}}" for name, value in macros.items()
    ]
    table = ["Majority & --- & 0.9117 & 0.0000 & 0.0000\\\\"]
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
    fixed = summary["fixed_threshold_metrics"]
    ablation = [f"Mean fill + scaling & {fixed['A0_mean']['f1']:.4f} & ---\\\\"]
    labels = [
        "Median fill",
        "+ Missing indicators",
        "+ Categorical encoding",
        "L2 $10^{-3}\\to10^{-4}$",
    ]
    for label, entry in zip(labels, summary["ablations"][:4]):
        ablation.append(
            f"{label} & {entry['f1_after']:.4f} & {entry['f1_delta']:+.4f}\\\\"
        )
    lines.append("\\newcommand{\\AblationRows}{\n" + "\n".join(ablation) + "\n}")
    Path("report/results.tex").write_text(
        "% Generated from saved measurements by build_report.py\n"
        + "\n".join(lines)
        + "\n"
    )
    shutil.copy2("latex-example-paper/IEEEtran.cls", "report/IEEEtran.cls")
    print(
        "Generated report/results.tex from saved measurements; copied unmodified official class."
    )


if __name__ == "__main__":
    main()

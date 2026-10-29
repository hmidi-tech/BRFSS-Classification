"""Bounded development-only optimizer and categorical-encoding diagnostics.

These runs do not replace the frozen configuration or evaluate new holdout
models. Models and row indices stay in an ignored local output directory.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data_io import current_commit, load_data, sha256, write_json
from evaluation import metrics, stratified_folds, stratified_split
from models import fit_logistic, logistic_value_gradient, predict_scores
from preprocessing import Preprocessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError("Choose a new diagnostic output directory")
    out.mkdir(parents=True)
    data = load_data(args.data_dir, include_test=False)
    x, y, names = data["x_train"], data["y"], data["names"]
    development, tuning, _ = stratified_split(y, 433)
    result = {
        "git_commit": current_commit(),
        "seed": 433,
        "data_fingerprints": data["fingerprints"],
        "sources": {
            p: sha256(p)
            for p in [
                "models.py",
                "preprocessing.py",
                "evaluation.py",
                "data_io.py",
                "review_experiments.py",
            ]
        },
        "scope": "Development/tuning diagnostics after the historical holdout was opened; no new independent final estimate and no replacement selection.",
        "plan": "Compare legacy and float64 objective/gradient arithmetic at 120 updates. Only if float64 remains above tolerance and decreases objective by >1e-7 over its last 50 updates, run one 240-update diagnostic. Check categorical encoding in three shared development-only folds, lambda=0.001, fixed threshold 0.20, 120 legacy updates. No holdout labels used.",
        "optimizer": [],
        "categorical_folds": [],
    }
    write_json(out / "diagnostics.json", result)
    pre = Preprocessor(imputation="median", indicators=True, categorical=True).fit(
        x, names, development
    )
    train_x, tune_x = pre.transform(x, development), pre.transform(x, tuning)
    train_y = (y[development] == 1).astype(float)
    for precision in ["legacy_float32", "float64"]:
        w, info = fit_logistic(train_x, train_y, 1e-4, 120, 1e-5, precision=precision)
        exact_value, exact_grad = logistic_value_gradient(train_x, train_y, w, 1e-4)
        row = {
            "precision": precision,
            "budget": 120,
            "features": train_x.shape[1],
            "fit": info,
            "float64_objective": exact_value,
            "float64_gradient_inf_norm": float(np.max(np.abs(exact_grad))),
            "tuning_fixed_threshold": metrics(
                y[tuning], predict_scores(tune_x, w, "logistic"), 0.20
            ),
        }
        result["optimizer"].append(row)
        np.savez(out / f"optimizer_{precision}_120.npz", weights=w)
        write_json(out / "diagnostics.json", result)
        print(
            "OPTIMIZER",
            precision,
            info["status"],
            info["gradient_inf_norm"],
            row["tuning_fixed_threshold"]["f1"],
            flush=True,
        )
    trace = info["trace"]
    if (
        not info["converged"]
        and len(trace) > 50
        and trace[-51]["objective"] - trace[-1]["objective"] > 1e-7
    ):
        w, info = fit_logistic(train_x, train_y, 1e-4, 240, 1e-5, precision="float64")
        row = {
            "precision": "float64",
            "budget": 240,
            "features": train_x.shape[1],
            "fit": info,
            "float64_objective": info["objective"],
            "float64_gradient_inf_norm": info["gradient_inf_norm"],
            "tuning_fixed_threshold": metrics(
                y[tuning], predict_scores(tune_x, w, "logistic"), 0.20
            ),
        }
        result["optimizer"].append(row)
        np.savez(out / "optimizer_float64_240.npz", weights=w)
        write_json(out / "diagnostics.json", result)
        print(
            "OPTIMIZER float64 240",
            info["status"],
            info["gradient_inf_norm"],
            row["tuning_fixed_threshold"]["f1"],
            flush=True,
        )
    del train_x, tune_x
    folds = stratified_folds(y[development], 3, 433)
    for i in range(3):
        train_ids = development[np.concatenate([folds[j] for j in range(3) if j != i])]
        valid_ids = development[folds[i]]
        fold = {
            "fold": i,
            "training_rows": len(train_ids),
            "validation_rows": len(valid_ids),
            "models": {},
        }
        np.savez(out / f"fold_{i}_indices.npz", train=train_ids, valid=valid_ids)
        for categorical in [False, True]:
            start = time.perf_counter()
            pre = Preprocessor(
                imputation="median", indicators=True, categorical=categorical
            ).fit(x, names, train_ids)
            train_x, valid_x = pre.transform(x, train_ids), pre.transform(x, valid_ids)
            w, fit = fit_logistic(
                train_x,
                (y[train_ids] == 1).astype(float),
                0.001,
                120,
                1e-5,
                precision="legacy_float32",
            )
            key = "categorical" if categorical else "numeric"
            fold["models"][key] = {
                "features": train_x.shape[1],
                "fit": fit,
                "metrics": metrics(
                    y[valid_ids], predict_scores(valid_x, w, "logistic"), 0.20
                ),
                "total_seconds": time.perf_counter() - start,
            }
            np.savez(out / f"fold_{i}_{key}.npz", weights=w)
            del train_x, valid_x
        fold["f1_delta"] = (
            fold["models"]["categorical"]["metrics"]["f1"]
            - fold["models"]["numeric"]["metrics"]["f1"]
        )
        result["categorical_folds"].append(fold)
        write_json(out / "diagnostics.json", result)
        print("FOLD", i, "categorical F1 delta", fold["f1_delta"], flush=True)
    result["status"] = "completed"
    result["categorical_f1_delta_mean"] = float(
        np.mean([f["f1_delta"] for f in result["categorical_folds"]])
    )
    write_json(out / "diagnostics.json", result)


if __name__ == "__main__":
    main()

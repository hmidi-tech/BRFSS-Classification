"""Bounded staged comparison, tuning-only selection, and held-out evaluation."""

import argparse
import csv
import json
import os
import resource
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from data_io import load_data, sha256
from evaluation import choose_threshold, metrics, stratified_folds, stratified_split
from models import fit_logistic, fit_ridge, predict_scores
from preprocessing import Preprocessor


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def append_event(path, record):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


def source_fingerprint():
    paths = [
        "implementations.py",
        "data_io.py",
        "preprocessing.py",
        "models.py",
        "evaluation.py",
        "experiments.py",
    ]
    return {p: sha256(p) for p in paths}


def train(x, y, config):
    if config["family"] == "logistic":
        return fit_logistic(
            x, y, config["lambda"], config["max_iters"], config["tolerance"]
        )
    return fit_ridge(x, y, config["lambda"])


def run(
    data_dir,
    seed,
    cross_validate,
    output_dir="results",
    artifact_dir="artifacts",
    config_dir="config",
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = Path(artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    selected_path = Path(config_dir) / "selected_model.json"
    run_id = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%dT%H%M%S%f")
    log = out / "experiment_results.jsonl"
    if selected_path.exists():
        raise FileExistsError(
            "Preserve the selected model; choose new output, artifact and config directories for another search."
        )
    d = load_data(data_dir, include_test=False)
    x, y, names = d["x_train"], d["y"], d["names"]
    development, tuning, holdout = stratified_split(y, seed)
    split_path = artifacts / "split_indices.npz"
    np.savez(split_path, development=development, tuning=tuning, holdout=holdout)
    protocol = {
        "run_id": run_id,
        "seed": seed,
        "split": "60/20/20 stratified development/tuning/untouched holdout",
        "sizes": {
            "development": len(development),
            "tuning": len(tuning),
            "holdout": len(holdout),
        },
        "selection_objective": "f1",
        "metric_status": "provisional F1 from observed leaderboard; written rules specify accuracy; unresolved",
        "selection_rule": "Highest tuning F1 at its tuning-only fixed-grid threshold; within 0.002 F1 of maximum prefer fewer features, then stronger regularization, then higher tuning F1.",
        "threshold_grid": "0.05 to 0.65 inclusive, step 0.01; ties prefer higher threshold",
        "budget": "10 configurations, maximum 120 optimizer iterations each; optional 3-fold fixed-configuration diagnostic with internal threshold calibration",
        "holdout_policy": "Open only after selected configuration is written. Never change selection based on holdout.",
        "fingerprints": d["fingerprints"],
        "sources": source_fingerprint(),
        "git_commit": None,
        "python": sys.version,
        "numpy": np.__version__,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    write_json(out / "protocol.json", protocol)
    configs = []
    variants = [
        ("A0_mean", dict(imputation="mean", indicators=False, categorical=False)),
        ("A1_median", dict(imputation="median", indicators=False, categorical=False)),
        ("A2_missing", dict(imputation="median", indicators=True, categorical=False)),
        ("A3_categories", dict(imputation="median", indicators=True, categorical=True)),
    ]
    for name, pre in variants:
        configs.append(
            dict(
                name=name,
                preprocessing=pre,
                family="logistic",
                **{"lambda": 0.001},
                max_iters=120,
                tolerance=1e-5,
            )
        )
    for lam in [0.0, 0.0001, 0.01, 0.1]:
        configs.append(
            dict(
                name=f"A3_l2_{lam:g}",
                preprocessing=variants[-1][1],
                family="logistic",
                **{"lambda": lam},
                max_iters=120,
                tolerance=1e-5,
            )
        )
    for lam in [0.0, 0.001]:
        configs.append(
            dict(
                name=f"linear_{lam:g}",
                preprocessing=variants[2][1],
                family="ridge",
                **{"lambda": lam},
                max_iters=1,
                tolerance=0,
            )
        )
    write_json(out / "planned_configurations.json", configs)
    pre = Preprocessor(**variants[2][1]).fit(x, names, development[:4000])
    smoke_x = pre.transform(x, development[:4000])
    smoke_w, smoke_info = fit_logistic(
        smoke_x, (y[development[:4000]] == 1).astype(float), max_iters=20
    )
    write_json(
        out / "smoke.json",
        {
            "rows": 4000,
            "features": smoke_x.shape[1],
            "finite_weights": bool(np.isfinite(smoke_w).all()),
            "optimizer": smoke_info,
        },
    )
    print("smoke complete", flush=True)
    successful = []
    cached_pre_config = None
    for number, config in enumerate(configs):
        experiment_id = f"{run_id}_{number:02d}_{config['name']}"
        base = {
            "experiment_id": experiment_id,
            "timestamp_zurich": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
            "run_id": run_id,
            "git_commit": None,
            "dataset_fingerprints": d["fingerprints"],
            "source_fingerprints": protocol["sources"],
            "seed": seed,
            "split_file": str(split_path),
            "config": config,
        }
        append_event(log, {**base, "status": "started"})
        start = time.perf_counter()
        try:
            if cached_pre_config != config["preprocessing"]:
                if cached_pre_config is not None:
                    del train_x, tune_x
                pre = Preprocessor(**config["preprocessing"]).fit(x, names, development)
                train_x = pre.transform(x, development)
                tune_x = pre.transform(x, tuning)
                cached_pre_config = config["preprocessing"]
            print(experiment_id, "shape", train_x.shape, flush=True)
            w, info = train(train_x, (y[development] == 1).astype(float), config)
            scores = predict_scores(tune_x, w, config["family"])
            threshold, curve = choose_threshold(y[tuning], scores)
            accuracy_threshold, _ = choose_threshold(
                y[tuning], scores, objective="accuracy"
            )
            probabilities = config["family"] == "logistic"
            result = {
                **base,
                "status": "successful",
                "feature_count": train_x.shape[1],
                "fit": info,
                "elapsed_seconds": time.perf_counter() - start,
                "tuning_at_0_5": metrics(y[tuning], scores, 0.5, probabilities),
                "tuning": metrics(y[tuning], scores, threshold, probabilities),
                "tuning_accuracy_optimal": metrics(
                    y[tuning], scores, accuracy_threshold, probabilities
                ),
                "training_at_selected_threshold": metrics(
                    y[development],
                    predict_scores(train_x, w, config["family"]),
                    threshold,
                    probabilities,
                ),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024,
            }
            if not np.isfinite(w).all():
                raise FloatingPointError("Nonfinite model weights")
            candidate_dir = artifacts / "candidates"
            candidate_dir.mkdir(exist_ok=True)
            result["artifact_stem"] = str(candidate_dir / experiment_id)
            np.savez(result["artifact_stem"] + ".npz", weights=w, tuning_scores=scores)
            write_json(
                result["artifact_stem"] + ".json",
                {
                    "preprocessor": pre.state(),
                    "config": config,
                    "result": result,
                    "threshold_curve": curve,
                },
            )
            append_event(log, result)
            successful.append(result)
            print(
                "complete",
                config["name"],
                "F1",
                result["tuning"]["f1"],
                "accuracy",
                result["tuning"]["accuracy"],
                "threshold",
                threshold,
                "fit_s",
                info["fit_seconds"],
                info.get("status"),
                flush=True,
            )
        except (Exception, KeyboardInterrupt) as exc:
            append_event(
                log,
                {
                    **base,
                    "status": (
                        "interrupted"
                        if isinstance(exc, KeyboardInterrupt)
                        else "failed"
                    ),
                    "reason": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if isinstance(exc, KeyboardInterrupt):
                raise
            print("failed", experiment_id, repr(exc), flush=True)
    if not successful:
        raise RuntimeError("No successful experiment; no model selected")
    maximum = max(r["tuning"]["f1"] for r in successful)
    eligible = [r for r in successful if r["tuning"]["f1"] >= maximum - 0.002]
    selected = min(
        eligible,
        key=lambda r: (r["feature_count"], -r["config"]["lambda"], -r["tuning"]["f1"]),
    )
    selection = {
        "experiment_id": selected["experiment_id"],
        "seed": seed,
        "model": selected["config"],
        "threshold": selected["tuning"]["threshold"],
        "feature_count": selected["feature_count"],
        "selection_rule": protocol["selection_rule"],
        "metric_status": protocol["metric_status"],
        "dataset_fingerprints": d["fingerprints"],
        "candidate_artifact": selected["artifact_stem"],
        "frozen_before_holdout_at": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        "git_commit": None,
    }
    write_json(selected_path, selection)
    state = json.loads(Path(selected["artifact_stem"] + ".json").read_text())
    pre = Preprocessor.from_state(state["preprocessor"])
    w = np.load(selected["artifact_stem"] + ".npz")["weights"]
    hold_x = pre.transform(x, holdout)
    scores = predict_scores(hold_x, w, selected["config"]["family"])
    held = metrics(
        y[holdout],
        scores,
        selection["threshold"],
        selected["config"]["family"] == "logistic",
    )
    np.savez(
        artifacts / "selected_holdout.npz", indices=holdout, y=y[holdout], scores=scores
    )
    write_json(
        out / "holdout.json",
        {
            "experiment_id": selected["experiment_id"],
            "selected": held,
            "majority_baseline": metrics(y[holdout], np.zeros(holdout.size)),
            "selected_at_0_5": metrics(
                y[holdout], scores, 0.5, selected["config"]["family"] == "logistic"
            ),
            "evaluated_after_freeze": True,
        },
    )
    write_json(
        out / "leaderboard.json",
        sorted(successful, key=lambda r: r["tuning"]["f1"], reverse=True),
    )
    with (out / "experiment_results.csv").open("w", newline="") as handle:
        fields = [
            "name",
            "family",
            "lambda",
            "features",
            "threshold",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
            "fit_seconds",
            "optimizer_status",
            "experiment_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in sorted(successful, key=lambda r: r["tuning"]["f1"], reverse=True):
            writer.writerow(
                {
                    "name": r["config"]["name"],
                    "family": r["config"]["family"],
                    "lambda": r["config"]["lambda"],
                    "features": r["feature_count"],
                    **{
                        key: r["tuning"][key]
                        for key in [
                            "threshold",
                            "accuracy",
                            "f1",
                            "precision",
                            "recall",
                            "balanced_accuracy",
                        ]
                    },
                    "fit_seconds": r["fit"]["fit_seconds"],
                    "optimizer_status": r["fit"].get("status", "closed_form"),
                    "experiment_id": r["experiment_id"],
                }
            )
    print("SELECTED", json.dumps(selection), "HOLDOUT", json.dumps(held), flush=True)
    del train_x, tune_x, hold_x
    if cross_validate:
        cv_results = []
        outer_pool = np.concatenate([development, tuning])
        folds = stratified_folds(y[outer_pool], 3, seed)
        for i in range(3):
            outer_valid = outer_pool[folds[i]]
            outer_train = outer_pool[
                np.concatenate([folds[j] for j in range(3) if j != i])
            ]
            a, b, c = stratified_split(y[outer_train], seed + i + 1)
            inner_fit = outer_train[np.concatenate([a, b])]
            inner_cal = outer_train[c]
            pre_inner = Preprocessor(**selection["model"]["preprocessing"]).fit(
                x, names, inner_fit
            )
            inner_x = pre_inner.transform(x, inner_fit)
            wi, info_i = train(
                inner_x, (y[inner_fit] == 1).astype(float), selection["model"]
            )
            cal_x = pre_inner.transform(x, inner_cal)
            threshold, _ = choose_threshold(
                y[inner_cal], predict_scores(cal_x, wi, selection["model"]["family"])
            )
            del inner_x, cal_x
            pre_outer = Preprocessor(**selection["model"]["preprocessing"]).fit(
                x, names, outer_train
            )
            outer_x = pre_outer.transform(x, outer_train)
            wo, info_o = train(
                outer_x, (y[outer_train] == 1).astype(float), selection["model"]
            )
            valid_x = pre_outer.transform(x, outer_valid)
            cv_result = {
                "fold": i,
                "seed": seed,
                "train_rows": len(outer_train),
                "validation_rows": len(outer_valid),
                "features": outer_x.shape[1],
                "inner_fit": info_i,
                "outer_fit": info_o,
                "metrics": metrics(
                    y[outer_valid],
                    predict_scores(valid_x, wo, selection["model"]["family"]),
                    threshold,
                    selection["model"]["family"] == "logistic",
                ),
                "coefficient_norm": float(np.linalg.norm(wo)),
            }
            cv_results.append(cv_result)
            append_event(
                log,
                {
                    "experiment_id": f"{run_id}_cv_{i}",
                    "status": "successful",
                    "config": selection["model"],
                    "result": cv_result,
                    "dataset_fingerprints": d["fingerprints"],
                    "seed": seed,
                },
            )
            write_json(
                out / "cross_validation.json",
                {
                    "scope": "Fixed selected configuration, 3 folds on the 80% non-holdout pool. Threshold calibrated inside each outer training partition. Family/hyperparameter selection was not nested; consistency diagnostic, not an unbiased search estimate.",
                    "folds": cv_results,
                },
            )
            print("CV", i, cv_result["metrics"], flush=True)
            del outer_x, valid_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--seed", type=int, default=433)
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()
    run(
        args.data_dir,
        args.seed,
        args.cross_validate,
        args.output_dir,
        args.artifact_dir,
        args.config_dir,
    )

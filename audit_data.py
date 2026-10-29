"""Measured EDA and data-integrity audit; never fits prediction preprocessing."""

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from data_io import load_data, sha256
from evaluation import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    start = time.perf_counter()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh --output-dir; preserve historical audits")
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(args.data_dir)
    x, xt, y, names = d["x_train"], d["x_test"], d["y"], d["names"]
    records = []
    for j, name in enumerate(names):
        record = {"feature": name}
        for label, matrix in [("train", x), ("test", xt)]:
            col = np.array(matrix[:, j])
            valid = np.isfinite(col)
            values = col[valid]
            unique, counts = np.unique(values, return_counts=True)
            summary = {
                "missing": int(np.isnan(col).sum()),
                "infinite": int(np.isinf(col).sum()),
                "missing_fraction": float(np.isnan(col).mean()),
                "unique_observed": unique.size,
                "min": float(np.min(values)) if values.size else None,
                "max": float(np.max(values)) if values.size else None,
                "mean": float(np.mean(values)) if values.size else None,
                "std": float(np.std(values)) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "mode": float(unique[np.argmax(counts)]) if values.size else None,
                "mode_fraction_observed": (
                    float(counts.max() / values.size) if values.size else None
                ),
                "small_integer_set": bool(
                    unique.size <= 12 and np.equal(unique, np.floor(unique)).all()
                ),
            }
            record.update({f"{label}_{key}": value for key, value in summary.items()})
        record["missing_fraction_shift"] = (
            record["test_missing_fraction"] - record["train_missing_fraction"]
        )
        records.append(record)
    with (out / "feature_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("feature statistics complete", flush=True)
    # SHA-256 candidates are verified against the actual rows with equal_nan.
    seen = {}
    duplicates = {
        "train": 0,
        "test": 0,
        "cross_split": 0,
        "conflicting_train_labels": 0,
    }
    for label, matrix in [("train", x), ("test", xt)]:
        for i, row in enumerate(matrix):
            digest = hashlib.sha256(row.tobytes()).digest()
            if digest in seen:
                old_label, old_i = seen[digest]
                old = x[old_i] if old_label == "train" else xt[old_i]
                if not np.array_equal(row, old, equal_nan=True):
                    raise AssertionError("Unexpected row-hash collision")
                if old_label != label:
                    duplicates["cross_split"] += 1
                else:
                    duplicates[label] += 1
                    if label == "train" and y[i] != y[old_i]:
                        duplicates["conflicting_train_labels"] += 1
            else:
                seen[digest] = (label, i)
    print("duplicate audit complete", flush=True)
    # Compact correlation screening uses only a deterministic training sample.
    rng = np.random.default_rng(433)
    sample = np.array(
        x[rng.choice(x.shape[0], size=min(20000, x.shape[0]), replace=False)]
    )
    finite = np.isfinite(sample)
    means = np.divide(
        np.where(finite, sample, 0).sum(axis=0),
        finite.sum(axis=0),
        out=np.zeros(sample.shape[1]),
        where=finite.sum(axis=0) > 0,
    )
    sample = np.where(finite, sample, means)
    sample -= sample.mean(axis=0)
    scales = np.sqrt((sample * sample).sum(axis=0))
    sample /= np.where(scales > 0, scales, 1)
    correlation = sample.T @ sample
    correlated = []
    for j in range(len(names)):
        for k in range(j + 1, len(names)):
            if abs(correlation[j, k]) >= 0.995:
                correlated.append(
                    {
                        "feature_a": names[j],
                        "feature_b": names[k],
                        "correlation": float(correlation[j, k]),
                    }
                )
    with (out / "correlated_features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["feature_a", "feature_b", "correlation"]
        )
        writer.writeheader()
        writer.writerows(correlated)
    proxy = x[:, names.index("HAREHAB1")]
    available = np.isfinite(proxy)
    with (Path(args.data_dir) / "sample_submission.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    sample_array = np.asarray(rows[1:], dtype=np.int64)
    summary = {
        "timestamp_zurich": datetime.now(ZoneInfo("Europe/Zurich")).isoformat(),
        "shape_train": list(x.shape),
        "shape_test": list(xt.shape),
        "positive": int((y == 1).sum()),
        "negative": int((y == -1).sum()),
        "positive_fraction": float((y == 1).mean()),
        "train_missing_cells": sum(r["train_missing"] for r in records),
        "test_missing_cells": sum(r["test_missing"] for r in records),
        "train_missing_fraction": sum(r["train_missing"] for r in records) / x.size,
        "test_missing_fraction": sum(r["test_missing"] for r in records) / xt.size,
        "all_missing_train": [
            r["feature"] for r in records if not r["train_unique_observed"]
        ],
        "single_observed_value_train": [
            r["feature"] for r in records if r["train_unique_observed"] == 1
        ],
        "near_constant_observed_train": [
            r["feature"]
            for r in records
            if r["train_mode_fraction_observed"] is not None
            and r["train_mode_fraction_observed"] >= 0.999
        ],
        "small_integer_set_features": sum(
            r["train_small_integer_set"] for r in records
        ),
        "infinite_cells": sum(
            r["train_infinite"] + r["test_infinite"] for r in records
        ),
        "row_duplicates_with_all_321_features": duplicates,
        "train_test_column_alignment": True,
        "train_label_id_alignment": True,
        "unique_ids_each_split": True,
        "train_test_id_overlap": 0,
        "row_lengths_valid": True,
        "HAREHAB1_available": int(available.sum()),
        "HAREHAB1_available_positive": int(((y == 1) & available).sum()),
        "sample_submission": {
            "header": rows[0],
            "rows": len(rows) - 1,
            "ids_align": bool(np.array_equal(sample_array[:, 0], d["test_ids"])),
            "labels": np.unique(sample_array[:, 1]).tolist(),
            "sha256": sha256(Path(args.data_dir) / "sample_submission.csv"),
        },
        "majority_baseline_all_labels_descriptive": metrics(y, np.zeros(y.size)),
        "fingerprints": d["fingerprints"],
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }
    (out / "data_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

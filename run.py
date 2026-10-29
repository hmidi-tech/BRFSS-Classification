"""Reproduce the selected model's submission from stored NumPy weights."""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from data_io import current_commit, load_features, sha256
from helpers import create_csv_submission
from models import predict_scores
from preprocessing import Preprocessor


def verify_submission(path, expected_ids):
    """Require exact row count, order, integral IDs and -1/+1 predictions."""
    expected_ids = np.asarray(expected_ids)
    if (
        expected_ids.ndim != 1
        or expected_ids.dtype.kind not in "iu"
        or np.unique(expected_ids).size != expected_ids.size
    ):
        raise ValueError("Expected IDs must be distinct integers in a vector")
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["Id", "Prediction"]:
            raise ValueError("Invalid submission header")
        count = 0
        positives = 0
        for count, row in enumerate(reader, start=1):
            if len(row) != 2 or count > len(expected_ids):
                raise ValueError("Unexpected submission column or row count")
            if int(row[0]) != expected_ids[count - 1] or row[1] not in {"-1", "1"}:
                raise ValueError("Invalid ID ordering or prediction encoding")
            positives += row[1] == "1"
        if count != len(expected_ids):
            raise ValueError("Missing predictions")
    return {
        "rows": count,
        "positive_predictions": int(positives),
        "negative_predictions": int(count - positives),
        "labels": [-1, 1] if 0 < positives < count else ([1] if positives else [-1]),
        "sha256": sha256(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--model", default="artifacts/final_model.npz")
    parser.add_argument("--seed", type=int, default=433)
    parser.add_argument("--cache-dir", default="artifacts/cache")
    args = parser.parse_args()
    output = Path(args.output)
    for path in [output, Path(str(output) + ".json"), Path(str(output) + ".sha256")]:
        if path.exists():
            raise FileExistsError(
                "Preserve existing CSV and sidecars; choose a new --output path"
            )
    model_path = Path(args.model)
    metadata = json.loads(model_path.with_suffix(".json").read_text())
    if sha256(model_path) != metadata["weights_sha256"]:
        raise ValueError("Stored weights checksum mismatch")
    selected = metadata["selection"]
    if args.seed != selected["seed"]:
        raise ValueError(
            "Use the documented selection seed; inference itself is deterministic"
        )
    x, test_ids, names, fingerprint = load_features(
        Path(args.data_dir) / "x_test.csv", args.cache_dir
    )
    pre = Preprocessor.from_state(metadata["preprocessor"])
    if names != pre.input_names:
        raise ValueError("Test feature names or order differ from the trained model")
    # Training fingerprints and schema are saved for audit. Test fingerprint is
    # recorded here, allowing the same fitted model to run on an ordered input.
    w = np.load(model_path, allow_pickle=False)["weights"]
    if w.shape != (len(pre.feature_names),) or not np.isfinite(w).all():
        raise ValueError("Invalid stored weights")
    threshold = selected["threshold"]
    if np.ndim(threshold) != 0 or not np.isfinite(threshold):
        raise ValueError("Stored threshold must be a finite scalar")
    predictions = np.empty(len(test_ids), dtype=np.int8)
    # Float64 score evaluation matches the saved model; bounding rows prevents
    # an implicit float64 copy of the entire expanded float32 design matrix.
    for start in range(0, len(test_ids), 8192):
        stop = min(start + 8192, len(test_ids))
        design = pre.transform(x, np.arange(start, stop), names=names)
        scores = predict_scores(design, w, selected["model"]["family"])
        if not np.isfinite(scores).all():
            raise FloatingPointError("Invalid prediction scores")
        predictions[start:stop] = np.where(scores >= threshold, 1, -1)
    if len(test_ids) != len(predictions):
        raise ValueError("ID/prediction length mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".prediction-", dir=output.parent
    ) as directory:
        temporary = Path(directory) / "predictions.csv"
        create_csv_submission(test_ids, predictions, str(temporary))
        result = verify_submission(temporary, test_ids)
        # link publishes the verified file without ever replacing an existing
        # output, including one created concurrently after the preflight check.
        os.link(temporary, output)
    result.update(
        output=str(output),
        model=str(model_path),
        model_sha256=sha256(model_path),
        sources_sha256={
            name: sha256(Path(__file__).resolve().parent / name)
            for name in [
                "run.py",
                "data_io.py",
                "models.py",
                "preprocessing.py",
                "helpers.py",
            ]
        },
        input_test_sha256=fingerprint,
        seed=args.seed,
        threshold=selected["threshold"],
        python=sys.version,
        numpy=np.__version__,
        command=[sys.executable, *sys.argv],
        git_commit=current_commit(),
        aicrowd_status="not_uploaded",
    )
    with Path(str(output) + ".sha256").open("x") as handle:
        handle.write(result["sha256"] + "  " + output.name + "\n")
    with Path(str(output) + ".json").open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        raise SystemExit(f"Prediction failed: {exc}") from exc

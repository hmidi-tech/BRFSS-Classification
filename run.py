"""Reproduce the selected model's submission from stored NumPy weights."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from data_io import _read_features, sha256
from helpers import create_csv_submission
from models import predict_scores
from preprocessing import Preprocessor


def verify_submission(path, expected_ids):
    """Require exact row count, order, integral IDs and -1/+1 predictions."""
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["Id", "Prediction"]:
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
    args = parser.parse_args()
    model_path = Path(args.model)
    metadata = json.loads(model_path.with_suffix(".json").read_text())
    if sha256(model_path) != metadata["weights_sha256"]:
        raise ValueError("Stored weights checksum mismatch")
    selected = metadata["selection"]
    if args.seed != selected["seed"]:
        raise ValueError(
            "Use the documented selection seed; inference itself is deterministic"
        )
    x, test_ids, names, fingerprint = _read_features(
        Path(args.data_dir) / "x_test.csv", "artifacts/cache"
    )
    pre = Preprocessor.from_state(metadata["preprocessor"])
    if names != pre.input_names:
        raise ValueError("Test feature names or order differ from the trained model")
    # Training fingerprints and schema are saved for audit. Test fingerprint is
    # recorded here, allowing the same fitted model to run on an ordered input.
    design = pre.transform(x)
    w = np.load(model_path, allow_pickle=False)["weights"]
    if w.shape != (design.shape[1],) or not np.isfinite(w).all():
        raise ValueError("Invalid stored weights")
    scores = predict_scores(design, w, selected["model"]["family"])
    if not np.isfinite(scores).all():
        raise FloatingPointError("Invalid prediction scores")
    predictions = np.where(scores >= selected["threshold"], 1, -1)
    if len(test_ids) != len(predictions):
        raise ValueError("ID/prediction length mismatch")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve the existing CSV; choose a new --output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(test_ids, predictions, str(output))
    result = verify_submission(output, test_ids)
    result.update(
        output=str(output),
        model=str(model_path),
        model_sha256=sha256(model_path),
        input_test_sha256=fingerprint,
        seed=args.seed,
        threshold=selected["threshold"],
        python=sys.version,
        numpy=np.__version__,
        command=[sys.executable, *sys.argv],
        git_commit=None,
        aicrowd_status="not_uploaded",
    )
    Path(str(output) + ".sha256").write_text(
        result["sha256"] + "  " + output.name + "\n"
    )
    Path(str(output) + ".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        raise SystemExit(f"Prediction failed: {exc}") from exc

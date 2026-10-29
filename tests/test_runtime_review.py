"""Exercise CLI inference with test data and a complete saved model only."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from data_io import sha256
from preprocessing import Preprocessor
from run import verify_submission


def _saved_example(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "x_test.csv").write_text("Id,a\n9,-1\n4,1\n7,\n")
    x = np.array([[-1.0], [1.0], [np.nan]])
    pre = Preprocessor(indicators=True).fit(x, ["a"], np.arange(3))
    model = tmp_path / "model.npz"
    np.savez(model, weights=np.array([0.0, 1.0, 0.0]))
    metadata = {
        "weights_sha256": sha256(model),
        "preprocessor": pre.state(),
        "selection": {"seed": 433, "threshold": 0.5, "model": {"family": "logistic"}},
    }
    model.with_suffix(".json").write_text(json.dumps(metadata))
    return data, model


def test_cli_with_only_test_csv_and_saved_state(tmp_path):
    data, model = _saved_example(tmp_path)
    code = Path(__file__).resolve().parents[1]
    output = tmp_path / "fresh.csv"
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "4"
    command = [
        sys.executable,
        str(code / "run.py"),
        "--data-dir",
        str(data),
        "--model",
        str(model),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert verify_submission(output, np.array([9, 4, 7]))["positive_predictions"] == 2
    assert not (data / "y_train.csv").exists()
    # Reusing either the CSV or a sidecar must preserve it and fail clearly.
    previous = output.read_bytes()
    result = subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True
    )
    assert result.returncode != 0 and "Preserve existing" in result.stderr
    assert output.read_bytes() == previous
    output.unlink()
    result = subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True
    )
    assert result.returncode != 0 and "Preserve existing" in result.stderr
    assert not output.exists()


def test_shape_correct_but_tampered_feature_names_are_rejected():
    x = np.array([[1.0, 2.0], [2.0, 3.0]])
    pre = Preprocessor().fit(x, ["a", "b"], np.arange(2))
    import pytest

    with pytest.raises(ValueError):
        pre.transform(x, names=["b", "a"])


def test_failed_parse_publishes_no_cache(tmp_path):
    from data_io import load_features
    import pytest

    source = tmp_path / "bad.csv"
    source.write_text("Id,a\n3,1\n4,2,3\n")
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="row length"):
        load_features(source, cache)
    assert list(cache.iterdir()) == []

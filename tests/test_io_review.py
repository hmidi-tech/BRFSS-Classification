"""Regression checks for cache integrity, schema boundaries and saved inference."""

import json

import numpy as np
import pytest

from data_io import _read_features
from preprocessing import Preprocessor


def test_corrupt_same_shape_cache_is_rejected(tmp_path):
    source = tmp_path / "x_test.csv"
    source.write_text("Id,a,b\n4,1,\n5,2,3\n")
    cache = tmp_path / "cache"
    _read_features(source, cache)
    path = next(cache.glob("*.npy"))
    raw = np.load(path, mmap_mode="r+")
    raw[0, 1] = 900
    raw.flush()
    del raw
    with pytest.raises(ValueError, match="checksum"):
        _read_features(source, cache)


def test_empty_csv_error_is_explicit(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="header"):
        _read_features(p, tmp_path / "cache")


def test_preprocessor_does_not_use_id_and_rejects_bad_fit_schema():
    x = np.array([[4.0, 2], [5, 3], [6, 4]])
    pre = Preprocessor().fit(x, ["Id", "feature"], np.arange(3))
    assert "Id" not in pre.feature_names
    with pytest.raises(ValueError):
        Preprocessor().fit(x, ["wrong_count"], np.arange(3))
    with pytest.raises(ValueError):
        Preprocessor().fit(x, ["a", "a"], np.arange(3))


def test_transform_requires_fitting_and_integer_row_indices():
    with pytest.raises(ValueError, match="fitted"):
        Preprocessor().transform(np.empty((2, 0)))
    x = np.array([[1.0], [2.0]])
    pre = Preprocessor().fit(x, ["a"], np.arange(2))
    with pytest.raises(ValueError, match="indices"):
        pre.transform(x, np.array([True, False]))


def test_deserialized_schema_must_match_specifications():
    x = np.array([[1.0], [2.0]])
    pre = Preprocessor().fit(x, ["a"], np.arange(2))
    state = pre.state()
    state["specs"][0]["name"] = "another_feature"
    with pytest.raises(ValueError, match="schema"):
        Preprocessor.from_state(state)


def test_unsafe_experiment_directories_are_rejected_before_writing(tmp_path):
    from experiments import run

    results = tmp_path / "results"
    results.mkdir()
    sentinel = results / "protocol.json"
    sentinel.write_text('{"protected": true}')
    with pytest.raises(FileExistsError, match="fresh"):
        run(
            tmp_path / "absent_data",
            433,
            False,
            results,
            tmp_path / "artifacts",
            tmp_path / "config",
        )
    assert json.loads(sentinel.read_text()) == {"protected": True}

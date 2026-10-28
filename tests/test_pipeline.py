"""Checks for leakage boundaries, probability metrics, and custom optimization."""

import numpy as np
import pytest

from evaluation import metrics, stratified_folds, stratified_split
from models import fit_logistic, fit_ridge, predict_scores
from preprocessing import Preprocessor


def test_preprocessing_fit_boundary_and_round_trip():
    x = np.array([[1, 1, 1], [3, np.nan, 2], [10000, 99, 7]], dtype=float)
    pre = Preprocessor(indicators=True).fit(x, ["a", "b", "HAREHAB1"], np.array([0, 1]))
    a = pre.transform(x)
    b = Preprocessor.from_state(pre.state()).transform(x)
    np.testing.assert_array_equal(a, b)
    assert pre.specs[0]["mean"] == 2
    assert all(s["name"] != "HAREHAB1" for s in pre.specs)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a[:2, 1], [-1, 1])


def test_categorical_unseen_and_all_missing():
    x = np.array([[1, np.nan], [2, np.nan], [3, np.nan]])
    pre = Preprocessor(categorical=True, indicators=True).fit(
        x, ["category", "empty"], np.array([0, 1])
    )
    assert len(pre.specs) == 1
    assert pre.specs[0]["levels"] == [2.0]
    assert pre.transform(x)[2, 1] == 0


def test_metrics_ties_and_confusion():
    y = np.array([1, -1, 1, -1])
    m = metrics(y, np.full(4, 0.5))
    assert m["roc_auc"] == 0.5 and m["average_precision"] == 0.5
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 0, 2, 0)
    m = metrics(y, np.array([0.9, 0.1, 0.8, 0.2]))
    assert m["accuracy"] == m["f1"] == m["roc_auc"] == m["average_precision"] == 1
    assert metrics(y, np.zeros(4))["precision"] is None
    assert metrics(np.ones(4), np.ones(4))["specificity"] is None
    with pytest.raises(ValueError):
        metrics(y, np.full(4, np.nan))


def test_splits_disjoint_and_deterministic():
    y = np.r_[np.ones(100), -np.ones(900)]
    parts = stratified_split(y)
    assert [len(p) for p in parts] == [600, 200, 200]
    assert np.unique(np.concatenate(parts)).size == len(y)
    for a, b in zip(parts, stratified_split(y)):
        np.testing.assert_array_equal(a, b)
        assert np.mean(y[a] == 1) == 0.1
    folds = stratified_folds(y)
    assert np.unique(np.concatenate(folds)).size == len(y)


def test_optional_optimizer_convergence_and_ridge_equivalence():
    rng = np.random.default_rng(8)
    x = np.c_[np.ones(500), rng.normal(size=(500, 3))].astype(np.float32)
    y = (rng.random(500) < 1 / (1 + np.exp(-x @ np.array([-1, 0.3, 1, -0.7])))).astype(
        float
    )
    w, info = fit_logistic(x, y, 0.01, max_iters=100, tolerance=1e-5)
    p = predict_scores(x, w, "logistic")
    grad = x.T.astype(float) @ (p - y) / len(y) + 0.02 * w * np.array([0, 1, 1, 1])
    assert np.max(np.abs(grad)) < 3e-5
    assert info["converged"]
    assert info["objective"] < np.log(2)
    from implementations import ridge_regression

    np.testing.assert_allclose(
        fit_ridge(x, y, 0.01)[0], ridge_regression(y, x, 0.01)[0], atol=1e-12
    )


def test_submission_validation_rejects_truncation_and_encoding(tmp_path):
    from run import verify_submission

    path = tmp_path / "predictions.csv"
    path.write_text("Id,Prediction\n8,-1\n9,1\n")
    assert verify_submission(path, np.array([8, 9]))["rows"] == 2
    with pytest.raises(ValueError, match="Missing predictions"):
        verify_submission(path, np.array([8, 9, 10]))
    path.write_text("Id,Prediction\n8,0\n9,1\n")
    with pytest.raises(ValueError, match="encoding"):
        verify_submission(path, np.array([8, 9]))

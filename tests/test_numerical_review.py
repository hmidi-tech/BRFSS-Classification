"""Regression tests for numerical contracts identified in the final review."""

import numpy as np
import pytest

import implementations as impl
from evaluation import choose_threshold, metrics
from models import fit_logistic, logistic_value_gradient, sigmoid


def test_ridge_extreme_finite_penalty_remains_finite():
    with np.errstate(over="raise", invalid="raise"):
        w, loss = impl.ridge_regression(np.array([0.0, 1.0]), np.eye(2), 1e308)
    assert np.isfinite(w).all() and np.isfinite(loss)
    np.testing.assert_allclose(w, [0.0, 2.5e-309], atol=0, rtol=1e-12)


def test_optional_sigmoid_integer_logits_preserve_probabilities():
    np.testing.assert_array_equal(sigmoid(np.array([-1000, 0, 1000])), [0.0, 0.5, 1.0])
    assert sigmoid(np.array([0.0], dtype=np.float32)).dtype == np.float32


@pytest.mark.parametrize(
    "y,scores",
    [
        (np.array([[1], [-1]]), np.array([[0.8], [0.2]])),
        (np.array([1, 2]), np.array([0.8, 0.2])),
        (np.array([1, -1]), np.array([np.nan, 0.2])),
        (np.array([1, -1]), np.array([[0.8], [0.2]])),
    ],
)
def test_evaluation_rejects_invalid_or_broadcast_inputs(y, scores):
    for fn in [metrics, choose_threshold]:
        with pytest.raises(ValueError):
            fn(y, scores)


def test_optional_zero_iterations_and_final_gradient_status():
    x = np.array([[1, -1], [1, 2], [1, 0]], dtype=np.float32)
    y = np.array([0.0, 1, 0])
    _, info = fit_logistic(x, y, max_iters=0)
    assert info["iterations"] == 0
    # An intentionally loose tolerance should also be checked after the final
    # allowed update, rather than waiting for an iteration that never runs.
    _, info = fit_logistic(x, y, max_iters=1, tolerance=0.4)
    assert info["gradient_inf_norm"] <= 0.4
    assert info["converged"] and info["status"] == "gradient_tolerance"


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_optional_optimizer_rejects_invalid_lambda(value):
    with pytest.raises(ValueError):
        fit_logistic(np.ones((2, 1)), np.array([0.0, 1.0]), lambda_=value)


@pytest.mark.parametrize("scale", [0.01, 1.0, 20.0])
@pytest.mark.parametrize("epsilon", [1e-4, 1e-5, 1e-6])
def test_optional_gradient_multiple_scales(scale, epsilon):
    rng = np.random.default_rng(98)
    x = np.c_[np.ones(41), scale * rng.normal(size=(41, 4))].astype(np.float32)
    y = (np.arange(41) % 3 == 0).astype(float)
    w = np.array([-0.5, 0.1, -0.2, 0.3, -0.1])
    value, gradient = logistic_value_gradient(x, y, w, 0.17)
    finite = []
    for e in np.eye(len(w)):
        plus = logistic_value_gradient(x, y, w + epsilon * e, 0.17)[0]
        minus = logistic_value_gradient(x, y, w - epsilon * e, 0.17)[0]
        finite.append((plus - minus) / (2 * epsilon))
    np.testing.assert_allclose(gradient, finite, atol=3e-7, rtol=3e-6)
    unregularized, grad0 = logistic_value_gradient(x, y, w, 0.0)
    assert gradient[0] == grad0[0]
    np.testing.assert_allclose(value - unregularized, 0.17 * (w[1:] @ w[1:]))


def test_optional_trace_final_state_and_descent():
    rng = np.random.default_rng(19)
    x = np.c_[np.ones(73), rng.normal(size=(73, 5))].astype(np.float32)
    y = (rng.random(73) < 0.3).astype(float)
    w, info = fit_logistic(x, y, 0.01, max_iters=50)
    objective, gradient = logistic_value_gradient(x, y, w, 0.01)
    np.testing.assert_allclose(info["objective"], objective, atol=1e-14)
    np.testing.assert_allclose(
        info["gradient_inf_norm"], np.max(np.abs(gradient)), atol=1e-14
    )
    trace = info["trace"]
    assert trace[-1]["iteration"] == info["iterations"]
    assert trace[-1]["objective"] == info["objective"]
    assert np.all(np.diff([r["objective"] for r in trace]) <= 1e-14)

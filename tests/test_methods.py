"""Independent analytical checks beyond the official signature tests."""

import numpy as np
import pytest

import implementations as impl


@pytest.fixture
def example():
    return (
        np.array([0.0, 1.0, 1.0]),
        np.array([[1.0, -2], [1, 0.5], [1, 2.0]]),
        np.array([0.2, -0.1]),
    )


@pytest.mark.parametrize("regularization", [0.0, 0.13])
def test_logistic_gradient_finite_difference(example, regularization):
    y, x, w = example

    def objective(v):
        z = x @ v
        return np.mean(np.logaddexp(0, z) - y * z) + regularization * (v @ v)

    eps = 1e-6
    finite = np.array(
        [
            (objective(w + eps * e) - objective(w - eps * e)) / (2 * eps)
            for e in np.eye(w.size)
        ]
    )
    result, loss = impl.reg_logistic_regression(y, x, regularization, w, 1, 0.07)
    np.testing.assert_allclose((w - result) / 0.07, finite, atol=1e-9)
    z = x @ result
    np.testing.assert_allclose(loss, np.mean(np.logaddexp(0, z) - y * z))


def test_mse_gradient_finite_difference(example):
    y, x, w = example
    eps = 1e-6

    def objective(v):
        return np.mean((x @ v - y) ** 2) / 2

    finite = np.array(
        [
            (objective(w + eps * e) - objective(w - eps * e)) / (2 * eps)
            for e in np.eye(w.size)
        ]
    )
    result, loss = impl.mean_squared_error_gd(y, x, w, 1, 0.05)
    np.testing.assert_allclose((w - result) / 0.05, finite, atol=1e-9)
    np.testing.assert_allclose(loss, objective(result))


@pytest.mark.parametrize(
    "method",
    [
        "mean_squared_error_gd",
        "mean_squared_error_sgd",
        "logistic_regression",
        "reg_logistic_regression",
    ],
)
def test_zero_iterations_and_no_mutation(example, method):
    y, x, w = example
    originals = [a.copy() for a in (y, x, w)]
    fn = getattr(impl, method)
    args = (y, x, 0.2, w, 0, 0.1) if method.startswith("reg_") else (y, x, w, 0, 0.1)
    result, loss = fn(*args)
    np.testing.assert_array_equal(result, w)
    assert not np.shares_memory(result, w)
    assert np.ndim(loss) == 0 and np.isfinite(loss)
    args = (y, x, 0.2, w, 2, 0.1) if method.startswith("reg_") else (y, x, w, 2, 0.1)
    fn(*args)
    for a, original in zip((y, x, w), originals):
        np.testing.assert_array_equal(a, original)


def test_ridge_factor_and_intercept(example):
    y, x, _ = example
    lam = 0.3
    w, loss = impl.ridge_regression(y, x, lam)
    np.testing.assert_allclose(x.T @ (x @ w - y) / y.size + 2 * lam * w, 0, atol=1e-14)
    np.testing.assert_allclose(loss, np.mean((x @ w - y) ** 2) / 2)
    np.testing.assert_allclose(
        impl.ridge_regression(y, x, 0)[0], impl.least_squares(y, x)[0]
    )


def test_sgd_seed_and_one_sample_updates(example):
    y, x, w = example
    np.random.seed(433)
    manual = w.copy()
    for _ in range(7):
        i = np.random.randint(y.size)
        manual -= 0.01 * x[i] * (x[i] @ manual - y[i])
    np.random.seed(433)
    result, loss = impl.mean_squared_error_sgd(y, x, w, 7, 0.01)
    np.testing.assert_array_equal(result, manual)
    np.testing.assert_allclose(loss, np.mean((x @ result - y) ** 2) / 2)


def test_extreme_logits_and_label_validation():
    x = np.array([[1.0], [-1.0]])
    for y in [np.array([1.0, 0]), np.array([0.0, 1])]:
        with np.errstate(over="raise", invalid="raise"):
            for fn in [impl.logistic_regression, impl.reg_logistic_regression]:
                args = (
                    (y, x, 0.0, np.array([1000.0]), 2, 0.1)
                    if fn == impl.reg_logistic_regression
                    else (y, x, np.array([1000.0]), 2, 0.1)
                )
                w, loss = fn(*args)
                assert np.isfinite(w).all() and np.isfinite(loss)
    with pytest.raises(ValueError, match="0/1"):
        impl.logistic_regression(np.array([-1, 1]), x, np.zeros(1), 1, 0.1)


def test_singular_and_nearly_singular():
    with pytest.raises(np.linalg.LinAlgError):
        impl.least_squares(np.array([0.0, 1]), np.ones((2, 2)))
    x = np.array([[1, 1], [1, 1 + 1e-6], [1, 1 - 1e-6]])
    w, loss = impl.ridge_regression(np.array([0, 1, 0]), x, 1e-3)
    assert np.isfinite(w).all() and np.isfinite(loss)


def test_invalid_inputs():
    for y, x in [
        (np.array([]), np.empty((0, 2))),
        (np.ones(2), np.ones((3, 2))),
        (np.ones(2), np.array([[1, np.nan], [1, 2]])),
    ]:
        with pytest.raises(ValueError):
            impl.least_squares(y, x)


def test_zero_regularization_logistic_equivalence(example):
    y, x, w = example
    ordinary = impl.logistic_regression(y, x, w, 7, 0.03)
    regularized = impl.reg_logistic_regression(y, x, 0, w, 7, 0.03)
    np.testing.assert_array_equal(ordinary[0], regularized[0])
    np.testing.assert_array_equal(ordinary[1], regularized[1])

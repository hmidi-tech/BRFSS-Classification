"""The six CS-433 Project 1 methods, using only NumPy.

All methods return the final weights and the unregularized data loss.
The supplied intercept, if any, is regularized like every other coefficient.
"""

import numpy as np


def _inputs(y, tx, initial_w=None, logistic=False):
    """Validate dimensions without changing caller-owned arrays."""
    x = np.asarray(tx, dtype=float)
    target = np.asarray(y, dtype=float)
    if x.ndim != 2 or target.ndim != 1 or x.shape[0] != target.size:
        raise ValueError("Expected tx with shape (N,D) and y with shape (N,).")
    if not target.size or not x.shape[1]:
        raise ValueError("The dataset must contain observations and features.")
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError("Inputs must be finite; preprocess missing values first.")
    if logistic and not np.isin(target, [0, 1]).all():
        raise ValueError("Logistic regression requires labels encoded as 0/1.")
    if initial_w is None:
        return target, x
    w = np.array(initial_w, dtype=float, copy=True)
    if w.shape != (x.shape[1],) or not np.isfinite(w).all():
        raise ValueError("initial_w must be a finite vector with shape (D,).")
    return target, x, w


def _iterations(max_iters, gamma):
    if (
        isinstance(max_iters, (bool, np.bool_))
        or not isinstance(max_iters, (int, np.integer))
        or max_iters < 0
    ):
        raise ValueError("max_iters must be a nonnegative integer.")
    if (
        np.ndim(gamma) != 0
        or not np.isreal(gamma)
        or not np.isfinite(gamma)
        or gamma < 0
    ):
        raise ValueError("gamma must be finite and nonnegative.")


def _regularization(lambda_):
    if (
        np.ndim(lambda_) != 0
        or not np.isreal(lambda_)
        or not np.isfinite(lambda_)
        or lambda_ < 0
    ):
        raise ValueError("lambda_ must be finite and nonnegative.")


def _mse(y, x, w):
    residual = x @ w - y
    return np.mean(residual * residual) / 2


def _sigmoid(z):
    """Stable sigmoid, avoiding exponentiation of a positive large logit."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1 + ez)
    return out


def _logistic_loss(y, x, w):
    z = x @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Run exactly max_iters full gradients of ||tx @ w - y||²/(2N).

    Return (final_w, final_full_data_mse), without mutating initial_w.
    """
    y, x, w = _inputs(y, tx, initial_w)
    _iterations(max_iters, gamma)
    for _ in range(max_iters):
        w -= gamma * (x.T @ (x @ w - y)) / y.size
    return w, _mse(y, x, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Run exactly max_iters stochastic MSE updates with batch size one.

    Draw uniformly with replacement using NumPy's global RNG. A caller can
    set np.random.seed for repeatable sampling. Return full-data final MSE.
    """
    y, x, w = _inputs(y, tx, initial_w)
    _iterations(max_iters, gamma)
    for _ in range(max_iters):
        i = np.random.randint(y.size)
        w -= gamma * x[i] * (x[i] @ w - y[i])
    return w, _mse(y, x, w)


def least_squares(y, tx):
    """Solve the normal equations and return (w, unregularized MSE).

    A singular normal system raises numpy.linalg.LinAlgError. No implicit
    regularization, matrix inverse, or numpy.linalg.lstsq is used.
    """
    y, x = _inputs(y, tx)
    w = np.linalg.solve(x.T @ x, x.T @ y)
    return w, _mse(y, x, w)


def ridge_regression(y, tx, lambda_):
    """Minimize MSE + lambda_ * ||w||² using the normal equations.

    Solve (tx.T @ tx + 2N * lambda_ * I) w = tx.T @ y. Regularize
    every supplied coordinate, including any intercept. Return data MSE.
    """
    y, x = _inputs(y, tx)
    _regularization(lambda_)
    if lambda_ > np.finfo(float).max / (2 * y.size):
        # Divide the entire normal system by 2N*lambda without ever forming
        # that overflowing product. This is the same mathematical objective.
        reciprocal = (0.5 / y.size) / lambda_
        gram = (x.T @ x) * reciprocal + np.eye(x.shape[1])
        w = np.linalg.solve(gram, (x.T @ y) * reciprocal)
    else:
        gram = x.T @ x + (2 * y.size * lambda_) * np.eye(x.shape[1])
        w = np.linalg.solve(gram, x.T @ y)
    return w, _mse(y, x, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Run max_iters full GD steps for mean binary cross-entropy (0/1 y).

    Use a stable sigmoid and logaddexp loss. Return final weights and the
    corresponding mean data loss; a zero-step run evaluates initial_w.
    """
    y, x, w = _inputs(y, tx, initial_w, logistic=True)
    _iterations(max_iters, gamma)
    for _ in range(max_iters):
        w -= gamma * (x.T @ (_sigmoid(x @ w) - y)) / y.size
    return w, _logistic_loss(y, x, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Run full GD for mean cross-entropy + lambda_ * ||w||².

    The penalty gradient is 2 * lambda_ * w on all coordinates.
    Labels must be 0/1. Return final weights and unregularized data loss.
    """
    y, x, w = _inputs(y, tx, initial_w, logistic=True)
    _iterations(max_iters, gamma)
    _regularization(lambda_)
    for _ in range(max_iters):
        grad = (x.T @ (_sigmoid(x @ w) - y)) / y.size + 2 * lambda_ * w
        w -= gamma * grad
    return w, _logistic_loss(y, x, w)

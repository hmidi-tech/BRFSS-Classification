"""Additional transparent NumPy optimizers, separate from mandatory methods."""

import time

import numpy as np


def sigmoid(z):
    z = np.asarray(z)
    if not np.issubdtype(z.dtype, np.floating):
        z = z.astype(np.float64)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1 + exp_z)
    return result


def predict_scores(x, w, family):
    if family not in {"logistic", "ridge"}:
        raise ValueError("family must be 'logistic' or 'ridge'")
    z = x @ w
    return sigmoid(z) if family == "logistic" else z


def fit_model(x, y, config):
    """Fit a declared model; historical configurations retain legacy arithmetic."""
    if config["family"] == "logistic":
        return fit_logistic(
            x,
            y,
            config["lambda"],
            config["max_iters"],
            config["tolerance"],
            precision=config.get("precision", "legacy_float32"),
        )
    if config["family"] == "ridge":
        return fit_ridge(x, y, config["lambda"])
    raise ValueError("Unknown model family")


def fit_ridge(x, y, lambda_):
    """Chunked float64 Gram accumulation for binary 0/1 regression targets.

    Penalize all weights including intercept, matching the required method.
    """
    start = time.perf_counter()
    gram = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    rhs = np.zeros(x.shape[1], dtype=np.float64)
    for i in range(0, y.size, 16384):
        block = x[i : i + 16384].astype(np.float64)
        gram += block.T @ block
        rhs += block.T @ y[i : i + len(block)]
    gram.flat[:: x.shape[1] + 1] += 2 * y.size * lambda_
    rank = None
    solver = "normal_equations_float64"
    if lambda_ == 0:
        # Optional prediction baseline: an explicit Gram pseudoinverse handles
        # redundant survey features. The mandatory least_squares uses solve.
        values, vectors = np.linalg.eigh(gram)
        keep = values > max(gram.shape) * np.finfo(float).eps * values.max()
        rank = int(keep.sum())
        w = vectors[:, keep] @ ((vectors[:, keep].T @ rhs) / values[keep])
        solver = "normal_equations_eigen_pseudoinverse"
    else:
        w = np.linalg.solve(gram, rhs)
    return w, {
        "fit_seconds": time.perf_counter() - start,
        "iterations": 1,
        "converged": True,
        "solver": solver,
        "rank": rank,
    }


def _validate_logistic(x, y, lambda_, w=None):
    x, y = np.asarray(x), np.asarray(y, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 1
        or x.shape[0] != y.size
        or not y.size
        or not x.shape[1]
    ):
        raise ValueError("Expected nonempty x (N,D) and y (N,)")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Optional logistic labels must be 0/1")
    if np.ndim(lambda_) != 0 or not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda_ must be a finite nonnegative scalar")
    for i in range(0, len(x), 16384):
        if not np.isfinite(x[i : i + 16384]).all():
            raise ValueError("Preprocessed features must be finite")
    if w is not None and (np.shape(w) != (x.shape[1],) or not np.isfinite(w).all()):
        raise ValueError("Weights must be finite with shape (D,)")
    return x, y


def _logistic_value_gradient(x, y, w, lambda_, precision):
    penalty = w.copy()
    penalty[0] = 0
    if precision == "legacy_float32":
        # Preserve the exact arithmetic that produced the frozen baseline.
        z = x @ w.astype(np.float32)
        prob = sigmoid(z)
        value = float(np.mean(np.logaddexp(0, z.astype(np.float64)) - y * z))
        grad = (x.T @ (prob - y).astype(np.float32)).astype(np.float64) / y.size
    elif precision == "float64":
        value = 0.0
        grad = np.zeros_like(w, dtype=np.float64)
        for start in range(0, y.size, 16384):
            block = np.asarray(x[start : start + 16384], dtype=np.float64)
            labels = y[start : start + len(block)]
            z = block @ w
            value += float(np.sum(np.logaddexp(0, z) - labels * z))
            grad += block.T @ (sigmoid(z) - labels)
        value /= y.size
        grad /= y.size
    else:
        raise ValueError("precision must be 'float64' or 'legacy_float32'")
    return value + lambda_ * np.dot(penalty, w), grad + 2 * lambda_ * penalty


def logistic_value_gradient(x, y, w, lambda_=1e-3, *, precision="float64"):
    """Mean binary cross-entropy plus L2, excluding coordinate zero from L2.

    Float64 evaluation uses bounded row blocks and differentiates the actual
    continuous objective. Legacy mode retains the baseline's quantized matvecs
    for reproduction, so very small finite differences need not agree with it.
    """
    x, y = _validate_logistic(x, y, lambda_, w)
    return _logistic_value_gradient(
        x, y, np.asarray(w, dtype=float), lambda_, precision
    )


def fit_logistic(
    x, y, lambda_=1e-3, max_iters=120, tolerance=1e-5, *, precision="float64"
):
    """Full-batch L-BFGS with Armijo line search for mean BCE + L2.

    Implemented here directly in NumPy; no external optimizer is used. The
    intercept is unpenalized in this optional model. Float64 objective/gradient
    evaluation is the default; legacy_float32 explicitly reproduces historical
    fits. Stored designs may remain float32. Curvature history is at most ten.
    """
    start = time.perf_counter()
    x, y = _validate_logistic(x, y, lambda_)
    if (
        isinstance(max_iters, bool)
        or not isinstance(max_iters, (int, np.integer))
        or max_iters < 0
    ):
        raise ValueError("max_iters must be a nonnegative integer")
    if np.ndim(tolerance) != 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative scalar")
    w = np.zeros(x.shape[1], dtype=np.float64)
    prevalence = np.mean(y)
    if not 0 < prevalence < 1:
        raise ValueError("Fitting an unpenalized intercept requires both classes")
    w[0] = np.log(prevalence / (1 - prevalence))

    def value_gradient(v):
        return _logistic_value_gradient(x, y, v, lambda_, precision)

    value, grad = value_gradient(w)
    history = []
    trace = []
    converged = False
    status = "max_iters"
    updates = 0
    for iteration in range(max_iters):
        norm = float(np.max(np.abs(grad)))
        trace.append(
            {"iteration": iteration, "objective": value, "gradient_inf_norm": norm}
        )
        if norm <= tolerance:
            converged = True
            status = "gradient_tolerance"
            break
        q = grad.copy()
        alphas = []
        for s, t, rho in reversed(history):
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q -= alpha * t
        if history:
            s, t, _ = history[-1]
            q *= np.dot(s, t) / np.dot(t, t)
        for (s, t, rho), alpha in zip(history, reversed(alphas)):
            q += s * (alpha - rho * np.dot(t, q))
        direction = -q
        slope = float(grad @ direction)
        if slope >= 0:
            direction = -grad
            slope = -float(grad @ grad)
            history.clear()
        step = 1.0
        for _ in range(25):
            candidate = w + step * direction
            new_value, new_grad = value_gradient(candidate)
            if np.isfinite(new_value) and new_value <= value + 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            status = "line_search_stalled"
            break
        s, t = candidate - w, new_grad - grad
        curvature = float(s @ t)
        if curvature > 1e-12 * np.linalg.norm(s) * np.linalg.norm(t) and curvature > 0:
            history.append((s, t, 1 / curvature))
            history = history[-10:]
        w, value, grad = candidate, new_value, new_grad
        updates += 1
    final_norm = float(np.max(np.abs(grad)))
    converged = final_norm <= tolerance
    if converged:
        status = "gradient_tolerance"
    if not trace or trace[-1]["iteration"] != updates:
        trace.append(
            {"iteration": updates, "objective": value, "gradient_inf_norm": final_norm}
        )
    return w, {
        "fit_seconds": time.perf_counter() - start,
        "iterations": updates,
        "converged": converged,
        "status": status,
        "objective": value,
        "gradient_inf_norm": final_norm,
        "trace": trace,
        "solver": "numpy_lbfgs_armijo",
        "intercept_penalized": False,
        "precision": precision,
    }

"""Additional transparent NumPy optimizers, separate from mandatory methods."""

import time

import numpy as np


def sigmoid(z):
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1 + exp_z)
    return result


def predict_scores(x, w, family):
    z = x @ w
    return sigmoid(z) if family == "logistic" else z


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


def fit_logistic(x, y, lambda_=1e-3, max_iters=120, tolerance=1e-5):
    """Full-batch L-BFGS with Armijo line search for mean BCE + L2.

    Implemented here directly in NumPy; no external optimizer is used. The
    intercept is unpenalized in this optional model. Float32 matrix storage
    and matvecs are used; weights, objective reductions and optimizer state
    use float64. Curvature pairs are limited to ten for bounded memory.
    """
    start = time.perf_counter()
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(x.shape[1], dtype=np.float64)
    prevalence = np.mean(y)
    w[0] = np.log(prevalence / (1 - prevalence))
    penalty = np.ones(x.shape[1])
    penalty[0] = 0

    def value_gradient(v):
        z = x @ v.astype(np.float32)
        prob = sigmoid(z)
        objective = float(
            np.mean(np.logaddexp(0, z.astype(np.float64)) - y * z)
        ) + lambda_ * np.dot(penalty * v, v)
        grad = (x.T @ (prob - y).astype(np.float32)).astype(np.float64) / y.size
        grad += 2 * lambda_ * penalty * v
        return objective, grad

    value, grad = value_gradient(w)
    history = []
    trace = []
    converged = False
    status = "max_iters"
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
    return w, {
        "fit_seconds": time.perf_counter() - start,
        "iterations": len(trace),
        "converged": converged,
        "status": status,
        "objective": value,
        "gradient_inf_norm": float(np.max(np.abs(grad))),
        "trace": trace,
        "solver": "numpy_lbfgs_armijo",
        "intercept_penalized": False,
    }

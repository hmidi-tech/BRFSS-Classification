# Mathematical and implementation review

`implementations.py` contains exactly the six required public names and their
mandated parameter order. Helpers are private. Input vectors are one dimensional;
weights are copied before updates. Empty, misaligned, nonfinite and invalid-label
inputs raise explicit errors. Iterative methods execute exactly `max_iters`
updates, including zero, with no early convergence exit.

- MSE is `sum((Xw-y)**2)/(2N)`, gradient `X.T@(Xw-y)/N`.
- SGD draws one uniform observation with replacement per update, using the
  global NumPy RNG so the caller can control `np.random.seed`. Final loss is
  the full-dataset MSE, not the last sample's loss.
- Least squares solves `X.T@X w = X.T@y`. Singular systems raise `LinAlgError`;
  there is no silent ridge penalty or forbidden `numpy.linalg.lstsq` call.
- Ridge solves `(X.T@X + 2N*lambda*I)w=X.T@y`. Every supplied coordinate,
  including an intercept, is penalized. Returned loss excludes the penalty.
- Logistic data loss is `mean(logaddexp(0,z)-y*z)`, with `y` in `{0,1}` and
  `z=Xw`. Its gradient is `X.T@(sigmoid(z)-y)/N`.
- Regularized logistic adds `lambda*sum(w**2)` to the optimization objective
  and `2*lambda*w` to the gradient. It returns data loss only.

The prediction pipeline uses additional, explicitly separate methods in
`models.py`. Logistic optimization uses a directly implemented L-BFGS two-loop
recursion (history ten) and Armijo backtracking (initial step one, sufficient
decrease `1e-4`, at most 25 halvings). The intercept is unpenalized in this
optional model. Matrix storage and optimization matrix-vector products use
float32; parameter vectors, reductions and curvature state use float64.
Predictions use stored float64 weights. Iterations are limited to 120, gradient
infinity-norm target `1e-5`. A budget stop is recorded and must not be called
convergence. No external optimizer, library ML method or pretrained external
model is used.

The optional linear prediction baseline computes a minimum-norm solution of
the normal equations via an eigen pseudoinverse of the Gram matrix when
`lambda=0`. Positive-ridge baselines use a direct solve. This handles redundant
survey features without changing the required `least_squares` interface.

Forming normal equations costs O(ND²), solving costs O(D³), and the Gram matrix
requires O(D²) storage. Squaring the design matrix's condition number can amplify
roundoff. Gram accumulation uses float64 in bounded row blocks. Ridge improves
conditioning; the eigen cutoff is `D*eps*max_eigenvalue` for the optional
unregularized baseline. A direct QR/SVD approach to the design would have better
conditioning but is not the required normal-equation method.

Independent tests check finite-difference gradients, analytical regularization,
batch-size-one SGD, zero iterations, no mutation, stable extreme logits, rank
deficiency, invalid inputs, leakage boundaries, serialization, tied-score AUC/AP,
stratification, and stationarity of the optional logistic optimizer on a toy
problem. These augment the original public tests; passing both does not prove
all possible numerical cases correct.

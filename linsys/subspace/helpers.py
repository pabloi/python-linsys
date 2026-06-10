"""Helper routines for subspace identification (port of matlab-linsys/subspace).

MATLAB sources: myhankel.m, projectMat.m, projectPerp.m, projectObliq.m,
matrixPowers.m, fitMatrixPowers.m, matrixPolyRoots.m,
estimateTransitionMatrix.m, estimateTransitionMatrixv2.m, and
observabilityMatrix.m (from EM/test, needed by the subspace algorithms).
"""
from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import numpy as np
import scipy.optimize

__all__ = [
    "my_hankel", "project_mat", "project_perp", "project_obliq",
    "matrix_powers", "fit_matrix_powers", "matrix_poly_roots",
    "estimate_transition_matrix", "estimate_transition_matrix_v2",
    "observability_matrix",
]


def _mrdivide(A, B):
    """MATLAB A/B (solve X @ B = A in the least-squares sense)."""
    return np.linalg.lstsq(B.T, A.T, rcond=None)[0].T


def _mldivide(A, B):
    """MATLAB A\\B (solve A @ X = B in the least-squares sense)."""
    return np.linalg.lstsq(A, B, rcond=None)[0]


def my_hankel(A, i, j):
    """Block-Hankel matrix of a multivariate time series (MATLAB: myhankel).

    A is (d, n) with columns as time samples. Returns H of shape (i*d, j)
    where H[:, l] is A[:, l:l+i] flattened column-major (i.e. the block
    row k of H equals A[:, k:k+j]). Requires n >= i + j - 1.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.shape[1] < i + j - 1:
        raise ValueError("my_hankel: need at least i+j-1 columns")
    return np.vstack([A[:, k:k + j] for k in range(i)])


def project_mat(A, B):
    """Orthogonal projection of the rows of A onto the row space of B
    (MATLAB: projectMat). Returns A @ pinv(B) @ B."""
    return (A @ np.linalg.pinv(B)) @ B


def project_perp(A, B):
    """Projection of the rows of A onto the orthogonal complement of the
    row space of B (MATLAB: projectPerp)."""
    return A - project_mat(A, B)


def project_obliq(A, B, C):
    """Oblique projection of the rows of A along the row space of B onto
    the row space of C (MATLAB: projectObliq).

    Returns (Ap, pB) where pB = pinv(B) (reused by callers)."""
    pB = np.linalg.pinv(B)
    ApB = A - (A @ pB) @ B
    CpB = C - (C @ pB) @ B
    Ap = (ApB @ np.linalg.pinv(CpB)) @ C
    return Ap, pB


def matrix_powers(A, n):
    """Stack powers [A; A^2; ...; A^n] into an (n*d, d) matrix
    (MATLAB: matrixPowers). A may be (d, d) or a flat length-d^2 vector
    (interpreted column-major, as in MATLAB)."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        d = int(round(np.sqrt(A.size)))
        A = A.reshape(d, d, order="F")
    d = A.shape[0]
    out = np.empty((n * d, d))
    Ak = np.eye(d)
    for k in range(n):
        Ak = Ak @ A
        out[k * d:(k + 1) * d, :] = Ak
    return out


class FitMatrixPowersResult(NamedTuple):
    A: np.ndarray
    B: Optional[np.ndarray]


def fit_matrix_powers(power_estimates, no_bias_flag=False):
    """Fit A (and bias B) such that power_estimates ~ [A; A^2; ...; A^n] @ (I - B)
    in the least-squares sense (MATLAB: fitMatrixPowers).

    power_estimates: (n*d, d). Returns FitMatrixPowersResult(A, B); B is None
    when no_bias_flag is True.
    """
    PE = np.asarray(power_estimates, dtype=float)
    d = PE.shape[1]
    n = PE.shape[0] // d
    if n * d != PE.shape[0]:
        raise ValueError("fit_matrix_powers: row count must be a multiple of "
                         "the column count")
    A0 = PE[:d, :]
    I = np.eye(d)
    if no_bias_flag:
        def res(x):
            A = x.reshape(d, d, order="F")
            return (PE - matrix_powers(A, n)).ravel()
        x0 = A0.ravel(order="F")
    else:
        def res(x):
            A = x[:d * d].reshape(d, d, order="F")
            B = x[d * d:].reshape(d, d, order="F")
            return (PE - matrix_powers(A, n) @ (I - B)).ravel()
        x0 = np.concatenate([A0.ravel(order="F"), np.zeros(d * d)])
    sol = scipy.optimize.least_squares(res, x0, xtol=1e-14, ftol=1e-14,
                                       gtol=1e-14, max_nfev=10000)
    A = sol.x[:d * d].reshape(d, d, order="F")
    B = None if no_bias_flag else sol.x[d * d:].reshape(d, d, order="F")
    return FitMatrixPowersResult(A, B)


def matrix_poly_roots(B, w, method="minDetReal"):
    """Solve the matrix polynomial equation
    B = w[0]*A^n + w[1]*A^(n-1) + ... + w[n-1]*A for A
    via eigen-decomposition (MATLAB: matrixPolyRoots).

    Of the multiple solutions, returns (per eigenvalue) the root selected by
    `method`: 'minDetReal' (default; smallest |root|, restricted to real roots
    when any exist), 'minDet' (smallest |root|), or 'minPhase' (closest to the
    real line).
    """
    B = np.asarray(B)
    w = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    d, v = np.linalg.eig(B)
    e = np.zeros(d.shape, dtype=complex)
    eps = np.finfo(float).eps
    for l, dl in enumerate(d):
        rr = np.roots(np.concatenate([w, [-dl]]))
        if method == "minPhase":
            with np.errstate(invalid="ignore", divide="ignore"):
                aa = np.abs(rr.imag) / np.abs(rr)
            idx = np.argmin(aa)
        elif method == "minDet":
            idx = np.argmin(np.abs(rr))
        elif method == "minDetReal":
            real_mask = np.abs(rr.imag) <= 1e-9 * (1.0 + np.abs(rr))
            if real_mask.any():  # always the case for real w, dl, odd order
                rr = rr[real_mask].real + 0j
            idx = np.argmin(np.abs(rr))
        else:
            raise ValueError(f"matrix_poly_roots: unknown method '{method}'")
        el = rr[idx]
        # Some polynomials (e.g. B = A^2) admit opposite-sign roots: no unique
        # real solution; pick the one with non-negative real part.
        others = np.delete(rr, idx)
        if np.any(np.abs(others + el) < 100 * np.abs(dl) * eps):
            warnings.warn("matrix_poly_roots: found opposite roots; returning "
                          "the solution with positive real parts")
            if el.real < 0:
                el = -el
        e[l] = el
    A = (v * e) @ np.linalg.inv(v)
    if np.all(np.abs(A.imag) <= 1e-10 * max(1.0, np.abs(A.real).max())):
        A = A.real
    return A


def estimate_transition_matrix(X, ord=15):
    """Estimate A such that x[k+1] = A x[k] + v[k] from a (d, N) time series,
    by regressing a weighted sum of lags and solving the matrix polynomial
    (MATLAB: estimateTransitionMatrix).

    ord: scalar order (forced odd so a real solution always exists), or a
    weight vector.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    ord_arr = np.atleast_1d(np.asarray(ord, dtype=float))
    if ord_arr.size == 1:
        m = int(ord_arr[0])
        if m % 2 == 0:
            m += 1
        weights = np.ones(m)
    else:
        weights = ord_arr.ravel()
        m = weights.size
    n = X.shape[1]
    Xpp = np.zeros((X.shape[0], n - m))
    for k in range(1, m + 1):
        # MATLAB: weights(ord-k+1) * X(:, (k+1):(end-ord+k))
        Xpp += weights[m - k] * X[:, k:n - m + k]
    An = _mrdivide(Xpp, X[:, :n - m])  # estimates sum_p weights[m-p] A^p
    return matrix_poly_roots(An, weights)


def estimate_transition_matrix_v2(X, ord=15):
    """Estimate A by simultaneously fitting regressions of all lags 1..ord
    (estimates of A, A^2, ..., A^ord) via fit_matrix_powers
    (MATLAB: estimateTransitionMatrixv2). Slower but more accurate."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m = int(ord) if np.isscalar(ord) else np.asarray(ord).size
    n = X.shape[1]
    blocks = [_mrdivide(X[:, k:], X[:, :n - k]) for k in range(1, m + 1)]
    return fit_matrix_powers(np.vstack(blocks)).A


def observability_matrix(A, C, N):
    """Extended observability matrix [C; C@A; ...; C@A^(N-1)]
    (MATLAB: observabilityMatrix)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    blocks = []
    CA = C
    for _ in range(N):
        blocks.append(CA)
        CA = CA @ A
    return np.vstack(blocks)

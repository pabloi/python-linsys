"""Core numerical utilities (port of matlab-linsys/misc).

All functions assume float64 numpy arrays. See PORTING.md for conventions.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg

__all__ = [
    "cholcov", "cholcov2", "pinvchol", "pinvchol2", "logl_normal",
    "transform", "diagonalize_a", "canonize", "substitute_nans",
    "rob_cov", "fwd_sim",
]

_HALF_LOG_2PI = 0.91893853320467268


def cholcov(A):
    """Cholesky-like factor of a PSD matrix (MATLAB: mycholcov).

    Returns (U, r) where U is an (r, n) upper-staircase matrix with
    U.T @ U == A and r = rank(A). Unlike numpy's cholesky, accepts
    semidefinite matrices.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0)), 0
    if np.all(np.isfinite(A)):
        try:
            return scipy.linalg.cholesky(A, lower=False), n
        except (scipy.linalg.LinAlgError, ValueError):
            pass
    # Incomplete Cholesky for semidefinite matrices
    # (adapted from https://arxiv.org/pdf/0804.4809.pdf, as in MATLAB source)
    t = np.trace(A)
    if not t > np.finfo(float).eps:
        return np.zeros((0, n)), 0
    tol = 1e-3 * np.sqrt(t / n)
    L1 = np.zeros((n, n))
    r = 0
    for k in range(n):
        aux = A[k, :] - L1[:, k] @ L1
        ak = aux[k]
        a = np.sqrt(ak) if ak > 0 else 0.0
        if a > tol:
            L1[r, :] = aux / a
            r += 1
    return np.triu(L1[:r, :]), r


def cholcov2(A):
    """Cholesky-like factor of a PSD matrix, square output (MATLAB: mycholcov2).

    Returns cA of shape (n, n) with cA.T @ cA == A. For rank-deficient A, cA
    contains zero rows. Handles Inf diagonal elements (decoupled components)
    with the convention Inf*0 = 0: the factor gets Inf on that diagonal entry
    and zeros elsewhere in that row/column.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    d = np.diag(A)
    inf_idx = np.isinf(d)
    if inf_idx.any():
        cA = np.zeros((n, n))
        fin = np.where(~inf_idx)[0]
        if fin.size:
            cA[np.ix_(fin, fin)] = cholcov2(A[np.ix_(fin, fin)])
        cA[inf_idx, inf_idx] = np.inf
        return cA
    try:
        return scipy.linalg.cholesky(A, lower=False)
    except (scipy.linalg.LinAlgError, ValueError):
        pass
    # Symmetric eigendecomposition fallback (MATLAB uses LDL; eigh is more
    # robust in scipy and satisfies the same contract cA.T @ cA == A).
    w, V = scipy.linalg.eigh((A + A.T) / 2)
    dth = 1e-10
    w = w.copy()
    w[np.abs(w) < dth * max(1.0, np.abs(w).max())] = 0.0
    if (w < 0).any():
        raise np.linalg.LinAlgError("cholcov2: matrix is not PSD")
    return np.sqrt(w)[:, None] * V.T


def pinvchol(A):
    """Pseudo-inverse of a PSD matrix via Cholesky (MATLAB: pinvchol).

    Returns (cInvA, cA, invA): cInvA is (n, r) with cInvA @ cInvA.T == pinv(A),
    cA is the (r, n) cholcov factor of A, invA = pinv(A).
    """
    cA, r = cholcov(A)
    if r == 0:
        n = np.atleast_2d(A).shape[0]
        cInvA = np.zeros((n, 0))
    elif r == cA.shape[1] :
        cInvA = scipy.linalg.solve_triangular(cA, np.eye(r), lower=False)
    else:
        # Rank-deficient: min-norm solution gives exact pseudo-inverse factor
        cInvA = np.linalg.pinv(cA[:r, :])
    invA = cInvA @ cInvA.T
    return cInvA, cA, invA


def pinvchol2(A):
    """Pseudo-inverse factor handling Inf/0 diagonal elements (MATLAB: pinvchol2).

    Returns (cInvA, cA, invA) with cInvA (n, n) such that
    cInvA @ cInvA.T == pinv(A), where components with infinite variance
    contribute zero information (Inf*0 = 0 convention).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    d = np.diag(A)
    inf_idx = np.isinf(d)
    cA = cholcov2(A)
    cInvA = np.zeros((n, n))
    fin = np.where(~inf_idx)[0]
    if fin.size:
        Af = A[np.ix_(fin, fin)]
        w, V = scipy.linalg.eigh((Af + Af.T) / 2)
        tol = 1e-12 * max(1.0, np.abs(w).max())
        keep = w > tol
        if keep.any():
            sub = V[:, keep] / np.sqrt(w[keep])
            cInvA[np.ix_(fin, np.arange(keep.sum()))] = sub
    invA = cInvA @ cInvA.T
    return cInvA, cA, invA


def logl_normal(y, Sigma=None, chol_inv_sigma=None):
    """Log-likelihood of zero-mean multivariate normal samples (MATLAB: logLnormal).

    y: (d, N) or (d,). Provide either Sigma or chol_inv_sigma (a matrix M with
    M @ y having identity covariance and sum(log(diag(M))) = -0.5*log(det(Sigma))).
    Returns (logL, z2), each (N,) arrays (or scalars for 1-D y).
    """
    y = np.asarray(y, dtype=float)
    scalar = y.ndim == 1
    y2 = y[:, None] if scalar else y
    if chol_inv_sigma is None:
        chol_inv_sigma = pinvchol(Sigma)[0].T
    half_logdet_sigma = np.sum(np.log(np.diag(chol_inv_sigma)))
    icSy = chol_inv_sigma @ y2
    z2 = np.sum(icSy ** 2, axis=0)
    logL = -0.5 * z2 + half_logdet_sigma - y2.shape[0] * _HALF_LOG_2PI
    if scalar:
        return logL[0], z2[0]
    return logL, z2


def transform(V, A, B=None, C=None, Q=None, X=None, P=None):
    """Similarity transform of a state-space model (MATLAB: transform).

    New state is x' = V x. Returns the transformed subset of
    (A, B, C, Q, X, P) that was provided (others returned as None).
    X may be a (nx, N) array or list thereof; P a (N, nx, nx) stack or list.
    """
    iV = np.linalg.inv(V)
    A = V @ A @ iV
    B = V @ B if B is not None else None
    C = C @ iV if C is not None else None
    Q = V @ Q @ V.T if Q is not None else None
    if X is not None:
        if isinstance(X, list):
            X = [V @ x for x in X]
        else:
            X = V @ X
    if P is not None:
        def _tp(p):
            p = np.asarray(p, dtype=float)
            if p.ndim == 2:
                return V @ p @ V.T
            return np.einsum('ij,njk,lk->nil', V, p, V)
        P = [_tp(p) for p in P] if isinstance(P, list) else _tp(P)
    return A, B, C, Q, X, P


def diagonalize_a(A):
    """Diagonalize A, using the real Jordan form for complex eigenvalues
    (MATLAB: diagonalizeA). Returns (V, J) with J = inv(V) @ A @ V (block)
    diagonal, states sorted by eigenvalue."""
    w, V = np.linalg.eig(A)
    a, b = np.imag(w), np.real(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(a) / np.where(np.abs(b) == 0, np.finfo(float).tiny, np.abs(b))
    if np.any(ratio > 1e-15):  # truly complex eigenvalues: real Jordan form
        J, V = scipy.linalg.cdf2rdf(w, V)
    else:
        V = np.real(V)
        J = np.diag(np.real(w))
    # Sort states by diagonal of J
    idx = np.argsort(np.diag(J))
    J = J[np.ix_(idx, idx)]
    V = V[:, idx]
    return V, J


def _xinf(A, B, N=None):
    """Steady-state of x under a step input on the first input component."""
    I = np.eye(A.shape[0])
    b1 = B[:, 0]
    if N is not None:
        return np.linalg.solve(I - A, (I - np.linalg.matrix_power(A, N)) @ b1)
    return np.linalg.solve(I - A, b1)


def _orthomax(C, gamma=1.0, maxit=1000, tol=1e-8):
    """Orthomax factor rotation (gamma=1: varimax, 0: quartimax).

    Returns (Cr, T) with Cr = C @ T, T orthogonal."""
    p, k = C.shape
    T = np.eye(k)
    var = 0.0
    L = C @ T
    for _ in range(maxit):
        U, s, Vt = np.linalg.svd(
            C.T @ (L ** 3 - (gamma / p) * L * np.sum(L ** 2, axis=0)))
        T = U @ Vt
        L = C @ T
        var_new = np.sum(s)
        if var_new < var * (1 + tol):
            break
        var = var_new
    return L, T


def canonize(A, B, C, X=None, Q=None, P=None, method="canonical", N=None):
    """Transform a model to a unique (canonical) representation (MATLAB: canonize).

    Returns (J, B, C, X, V, Q, P) where J = V A inv(V) etc.
    Methods: 'canonical', 'canonicalAlt', 'orthonormal', 'eyeQ', 'diagQ',
    'orthomax', 'varimax', 'quartimax'.
    """
    A = np.asarray(A, dtype=float)
    nx = A.shape[0]
    if P is None:
        P = np.zeros((nx, nx))
    if Q is None:
        Q = np.zeros((nx, nx))
    if X is None:
        X = np.zeros((nx, 1))

    if method == "canonical":
        V0, _ = diagonalize_a(A)
        J, K, *_ = transform(np.linalg.inv(V0), A, B)
        scale = _xinf(J, K, N)
        bad = ~np.isfinite(scale) | (scale == 0)
        scale[bad] = 1.0
        V = np.diag(1.0 / scale) @ np.linalg.inv(V0)
    elif method == "canonicalAlt":
        V0, _ = diagonalize_a(A)
        _, K, C1, *_ = transform(np.linalg.inv(V0), A, B, C)
        idx = np.abs(K) == np.abs(K).max(axis=1, keepdims=True)
        scale = np.sqrt(np.sum(C1 ** 2, axis=0)) * np.sign(np.sum(K * idx, axis=1))
        scale[scale == 0] = 1.0
        V = np.diag(scale) @ np.linalg.inv(V0)
    elif method == "orthonormal":
        # V = D(1:k,:) @ Vt so that C @ inv(V) has orthonormal columns
        _, s, Vt = np.linalg.svd(C)
        k = C.shape[1]
        V = np.diag(s[:k]) @ Vt[:k, :]
        J, K, *_ = transform(V, A, B)
        scale = _xinf(J, K, N)
        V = np.diag(np.sign(scale)) @ V
    elif method in ("eyeQ", "diagQ"):
        tol = 1e-9
        w, E = scipy.linalg.eigh((Q + Q.T) / 2)
        if (w < -tol).any():
            raise ValueError("canonize: Q is not PSD")
        zero_states = np.abs(w) < tol
        if zero_states.any():
            warnings.warn("canonize: cannot make Q=I, ignoring zero-variance eigen-states")
        scl = np.where(zero_states, 1.0, 1.0 / np.sqrt(np.maximum(w, tol)))
        V = scl[:, None] * E.T  # V @ Q @ V.T == I on non-degenerate eigen-states
        J, K, CC, *_ = transform(V, A, B, C)
        scale = _xinf(J, K)
        V = np.diag(np.sign(scale)) @ V
        if method == "diagQ":
            _, _, CC, *_ = transform(V, A, B, C)
            scale = np.sqrt(np.sum(CC ** 2, axis=0))
            V = np.diag(scale) @ V
    elif method in ("orthomax", "varimax", "quartimax"):
        gamma = {"orthomax": 1.0, "varimax": 1.0, "quartimax": 0.0}[method]
        Cr, T = _orthomax(C, gamma)
        scale = np.sqrt(np.sum(Cr ** 2, axis=0)) * np.sign(np.max(B, axis=1))
        scale[scale == 0] = 1.0
        V = np.diag(scale) @ np.linalg.inv(T)
    else:
        raise ValueError(f"canonize: unrecognized method '{method}'")

    A2, B2, C2, Q2, X2, P2 = transform(V, A, B, C, Q, X, P)
    return A2, B2, C2, X2, V, Q2, P2


def substitute_nans(y):
    """Column-wise linear interpolation/extrapolation of NaN values
    (MATLAB: substituteNaNs). y is (N, d); returns a new array."""
    y = np.array(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
        squeeze = True
    else:
        squeeze = False
    n = y.shape[0]
    idx = np.arange(n)
    for i in range(y.shape[1]):
        good = ~np.isnan(y[:, i])
        if good.all():
            continue
        if not good.any():
            raise ValueError("substitute_nans: column is all-NaN")
        xg, yg = idx[good], y[good, i]
        # linear interp with linear extrapolation at the edges
        yi = np.interp(idx[~good], xg, yg)
        if xg.size >= 2:
            lo = idx[~good] < xg[0]
            hi = idx[~good] > xg[-1]
            if lo.any():
                slope = (yg[1] - yg[0]) / (xg[1] - xg[0])
                yi[lo] = yg[0] + slope * (idx[~good][lo] - xg[0])
            if hi.any():
                slope = (yg[-1] - yg[-2]) / (xg[-1] - xg[-2])
                yi[hi] = yg[-1] + slope * (idx[~good][hi] - xg[-1])
        y[~good, i] = yi
    return y[:, 0] if squeeze else y


def rob_cov(w, prc=95):
    """Robust covariance estimate of samples w (d, N).

    Substitute for the external robCov() used by matlab-linsys: trims samples
    with the largest Mahalanobis scores and rescales for consistency.
    """
    w = np.asarray(w, dtype=float)
    d, N = w.shape
    Q = (w @ w.T) / N
    from scipy.stats import chi2
    for _ in range(2):
        cInv, _, _ = pinvchol(Q)
        z2 = np.sum((cInv.T @ w) ** 2, axis=0)
        th = chi2.ppf(prc / 100.0, d)
        keep = z2 <= th
        if keep.all() or not keep.any():
            break
        # consistency factor for trimming at the prc-th percentile
        c = (prc / 100.0) / chi2.cdf(th, d + 2)
        Q = c * (w[:, keep] @ w[:, keep].T) / keep.sum()
    return Q


def fwd_sim(u, A, B, C, D, x0=None, Q=None, R=None, rng=None):
    """Simulate a linear system forward (MATLAB: fwdSim).

    u: (nu, N) input. Returns (out, state): out is (ny, N), state is (nx, N+1).
    Q, R add process/observation Gaussian noise if given.
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    N = u.shape[1]
    A = np.atleast_2d(A)
    C = np.atleast_2d(C)
    nx, ny = A.shape[0], C.shape[0]
    B = np.zeros((nx, u.shape[0])) if B is None else np.atleast_2d(B)
    D = np.zeros((ny, u.shape[0])) if D is None else np.atleast_2d(D)
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    state = np.zeros((nx, N + 1))
    if x0 is not None:
        state[:, 0] = np.asarray(x0, dtype=float).ravel()
    out = np.zeros((ny, N))
    q, _ = cholcov(Q) if Q is not None else (np.zeros((0, nx)), 0)
    r, _ = cholcov(R) if R is not None else (np.zeros((0, ny)), 0)
    for k in range(N):
        out[:, k] = C @ state[:, k] + D @ u[:, k]
        if r.shape[0]:
            out[:, k] += r.T @ rng.standard_normal(r.shape[0])
        state[:, k + 1] = A @ state[:, k] + B @ u[:, k]
        if q.shape[0]:
            state[:, k + 1] += q.T @ rng.standard_normal(q.shape[0])
    return out, state

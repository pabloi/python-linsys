"""Assorted model helpers (port of matlab-linsys/misc, non-legacy).

Ported here: matchModelInputs, getFlatModel, rotateFac, myPSDsum, linearize,
calcOutput, updateState, modelCheck.

Not ported from misc/ (and why):
- autodeal.m (relies on MATLAB ``inputname`` magic; use a dict literal),
- isOctave.m (environment check, meaningless in Python),
- GetMD5.m (third-party MEX; use ``hashlib``),
- pinvldl.m (superseded by utils.pinvchol/pinvchol2 elsewhere in the port),
- mycholcov/mycholcov2/pinvchol/pinvchol2/logLnormal/transform/canonize/
  diagonalizeA/substituteNaNs/fwdSim (already in :mod:`linsys.utils`),
- dataLogLikelihood/logLincomplete/logLComplete (in :mod:`linsys.stats`),
- bicaic/foldSplit/bestPairedMatch (in :mod:`linsys.model_selection`).
"""
from __future__ import annotations

import copy
import warnings
from typing import NamedTuple

import numpy as np

from .utils import cholcov, _orthomax
from .stats import data_log_likelihood

__all__ = [
    "match_model_inputs", "FlatModel", "get_flat_model", "rotate_fac",
    "my_psd_sum", "linearize", "calc_output", "update_state", "model_check",
]


def match_model_inputs(models, inputs):
    """Pad B/D of several models so they share a common input set
    (MATLAB: matchModelInputs).

    Useful when different models were fit to the same output data using
    different input subsets.

    Parameters
    ----------
    models : list of dicts each containing at least 'B' (nx_i, nu_i) and
        'D' (ny, nu_i).
    inputs : list of (nu_i, N) arrays, the input used for each model (rows
        are input channels; all must share N).

    Returns
    -------
    (expanded_models, expanded_inputs): models are (deep) copies with B and D
    zero-padded to the unique set of input rows; expanded_inputs is the
    (n_unique, N) array of unique input rows (sorted lexicographically, as
    MATLAB's ``unique(...,'rows')``).
    """
    all_inputs = np.vstack([np.atleast_2d(np.asarray(u, dtype=float))
                            for u in inputs])
    expanded_inputs, idxs_all = np.unique(all_inputs, axis=0,
                                          return_inverse=True)
    n_inputs = expanded_inputs.shape[0]
    counts = [np.atleast_2d(u).shape[0] for u in inputs]
    splits = np.split(idxs_all, np.cumsum(counts)[:-1])
    expanded_models = []
    for model, idx in zip(models, splits):
        m = copy.deepcopy(dict(model))
        B = np.atleast_2d(np.asarray(model["B"], dtype=float))
        D = np.atleast_2d(np.asarray(model["D"], dtype=float))
        newB = np.zeros((B.shape[0], n_inputs))
        newB[:, idx] = B
        newD = np.zeros((D.shape[0], n_inputs))
        newD[:, idx] = D
        m["B"], m["D"] = newB, newD
        expanded_models.append(m)
    return expanded_models, expanded_inputs


class FlatModel(NamedTuple):
    J: np.ndarray   # (1, 1) zeros: a static (flat) model
    B: np.ndarray   # (1, nu) zeros
    C: np.ndarray   # (ny, 1) ones
    D: np.ndarray   # (ny, nu) least-squares fit Y ~ D U
    Q: np.ndarray   # (1, 1) zeros
    R: np.ndarray   # (ny, ny) residual covariance (inf outside included idx)
    logL: float     # total log-likelihood of the flat model


def get_flat_model(Y, U, include_output_idx=None) -> FlatModel:
    """Static (0-state) model fit to data; the baseline for model selection
    (MATLAB: getFlatModel).

    ``Y``/``U`` may be lists of (ny, N_i)/(nu, N_i) arrays (multiple
    realizations; they are concatenated in time). NaN samples are dropped
    for the least-squares fit. ``include_output_idx`` restricts the outputs
    whose likelihood is modeled: excluded outputs get infinite variance in R.

    Note: the MATLAB function returns ``logLperSamplePerDim`` by name but
    actually stores statKalmanFilter's TOTAL log-likelihood (via
    dataLogLikelihood); the total is returned here as well, under the
    honest name ``logL``.
    """
    if isinstance(Y, (list, tuple)):
        Y = np.hstack([np.atleast_2d(np.asarray(y, dtype=float)) for y in Y])
        U = np.hstack([np.atleast_2d(np.asarray(u, dtype=float)) for u in U])
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    ny, nu = Y.shape[0], U.shape[0]
    bad = np.isnan(Y).any(axis=0)
    Yg, Ug = Y[:, ~bad], U[:, ~bad]

    J = np.zeros((1, 1))
    B = np.zeros((1, nu))
    Q = np.zeros((1, 1))
    C = np.ones((ny, 1))
    D = np.linalg.lstsq(Ug.T, Yg.T, rcond=None)[0].T  # Y/U
    res = Yg - D @ Ug
    Raux = res @ res.T / Yg.shape[1]

    if include_output_idx is None:
        include_output_idx = np.arange(ny)
    include_output_idx = np.asarray(include_output_idx, dtype=int)
    R = np.diag(np.full(ny, np.inf))
    R[np.ix_(include_output_idx, include_output_idx)] = \
        Raux[np.ix_(include_output_idx, include_output_idx)]

    ii = include_output_idx
    logL = data_log_likelihood(Yg[ii, :], Ug, J, B, C[ii, :], D[ii, :], Q,
                               R[np.ix_(ii, ii)], None, None, "exact")
    return FlatModel(J, B, C, D, Q, R, float(logL))


def rotate_fac(CD, XU, method="orthonormal", gamma=1.0):
    """Rotate a factorization Y = CD @ XU preserving the product
    (MATLAB: rotateFac).

    All methods except 'none' first orthonormalize via the SVD of the
    product (PCA-style: columns of CD orthonormal, factors ordered by
    variance explained). Methods:

    - 'orthonormal' (default): just the SVD orthonormalization.
    - 'orthomax': orthomax rotation of CD with coefficient ``gamma``,
      factors re-sorted by the norm of the rows of XU.
    - 'varimax' / 'quartimax': orthomax with gamma = 1 / 0.
    - 'pablo': orthomax (varimax) rotation of XU' instead; columns of CD
      normalized to unit norm, factors sorted by row-norms of XU.
    - 'none': pass-through.

    Returns (CD_rot, XU_rot). (The MATLAB version prints debugging output;
    that is dropped.)
    """
    CD = np.atleast_2d(np.asarray(CD, dtype=float))
    XU = np.atleast_2d(np.asarray(XU, dtype=float))
    if method != "none":
        Y = CD @ XU
        r = CD.shape[1]
        Uy, sy, Vty = np.linalg.svd(Y, full_matrices=False)
        CD = Uy[:, :r]
        XU = sy[:r, None] * Vty[:r, :]
    if method in ("orthonormal", "none"):
        return CD, XU
    if method == "varimax":
        return rotate_fac(CD, XU, "orthomax", 1.0)
    if method == "quartimax":
        return rotate_fac(CD, XU, "orthomax", 0.0)
    if method == "orthomax":
        CDrot, T = _orthomax(CD, gamma=gamma, maxit=1000)
        XUrot = np.linalg.solve(T, XU)
        scale = np.sqrt(np.sum(XUrot ** 2, axis=1))
        idx = np.argsort(scale)[::-1]
        return CDrot[:, idx], XUrot[idx, :]
    if method == "pablo":
        XUrot, T = _orthomax(XU.T, gamma=1.0, maxit=1000)
        XUrot = XUrot.T
        CDrot = CD @ np.linalg.inv(T).T  # CD / T'
        scale = np.sqrt(np.sum(CDrot ** 2, axis=0))
        CDrot = CDrot / scale
        XUrot = scale[:, None] * XUrot
        scale = np.sqrt(np.sum(XUrot ** 2, axis=1))
        idx = np.argsort(scale)[::-1]
        return CDrot[:, idx], XUrot[idx, :]
    warnings.warn("Unrecognized rotation method, orthonormalizing.")
    return CD, XU


def _cholupdate(R, x):
    """Rank-1 update of an upper-triangular Cholesky factor:
    returns chol(R'R + x x') (MATLAB: cholupdate)."""
    R = R.copy()
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    for k in range(n):
        r = np.hypot(R[k, k], x[k])
        if r == 0:
            continue
        c, s = R[k, k] / r, x[k] / r
        row = R[k, k:].copy()
        R[k, k:] = c * row + s * x[k:]   # Givens rotation of the stacked
        x[k:] = -s * row + c * x[k:]     # rows [R[k, :]; x], zeroing x[k]
    return R


def my_psd_sum(A, B=None, vB=None):
    """Sum of two PSD matrices through successive rank-1 Cholesky updates,
    guaranteeing a PSD result (MATLAB: myPSDsum).

    Parameters
    ----------
    A : (n, n) PSD matrix, or its upper-triangular Cholesky factor (detected
        as in MATLAB: if A equals triu(A) it is presumed already factored).
    B : (n, n) hermitian PSD matrix (ignored if ``vB`` is given).
    vB : optional (n, k) factor with orthogonal columns s.t. B = vB @ vB.T.

    Returns
    -------
    (cC, C): the upper-triangular factor of A + B and the sum itself
    (cC.T @ cC == C). (MATLAB only computes C when requested; here it is
    always returned.)
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[1]
    if np.array_equal(A, np.triu(A)):  # presumed already a Cholesky factor
        cC = A.copy()
    else:
        cC, _ = cholcov(A)
    if cC.shape[0] < n:  # semidefinite: pad with zero rows
        cC = np.vstack([cC, np.zeros((n - cC.shape[0], n))])

    if vB is None:
        B = np.atleast_2d(np.asarray(B, dtype=float))
        if not np.allclose(B, B.T, atol=1e-10):
            raise ValueError("Second matrix is not hermitian.")
        cB, _ = cholcov(B)
        _, d, Vt = np.linalg.svd(cB)  # B = V diag(d^2) V'
        vB = Vt.T[:, :d.size] * d
    else:
        vB = np.atleast_2d(np.asarray(vB, dtype=float))
        G = vB.T @ vB
        if np.abs(G - np.diag(np.diag(G))).max() > 1e-9:
            raise ValueError("Third argument must have orthogonal columns")

    for j in range(vB.shape[1]):
        cC = _cholupdate(cC, vB[:, j])
    return cC, cC.T @ cC


def linearize(fun, x0, d=1e-3):
    """First-order Taylor approximation of ``fun`` around ``x0``
    (MATLAB: linearize).

    Returns C such that ``fun(x) ~ fun(x0) + C @ (x - x0)``, computed with
    forward finite differences of step ``d``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    f0 = np.asarray(fun(x0), dtype=float).ravel()
    C = np.zeros((f0.size, x0.size))
    for j in range(x0.size):
        x = x0.copy()
        x[j] += d
        C[:, j] = (np.asarray(fun(x), dtype=float).ravel() - f0) / d
    return C


def calc_output(state, in_, C, D, R=None, r=None, rng=None):
    """One noisy output sample y = C x + D u + v, v ~ N(0, R)
    (MATLAB: calcOutput).

    ``r`` may pass a precomputed ``cholcov(R)`` factor to skip the
    factorization. ``rng`` seeds the noise (MATLAB uses the global randn).
    """
    if r is None:
        r, _ = cholcov(R)
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    C = np.atleast_2d(np.asarray(C, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    return (C @ np.asarray(state, dtype=float)
            + D @ np.asarray(in_, dtype=float)
            + r.T @ rng.standard_normal(r.shape[0]))


def update_state(state, in_, A, B, Q=None, q=None, rng=None):
    """One noisy state transition x+ = A x + B u + w, w ~ N(0, Q)
    (MATLAB: updateState).

    ``q`` may pass a precomputed ``cholcov(Q)`` factor. ``rng`` seeds the
    noise.
    """
    if q is None:
        q, _ = cholcov(Q)
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    return (A @ np.asarray(state, dtype=float)
            + B @ np.asarray(in_, dtype=float)
            + q.T @ rng.standard_normal(q.shape[0]))


def model_check(A, C, Q, R, tol=1e-10):
    """Basic sanity checks of a model (MATLAB: modelCheck).

    NOTE: the MATLAB source is an EMPTY STUB (its body is only comments
    listing intended checks); this implements those listed checks and
    returns them as a dict of booleans:
    ``observable``, ``a_invertible``, ``a_diagonalizable``, ``q_psd``,
    ``q_invertible``, ``r_psd``, ``r_invertible``.
    Raises ValueError for the conditions the comments mark as required
    (Q PSD, R PSD and invertible).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    nx = A.shape[0]

    obs_mat = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(nx)])
    checks = {"observable": np.linalg.matrix_rank(obs_mat) == nx}

    checks["a_invertible"] = abs(np.linalg.det(A)) > tol
    w, V = np.linalg.eig(A)
    checks["a_diagonalizable"] = np.linalg.matrix_rank(V) == nx

    wq = np.linalg.eigvalsh((Q + Q.T) / 2)
    checks["q_psd"] = bool((wq > -tol * max(1.0, np.abs(wq).max())).all())
    checks["q_invertible"] = bool((wq > tol).all())
    wr = np.linalg.eigvalsh((R + R.T) / 2)
    checks["r_psd"] = bool((wr > -tol * max(1.0, np.abs(wr).max())).all())
    checks["r_invertible"] = bool((wr > tol).all())

    if not checks["q_psd"]:
        raise ValueError("model_check: Q is not PSD")
    if not checks["r_psd"]:
        raise ValueError("model_check: R is not PSD")
    if not checks["r_invertible"]:
        raise ValueError("model_check: R is not invertible")
    return checks

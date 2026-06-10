"""Smooth/dynamic PCA (port of matlab-linsys/sPCA/sPCAv8.m).

Fits Y ~ C @ X + D, with states following discrete linear dynamics
X[:, k+1] = J X[:, k] + B, J diagonal with real poles only (decaying
exponentials), under a constant (step) input.

Note on orientation: MATLAB sPCAv8 takes Y as (N, D) with rows as time
samples; per the package conventions (PORTING.md), this port takes
Y as (ny, N) with COLUMNS as time samples (i.e. the MATLAB Y transposed).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .estimate_dyn import estimate_dyn

__all__ = ["SPCAModel", "spca"]


class SPCAModel(NamedTuple):
    """C: (ny, rank_c) loadings (unit-norm columns); J: (order, order)
    diagonal dynamics; X: (rank_c, N) states; B: (rank_c, 1) constant state
    input; D: (ny, rank_d) constant output term; r2: fraction of (uncentered)
    variance explained."""
    C: np.ndarray
    J: np.ndarray
    X: np.ndarray
    B: np.ndarray
    D: np.ndarray
    r2: float


def _mrdivide(A, B):
    """MATLAB A/B (least squares)."""
    return np.linalg.lstsq(B.T, A.T, rcond=None)[0].T


def _mldivide(A, B):
    """MATLAB A\\B (least squares)."""
    return np.linalg.lstsq(A, B, rcond=None)[0]


def _chng_init_state(A, B, C, D, X, new_x0):
    """Re-define the initial state of an LTI-SSM with given state
    trajectories, adjusting B and D so the output is unchanged
    (MATLAB: chngInitState, from misc/forDeletion but actively used by
    sPCAv8 and CVsPCA). Returns (B1, D1, X1)."""
    old_x0 = X[:, :1]
    dx0 = np.asarray(new_x0, dtype=float).reshape(-1, 1) - old_x0
    B1 = B - (A - np.eye(A.shape[0])) @ dx0
    D1 = D - C @ dx0
    X1 = X + dx0
    return B1, D1, X1


def spca(Y, dyn_order=2, force_pcs=False, null_bd=False, output_under_rank=0,
         rng=None):
    """Smooth PCA: best-fit state-space model assuming constant input
    (MATLAB: sPCAv8).

    Estimates Y ~ C @ X + D with X[:, k+1] = J X[:, k] + B, where J has
    strictly real, distinct poles (no complex or double-pole solutions).

    Y: (ny, N) data, columns are time samples (transposed vs. MATLAB).
    dyn_order: number of dynamic states (>= 2 recommended).
    force_pcs: if True, constrain the columns of C to the subspace spanned
        by the first principal components (no refinement iterations).
    null_bd: if True, no constant terms (B = 0 forced by convention, D
        empty); states decay from x(0) = scale to 0. If False (default),
        the model is re-expressed so x(0) = 0 and states grow.
    output_under_rank: rank deficit of C relative to dyn_order.
    rng: seed or Generator for the random restarts of the dynamics fit.

    Returns SPCAModel(C, J, X, B, D, r2).
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    real_poles_only = True  # only acceptable value right now (as in MATLAB)
    rank_c = dyn_order - output_under_rank
    rank_d = 0 if null_bd else 1
    rank_cd = rank_c + rank_d
    rng = rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)

    # First solution: uncentered PCA + dynamic fit over the PC coefficients
    # (fast and good enough). MATLAB: [p,c]=pca(Y','Centered',false).
    W, s, Vt = np.linalg.svd(Y, full_matrices=False)
    CD = W[:, :rank_cd] * s[:rank_cd]   # scores (ny, rank_cd)
    P = Vt[:rank_cd, :]                 # coefficients (rank_cd, N)

    # Estimate dynamics from the PCA coefficients. estimate_dyn returns
    # states Xh decaying exponentially (plus a constant row if ~null_bd),
    # with initial states all equal to 1 and asymptote 0.
    J, X, V, K, _ = estimate_dyn(P, real_poles_only, null_bd, dyn_order,
                                 rng=rng)
    CD = CD @ np.hstack([V, K])  # rotate PCs; equivalent to (CD @ P) / X

    if not force_pcs:
        # Iterate for the optimal solution (converges very quickly):
        for _ in range(5):
            CD = _mrdivide(Y, X)  # optimal subspace given state trajectories
            if output_under_rank > 0 and CD.shape[0] >= rank_c:
                # Reduced-rank regression when dim(data) >= rank
                Yfit = CD @ X
                ww, _, _ = np.linalg.svd(Yfit, full_matrices=False)
                ww = ww[:, :rank_cd]
                CD = ww @ ww.T @ CD
            # Optimal states given the projection onto the subspace:
            J, X, V, K, _ = estimate_dyn(_mldivide(CD, Y), real_poles_only,
                                         null_bd, J, rng=rng)
            CD = CD @ np.hstack([V, K])

    # Decompose solution:
    C = CD[:, :rank_c]
    X = X[:rank_c, :]
    D = CD[:, rank_c:]  # empty (ny, 0) if null_bd is True
    B = np.zeros((X.shape[0], 1))  # by convention of estimate_dyn results

    # Normalize columns of C (and scale X accordingly):
    scale = np.sqrt(np.sum(C ** 2, axis=0))
    C = C / scale
    X = X * scale[:, None]

    # Convention when constant terms are present: re-define the initial
    # state so x(0) = 0 and states grow:
    if not null_bd:
        B, D, X = _chng_init_state(J, B, C, D, X, np.zeros(X.shape[0]))
        B = -B
        C = -C
        X = -X

    # Reconstruction quality:
    ones = np.ones((D.shape[1], X.shape[1]))
    resid = Y - np.hstack([C, D]) @ np.vstack([X, ones])
    r2 = 1.0 - np.linalg.norm(resid, "fro") ** 2 \
        / np.linalg.norm(Y, "fro") ** 2

    return SPCAModel(C=C, J=J, X=X, B=B, D=D, r2=float(r2))

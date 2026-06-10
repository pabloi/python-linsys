"""Subspace identification algorithms (port of matlab-linsys/subspace).

MATLAB sources: subspaceID.m, subspaceIDunbiased.m, subspaceIDv2.m.

Not ported:
- subspaceIDalt.m: superseded by subspaceIDunbiased.m and contains a
  dimension bug in the B/D estimation loop (the ``[N1;N2]*F`` product only
  conforms for kk==1; subspaceIDunbiased.m fixes it by zero-padding).
- subspaceIDhyb.m / subspaceEMhybrid.m (identical files): require the EM
  module, which is not ported yet.

All functions take Y (ny, N) and U (nu, N) with columns as time samples.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import numpy as np

from .helpers import (
    _mldivide, _mrdivide, fit_matrix_powers, my_hankel, observability_matrix,
    project_mat, project_obliq,
)

__all__ = ["SubspaceIDResult", "subspace_id", "subspace_id_unbiased",
           "subspace_id_v2"]


class SubspaceIDResult(NamedTuple):
    """Identified state-space model. S is the process/observation noise
    cross-covariance (None where the MATLAB source does not compute it)."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    X: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    S: Optional[np.ndarray]


def _check_args(Y, U, nx, i):
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    if Y.shape[1] != U.shape[1]:
        raise ValueError("subspace_id: Y and U must have the same number of "
                         "columns (time samples)")
    if i is None:
        i = 10  # Arbitrary choice, only criterion is i > nx (as in MATLAB)
    if nx is None:
        warnings.warn("Automatic state number detection not implemented, "
                      "using 2")
        nx = 2
    j = Y.shape[1] - 2 * i
    if j < 2:
        raise ValueError(f"subspace_id: not enough samples (N={Y.shape[1]}) "
                         f"for horizon i={i}; need N > 2*i + 1")
    return Y, U, int(nx), int(i), j


def subspace_id(Y, U, nx=None, i=None):
    """Subspace identification, biased algorithm (MATLAB: subspaceID).

    Follows Shadmehr & Mussa-Ivaldi 2012, i.e. Algorithm 2 of Chapter 4 of
    Van Overschee & De Moor 1996 (a biased estimate in the presence of
    process noise). The SVD weighting corresponds to W1 = I and
    W2 = projector orthogonal to the future inputs.

    Y: (ny, N), U: (nu, N), nx: model order, i: block-Hankel horizon
    (default 10; requirement is nx < i << N).
    Returns SubspaceIDResult(A, B, C, D, X, Q, R, S).
    """
    Y, U, nx, i, j = _check_args(Y, U, nx, i)

    Y_p = my_hankel(Y, i, j)
    U_p = my_hankel(U, i, j)
    W_p = np.vstack([U_p, Y_p])
    U_f = my_hankel(U[:, i:], i, j)
    Y_f = my_hankel(Y[:, i:], i, j)

    # Oblique projection of future outputs along future inputs onto the past
    O_ip1, pinvU = project_obliq(Y_f, U_f, W_p)
    # SVD of the projection with the future-input component removed
    T, s, Vt = np.linalg.svd(O_ip1 - (O_ip1 @ pinvU) @ U_f,
                             full_matrices=False)
    V = Vt.T

    ss = np.sqrt(s[:nx])
    X_ip1 = ss[:, None] * V[:-1, :nx].T   # states at times i .. i+j-2
    X_ip2 = ss[:, None] * V[1:, :nx].T    # states at times i+1 .. i+j-1
    Y_ip1 = Y[:, i:i + j - 1]
    U_ip1 = U[:, i:i + j - 1]
    AB_CD = _mrdivide(np.vstack([X_ip2, Y_ip1]), np.vstack([X_ip1, U_ip1]))

    A = AB_CD[:nx, :nx]
    B = AB_CD[:nx, nx:]
    C = AB_CD[nx:, :nx]
    D = AB_CD[nx:, nx:]
    z = X_ip2 - A @ X_ip1 - B @ U_ip1
    w = Y_ip1 - C @ X_ip1 - D @ U_ip1
    Q = z @ z.T / z.shape[1]
    R = w @ w.T / w.shape[1]
    S = z @ w.T / w.shape[1]
    # Estimate states for all datapoints:
    X = _mldivide(C, Y - D @ U)
    return SubspaceIDResult(A, B, C, D, X, Q, R, S)


def subspace_id_unbiased(Y, U, nx=None, i=None):
    """Subspace identification, unbiased robust algorithm
    (MATLAB: subspaceIDunbiased).

    Implements Algorithm 1, Chapter 4, of Van Overschee & De Moor 1996, with
    most of the "robust algorithm" improvements of the same chapter
    (W1 = I, W2 = projector orthogonal to future inputs; B and D estimated
    directly from the residuals instead of via the Kalman-gain matrix; the
    observability matrix is recomputed from the estimated A, C).

    Y: (ny, N), U: (nu, N), nx: model order, i: block-Hankel horizon.
    Returns SubspaceIDResult(A, B, C, D, X, Q, R, S).
    """
    Y, U, nx, i, j = _check_args(Y, U, nx, i)
    ny = Y.shape[0]
    nu = U.shape[0]

    Y_p = my_hankel(Y, i, j)
    U_p = my_hankel(U, i, j)
    W_p = np.vstack([U_p, Y_p])
    U_f = my_hankel(U[:, i:], i, j)
    Y_f = my_hankel(Y[:, i:], i, j)

    O_i, pinvUf = project_obliq(Y_f, U_f, W_p)
    Z_i = project_mat(Y_f, np.vstack([W_p, U_f]))

    Y_pp = my_hankel(Y, i + 1, j)
    U_pp = my_hankel(U, i + 1, j)
    W_pp = np.vstack([U_pp, Y_pp])
    Z_ip1 = project_mat(Y_f[ny:, :], np.vstack([W_pp, U_f[nu:, :]]))

    T, s, _ = np.linalg.svd(O_i - (O_i @ pinvUf) @ U_f, full_matrices=False)

    L_i = T[:, :nx] * np.sqrt(s[:nx])      # extended observability matrix
    L_im1 = L_i[:-ny, :]

    pLim1 = np.linalg.pinv(L_im1)
    M = np.vstack([pLim1 @ Z_ip1, Y_f[:ny, :]])
    pLi = np.linalg.pinv(L_i)
    RR = np.vstack([pLi @ Z_i, U_f])
    ACK = M @ np.linalg.pinv(RR)
    A = ACK[:nx, :nx]
    C = ACK[nx:, :nx]

    # Optional improvement: recompute L_i from (A, C) for better performance
    L_i = observability_matrix(A, C, i)
    L_im1 = L_i[:-ny, :]
    pLim1 = np.linalg.pinv(L_im1)
    pLi = np.linalg.pinv(L_i)
    Sres = M - np.vstack([A, C]) @ pLi @ Z_i

    # B, D directly from the residuals (robust-algorithm recommendation)
    L = np.vstack([A, C]) @ pLi
    M_L = np.hstack([np.zeros((nx, ny)), pLim1]) - L[:nx, :]
    I_L = np.hstack([np.eye(ny), np.zeros((ny, ny * (i - 1)))]) - L[nx:, :]
    F = np.block([
        [np.eye(ny), np.zeros((ny, nx))],
        [np.zeros((L_im1.shape[0], ny)), L_im1],
    ])
    QN = np.zeros((j * (nx + ny), nu * (ny + nx)))
    for kk in range(i):
        N1 = M_L[:, kk * ny:]
        N2 = I_L[:, kk * ny:]
        G = np.hstack([np.vstack([N1, N2]),
                       np.zeros((nx + ny, kk * ny))]) @ F
        QN += np.kron(U_f[kk * nu:(kk + 1) * nu, :].T, G)
    DB = np.linalg.pinv(QN) @ Sres.ravel(order="F")
    D = DB[:ny * nu].reshape(ny, nu, order="F")
    B = DB[ny * nu:].reshape(nx, nu, order="F")

    # Noise covariances from the regression residuals
    zw = M - ACK @ RR
    z = zw[:nx, :]
    w = zw[nx:, :]
    Q = z @ z.T / z.shape[1]
    R = w @ w.T / w.shape[1]
    S = z @ w.T / w.shape[1]
    # Estimate states for all datapoints:
    X = _mldivide(C, Y - D @ U)
    return SubspaceIDResult(A, B, C, D, X, Q, R, S)


def subspace_id_v2(Y, U, nx=None, i=40):
    """Subspace identification, alternate (MATLAB: subspaceIDv2).

    Estimates C, D from a state-coefficient regression and A by fitting the
    powers of A read off the left singular vectors of the oblique projection.
    The MATLAB source notes this method performs worse than subspaceID
    (especially with non-zero process noise); ported for completeness.

    Y: (ny, N), U: (nu, N), nx: model order, i: horizon (default 40 as in
    MATLAB; must satisfy i > nx + 1).
    Returns SubspaceIDResult(A, B, C, D, X, Q, R, None).
    """
    Y, U, nx, i, j = _check_args(Y, U, nx, i)
    ny = Y.shape[0]
    N = Y.shape[1]

    Y_p = my_hankel(Y, i, j)
    U_p = my_hankel(U, i, j)
    W_p = np.vstack([U_p, Y_p])
    U_f = my_hankel(U[:, i:], i, j)
    Y_f = my_hankel(Y[:, i:], i, j)

    # Output that can be explained by a lagged history of output and input
    O_ip1, pinvU = project_obliq(Y_f, U_f, W_p)
    P, s, Vt = np.linalg.svd(O_ip1, full_matrices=False)
    V = Vt.T

    X_ip1 = s[:nx, None] * V[:-1, :nx].T
    X_ip2 = s[:nx, None] * V[1:, :nx].T
    Y_ip1 = Y[:, i:i + j - 1]
    U_ip1 = U[:, i:i + j - 1]

    CD = _mrdivide(Y_ip1, np.vstack([X_ip1, U_ip1]))
    C = CD[:, :nx]
    D = CD[:, nx:]

    # Estimate states for all datapoints:
    X = _mldivide(C, Y - D @ U)

    # Observation-noise residuals (note: MATLAB normalizes by N, the full
    # sample count, not by the number of residual columns):
    w = Y_ip1 - C @ X_ip1 - D @ U_ip1
    R = w @ w.T / N

    # Estimate A from the powers of A contained in the observability matrix
    sz = i - 1  # needs to be larger than nx for a unique solution
    blk = P[ny:ny * sz, :nx]                              # (ny*(sz-1), nx)
    P2 = blk.reshape(ny, sz - 1, nx, order="F").transpose(0, 2, 1)
    IA = _mldivide(C, P2.reshape(ny, nx * (sz - 1), order="F"))
    IA = IA.reshape(nx, nx, sz - 1, order="F")  # stack of A, A^2, ...
    ord_ = 1
    IAr = IA[:, :, :ord_].transpose(0, 2, 1).reshape(nx * ord_, nx,
                                                     order="F")
    A = fit_matrix_powers(IAr, no_bias_flag=True).A
    B = _mrdivide(X_ip2 - A @ X_ip1, U_ip1)

    z = X_ip2 - A @ X_ip1 - B @ U_ip1
    Q = z @ z.T / N
    return SubspaceIDResult(A, B, C, D, X, Q, R, None)

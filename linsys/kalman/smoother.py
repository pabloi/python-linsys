"""Stationary Kalman (RTS) smoother (port of statKalmanSmoother)."""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from ..utils import cholcov, cholcov2, pinvchol, pinvchol2
from .filter import filter_stationary, FilterResult
from .opts import KalmanOpts, process_kalman_inputs, process_fast_flag


class SmootherResult(NamedTuple):
    Xs: np.ndarray       # (nx, N) smoothed state estimates
    Ps: np.ndarray       # (N, nx, nx) smoothed covariances
    Pt: np.ndarray       # (N-1, nx, nx) smoothed transition covs cov(x[k+1],x[k])
    Xf: np.ndarray       # filtered states
    Pf: np.ndarray       # filtered covariances
    Xp: np.ndarray       # one-step-ahead predictions
    Pp: np.ndarray       # prediction covariances
    rejected: np.ndarray
    logL: float


def _back_step_rts(pp, pf, ps, xp, xf, prev_xs, A, cQ, bu, iA):
    """Rauch-Tung-Striebel backward recursion step (MATLAB: backStepRTS).

    Returns (new_Ps, new_xs, new_Pt, H)."""
    cPs, _ = cholcov(ps)
    inf_idx = np.diag(pp) > 1e4 * np.diag(ps)
    invertible_A = np.linalg.norm(A @ iA - np.eye(A.shape[0])) < 1e-9
    if not inf_idx.any():
        # Standard case: numerically well-conditioned prior from the filter
        icP, _, _ = pinvchol(pp)
        HcP = pf @ (A.T @ icP)
        H = HcP @ icP.T
        new_xs = xf + H @ (prev_xs - xp)
        new_Pt = ps @ H.T
        HcPs = H @ cPs.T
        new_Ps = HcPs @ HcPs.T + (pf - HcP @ HcP.T)
        return new_Ps, new_xs, new_Pt, H
    elif invertible_A:
        # Filtering started from an improper prior; alternate well-conditioned
        # form requiring inv(A) (MATLAB: backStepRTS_invA)
        icP, _, _ = pinvchol2(pp)
        return _back_step_rts_inv_a(icP, cPs, xp, prev_xs, cQ, bu, iA)
    else:
        warnings.warn("Improper priors with non-invertible A; will not smooth "
                      "last few samples.")
        return pf, xf, np.full_like(pf, np.nan), np.zeros_like(pf)


def _back_step_rts_inv_a(inv_chol_pp, chol_ps, xp, prev_xs, chol_q, bu, iA):
    """RTS-equivalent backward step assuming A invertible; tolerates infinite
    elements in Pp (MATLAB: backStepRTS_invA)."""
    icP = inv_chol_pp
    iAcQ = iA @ chol_q.T
    cQcP = chol_q @ icP
    F = iAcQ @ cQcP @ icP.T
    H = iA - F
    HcPs = H @ chol_ps.T
    new_Pt = chol_ps.T @ HcPs.T
    new_Ps = HcPs @ HcPs.T + iAcQ @ (np.eye(iA.shape[0]) - cQcP @ cQcP.T) @ iAcQ.T
    new_xs = iA @ (prev_xs - bu) + F @ (xp - prev_xs)
    return new_Ps, new_xs, new_Pt, H


def smoother_stationary(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None,
                        U=None, opts: KalmanOpts | None = None) -> SmootherResult:
    """Stationary Kalman smoother (MATLAB: statKalmanSmoother).

    Forward pass via filter_stationary, backward pass via the RTS recursion.
    With opts.fast_flag, steady-state smoothing is used for the middle samples.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ny, N = Y.shape
    nx = A.shape[0]
    x0, P0, B, D, U, opts = process_kalman_inputs(nx, N, x0, P0, B, D, U, opts,
                                                  ny=ny)
    M = process_fast_flag(opts.fast_flag, A, N)
    BU = B @ U

    fres = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U,
                             opts.copy(fast_flag=M + 1))
    Xf, Pf, Xp, Pp, rejected, logL = fres

    Xs = np.full_like(Xf, np.nan)
    Ps = np.full_like(Pf, np.nan)
    Pt = np.full((N - 1, nx, nx), np.nan)
    prev_xs = Xf[:, N - 1].copy()
    prev_ps = Pf[N - 1].copy()
    Xs[:, N - 1] = prev_xs
    Ps[N - 1] = prev_ps

    # Separate samples into exact and steady-state (fast) smoothing intervals:
    M1 = M2 = M
    n_fast = N - 1 - (M1 + M2)
    if n_fast <= 0:
        M1, M2, n_fast = N - 1, 0, 0

    try:
        iA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        warnings.warn("A is singular; using pseudo-inverse in the smoother.")
        iA = np.linalg.pinv(A)
    cQ = cholcov2(Q)

    pp = Pp[N - 1]
    pf = Pf[N - 1]
    H = None
    last_i = None
    # Exact smoothing for the last M1 samples:
    for i in range(N - 2, N - M1 - 2, -1):
        if i < 0:
            break
        xf, pf = Xf[:, i], Pf[i]
        xp, pp = Xp[:, i + 1], Pp[i + 1]
        bu = BU[:, i]
        prev_ps, prev_xs, new_pt, H = _back_step_rts(
            pp, pf, prev_ps, xp, xf, prev_xs, A, cQ, bu, iA)
        Xs[:, i] = prev_xs
        Pt[i] = new_pt
        Ps[i] = prev_ps
        last_i = i

    if n_fast > 0:
        # Steady-state smoothing for the middle samples, reusing gain H:
        i = last_i
        _, _, new_pt, H = _back_step_rts(pp, pf, prev_ps, Xp[:, i + 1], Xf[:, i],
                                         prev_xs, A, cQ, BU[:, i], iA)
        if np.any(np.abs(np.linalg.eigvals(H)) > 1):
            warnings.warn("statKS:unstableSmooth - unstable smoothing, "
                          "skipping the backward pass.")
            H = np.zeros_like(H)
        aux = Xf - H @ Xp[:, 1:]
        for i in range(N - M1 - 2, M2 - 1, -1):
            prev_xs = aux[:, i] + H @ prev_xs
            Xs[:, i] = prev_xs
        Ps[M2:N - M1 - 1] = prev_ps
        Pt[M2:N - M1 - 1] = new_pt

    # Exact smoothing for the first M2 samples:
    for i in range(M2 - 1, -1, -1):
        xf, pf = Xf[:, i], Pf[i]
        xp, pp = Xp[:, i + 1], Pp[i + 1]
        bu = BU[:, i]
        prev_ps, prev_xs, new_pt, H = _back_step_rts(
            pp, pf, prev_ps, xp, xf, prev_xs, A, cQ, bu, iA)
        Xs[:, i] = prev_xs
        Pt[i] = new_pt
        Ps[i] = prev_ps

    return SmootherResult(Xs, Ps, Pt, Xf, Pf, Xp, Pp, rejected, logL)

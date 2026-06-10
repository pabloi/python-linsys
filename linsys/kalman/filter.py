"""Stationary Kalman filter (port of statKalmanFilter)."""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy.stats import chi2

from ..utils import logl_normal
from .core import kf_update, kf_predict, reduce_model, info_to_state, state_to_info
from .opts import KalmanOpts, process_kalman_inputs, process_fast_flag


class FilterResult(NamedTuple):
    X: np.ndarray        # (nx, N) filtered state estimates
    P: np.ndarray        # (N, nx, nx) filtered covariances
    Xp: np.ndarray       # (nx, N+1) one-step-ahead predictions
    Pp: np.ndarray       # (N+1, nx, nx) prediction covariances
    rejected: np.ndarray  # (N,) outlier-rejected samples
    logL: float          # total log-likelihood of data


def filter_stationary(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None, U=None,
                      opts: KalmanOpts | None = None) -> FilterResult:
    """Stationary Kalman filter (MATLAB: statKalmanFilter).

    Model: x[k+1] = A x[k] + B u[k] + w, w~N(0,Q); y[k] = C x[k] + D u[k] + v,
    v~N(0,R); x[0] ~ N(x0, P0). The default initial condition is an improper
    flat prior (P0 = inf*I), handled through an information-filter pass.
    NaN samples in Y are skipped (missing data). With opts.fast_flag, exact
    filtering runs only for the first M samples and steady-state gains after.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    ny, N = Y.shape
    nx = A.shape[0]
    x0, P0, B, D, U, opts = process_kalman_inputs(nx, N, x0, P0, B, D, U, opts,
                                                  ny=ny)
    M = process_fast_flag(opts.fast_flag, A, N)
    if M != N and np.isnan(Y).any():
        warnings.warn("statKFfast:NaN - requested fast filtering but data "
                      "contains NaNs; no steady-state exists, filtering will "
                      "not be exact.")

    Xp = np.full((nx, N + 1), np.nan)
    X = np.full((nx, N), np.nan)
    Pp = np.full((N + 1, nx, nx), np.nan)
    P = np.full((N, nx, nx), np.nan)
    rejected = np.zeros(N, dtype=bool)

    prev_x = x0.copy()
    prev_P = P0.copy()
    Xp[:, 0] = x0
    Pp[0] = P0

    Y_D = Y - D @ U
    BU = B @ U

    # Components of the output with infinite variance carry no information:
    inf_var = np.isinf(np.diag(R))
    if inf_var.any():
        warnings.warn("statKF:infObsVar - model has infinite variance for some "
                      "observation components; ignoring those components.")
        Y_D = Y_D[~inf_var, :]
        R = R[np.ix_(~inf_var, ~inf_var)]
        C = C[~inf_var, :]
        ny = Y_D.shape[0]
    wR = np.linalg.eigvalsh((R + R.T) / 2)
    if (wR <= 0).any():
        raise ValueError("statKF:zeroObsVar - model has 0 observation variance "
                         "for some dimension, incompatible with the Kalman "
                         "framework. Try reducing the model.")
    if (wR < 1e-8).any():
        warnings.warn("statKF:smallObsVar - observation variance along some "
                      "dimension is very small; expect numerical issues.")

    logL = np.full(N, np.nan)
    rej_threshold = chi2.ppf(0.99, ny) if opts.outlier_flag else None

    # Reduce model if convenient for efficiency:
    if ny > nx and not opts.no_reduce_flag:
        CtRinvC, _, CtRinvY, _, logl_margin, _ = reduce_model(C, R, Y_D)
        C, R, Y_D = CtRinvC, CtRinvC, CtRinvY
        ny = nx
        rejected = rejected[:N]
    else:
        logl_margin = 0.0

    # Improper (infinite-variance) prior: run information filter until all
    # uncertainties are finite (requires nx non-NaN samples if observable).
    first_ind = 0
    if np.isinf(np.diag(prev_P)).any():
        from .info import true_info_filter
        good = ~np.isnan(Y_D).any(axis=0)
        # number of samples needed to resolve infinite uncertainty:
        hit = np.nonzero(np.cumsum(good) >= nx)[0]
        n_samp = (hit[0] + 1) if hit.size else N
        CtRinvC2, _, CtRinvY2, _, _, _ = reduce_model(C, R, Y_D[:, :n_samp])
        i0, I0 = state_to_info(prev_x, prev_P)
        ii, I, ip, Ip = true_info_filter(CtRinvY2, CtRinvC2, A, Q,
                                         BU[:, :n_samp], i0, I0)
        for k in range(n_samp):
            logL[k] = -np.inf  # improper prior: no proper likelihood here
            X[:, k], P[k] = info_to_state(ii[:, k], I[k])
            prev_x, prev_P = info_to_state(ip[:, k + 1], Ip[k + 1])
            Xp[:, k + 1] = prev_x
            Pp[k + 1] = prev_P
        first_ind = n_samp

    # Ensure at least nx non-NaN samples are processed exactly:
    if first_ind + 1 < N:
        good = ~np.isnan(Y_D[:, first_ind + 1:]).any(axis=0)
        hit = np.nonzero(np.cumsum(good) >= nx)[0]
        if hit.size:
            M = max(M, first_ind + hit[0] + 2)
    M = min(M, N)

    do_logl = not opts.no_logl
    iL = None
    for i in range(first_ind, M):
        y = Y_D[:, i]
        if not np.isnan(y).any():
            prev_x, prev_P, iL, rej, ll = kf_update(
                C, R, y, prev_x, prev_P, rej_threshold, want_logl=do_logl)
            rejected[i] = rej
            if do_logl:
                logL[i] = ll
        X[:, i] = prev_x
        P[i] = prev_P
        prev_x, prev_P = kf_predict(A, Q, prev_x, prev_P, BU[:, i])
        Xp[:, i + 1] = prev_x
        Pp[i + 1] = prev_P

    if M < N:  # fast (steady-state) filtering for remaining steps
        if opts.outlier_flag:
            raise ValueError("KFfilter:outlierRejectFast - outlier rejection "
                             "is incompatible with fast mode.")
        prev_x = X[:, M - 1].copy()
        P_steady = P[M - 1]
        Pp_steady = prev_P
        CicS = C.T @ iL
        K_steady = Pp_steady @ C.T @ (iL @ iL.T)
        G_steady = np.eye(nx) - Pp_steady @ (CicS @ CicS.T)
        GBU_KY = G_steady @ BU[:, M - 1:N - 1] + K_steady @ Y_D[:, M:N]
        GA = G_steady @ A
        P[M:] = P_steady
        for i in range(M, N):
            gbu_ky = GBU_KY[:, i - M]
            if not np.isnan(gbu_ky).any():
                prev_x = GA @ prev_x + gbu_ky  # predict + update combined
            else:
                prev_x = A @ prev_x + BU[:, i]  # predict only (missing data)
            X[:, i] = prev_x
        Xp[:, 1:] = A @ X + BU
        Pp[M + 1:] = A @ P_steady @ A.T + Q
        if do_logl:
            innov = Y_D - C @ Xp[:, :-1]
            ll, _ = logl_normal(innov[:, M:], chol_inv_sigma=iL.T)
            logL[M:] = ll

    if do_logl:
        aux = logL + logl_margin
        total = np.nansum(aux[first_ind:])
    else:
        total = np.nan
    return FilterResult(X, P, Xp, Pp, rejected, float(total))

"""Cheng & Sabes (2006) "lds-1.1" Kalman smoother.

Port of statKalmanSmootherCS2006Matlab together with the single-realization
core of ext/lds-1.1/SmoothLDSMatlab (Copyright (C) 2005 Philip N. Sabes, GPL;
the E-step of the EM algorithm of Shumway & Stoffer 1982 / Ghahramani &
Hinton 1996). The mex-based statKalmanSmootherCS2006 wrapper is equivalent
and is not ported separately.

Notes on deviations from MATLAB:
- logL returned here is the TOTAL log-likelihood of the data (matching
  filter_stationary/smoother_stationary), not the per-sample-per-dimension
  average returned by the MATLAB wrapper.
- The -ny/2*log(2*pi) constant is added only for non-missing samples (MATLAB
  adds it for every sample, including NaN ones that contribute no other term).
- On missing (NaN) samples, the filtered estimate is set to the predicted one.
  SmoothLDSMatlab leaves Xc/Vc at zero in that case, which breaks the
  backward pass (an upstream bug).
- statKalmanSmootherCS2006Matlab's improper-prior workaround loop
  (`while any(infVariances)`) never updates its condition (infinite loop in
  MATLAB); the intended single substitution P0 -> 1e9*I, x0 -> 0 is applied.
- The MATLAB wrapper passes the original D twice when the model is not
  reduced (once through Y-D*U and once through SmoothLDS's feed-through
  term), subtracting the input effect twice; here it is subtracted once.
- Pt uses this package's convention: Pt[k] = cov(x[k+1], x[k]), stack of
  length N-1 (SmoothLDSMatlab stores it at slice k+1 of an N-deep stack).
"""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from .core import reduce_model
from .opts import KalmanOpts, process_kalman_inputs

_LOG_2PI = 1.83787706640934529


class CS2006Result(NamedTuple):
    Xs: np.ndarray   # (nx, N) smoothed state estimates
    Ps: np.ndarray   # (N, nx, nx) smoothed covariances
    Pt: np.ndarray   # (N-1, nx, nx) smoothed transition covs cov(x[k+1],x[k])
    logL: float      # total log-likelihood of the data


def smoother_stationary_cs2006(Y, A, C, Q, R, x0=None, P0=None, B=None,
                               D=None, U=None,
                               opts: KalmanOpts | None = None) -> CS2006Result:
    """Stationary Kalman (RTS) smoother, lds-1.1 implementation
    (MATLAB: statKalmanSmootherCS2006Matlab / SmoothLDSMatlab).

    Same model and conventions as smoother_stationary. Improper priors
    (P0 = inf*I) are replaced by the large-but-finite P0 = 1e9*I, x0 = 0,
    as in the MATLAB wrapper. No fast (steady-state) mode.
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

    # Reduce model if useful:
    Y_D = Y - D @ U
    if not opts.no_reduce_flag and ny > nx:
        CtRinvC, _, CtRinvY, _, logl_margin, _ = reduce_model(C, R, Y_D)
        C, R, Y_D = CtRinvC, CtRinvC, CtRinvY
        ny = nx
    else:
        logl_margin = 0.0

    # CS2006 does not support improper (or very large) priors:
    if np.isinf(np.diag(P0)).any():
        warnings.warn("CS2006 smoother does not support improper priors; "
                      "using P0 = 1e9*I, x0 = 0 instead.")
        P0 = 1e9 * np.eye(nx)
        x0 = np.zeros(nx)

    BU = B @ U
    I = np.eye(nx)

    # Forward pass (Kalman filter):
    Xp = np.zeros((nx, N))
    Vp = np.zeros((N, nx, nx))
    Xc = np.zeros((nx, N))
    Vc = np.zeros((N, nx, nx))
    lik = 0.0
    for t in range(N):
        if t > 0:
            Xp[:, t] = A @ Xc[:, t - 1] + BU[:, t - 1]
            Vp[t] = Q + A @ Vc[t - 1] @ A.T
        else:
            Xp[:, 0] = x0
            Vp[0] = P0
        y = Y_D[:, t]
        if not np.isnan(y).any():
            Rp = C @ Vp[t] @ C.T + R
            invRp = np.linalg.inv(Rp)
            K = Vp[t] @ C.T @ invRp
            innov = y - C @ Xp[:, t]
            Xc[:, t] = Xp[:, t] + K @ innov
            Vc[t] = (I - K @ C) @ Vp[t]
            sign, logdetRp = np.linalg.slogdet(Rp)
            if sign <= 0:
                warnings.warn("CS2006: innovation covariance is not positive "
                              "definite")
            lik += (-0.5 * logdetRp - 0.5 * innov @ invRp @ innov
                    - 0.5 * ny * _LOG_2PI)
        else:
            # Missing sample: filtered = predicted (fixes upstream MATLAB bug
            # which left these at zero, breaking the backward pass).
            Xc[:, t] = Xp[:, t]
            Vc[t] = Vp[t]

    # Backward pass (RTS):
    Xs = Xc.copy()
    Ps = Vc.copy()
    Pt = np.full((N - 1, nx, nx), np.nan)
    for t in range(N - 2, -1, -1):
        J = Vc[t] @ A.T @ np.linalg.inv(Vp[t + 1])
        Xs[:, t] = Xc[:, t] + J @ (Xs[:, t + 1] - Xp[:, t + 1])
        Ps[t] = Vc[t] + J @ (Ps[t + 1] - Vp[t + 1]) @ J.T
        Pt[t] = Ps[t + 1] @ J.T  # cov(x[t+1], x[t] | all data)

    logL = lik + float(np.nansum(logl_margin))
    return CS2006Result(Xs, Ps, Pt, float(logL))

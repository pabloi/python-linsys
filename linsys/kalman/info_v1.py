"""Legacy (v1) information-form Kalman filter/smoother.

Ports of infoUpdate2, statInfoFilter and statInfoSmoother. The MATLAB
statInfoFilter is marked deprecated upstream (in favor of statInfoFilter2,
ported as info.info_filter_stationary) but its algorithm — a hybrid that uses
the information form for the measurement update and the covariance form for
the prediction — is ported here for completeness. statInfoSmoother is a thin
wrapper around statInfoSmoother2 that converts the information-form outputs
back to state space.

Notes on deviations from MATLAB:
- logL is returned as the TOTAL log-likelihood (sum over samples), matching
  filter_stationary, instead of MATLAB's per-sample per-output-dimension
  average.
- The measurement update is a corrected port of infoUpdate.m: it uses the
  factor of newI (not of newP) in the log-determinant term. (The existing
  core.info_update port uses chol_P — the factor of the posterior covariance —
  which flips the sign of logdet(newI); see the bug report accompanying this
  port. States/covariances are unaffected, only logL.)
- On missing samples the prior/posterior information matrices stored in the
  Ip/I outputs are recomputed from the current covariance (MATLAB stores
  stale values from the last updated sample).
- MATLAB declares Ip with N+1 slots but fills only N; here Ip is (N, nx, nx),
  with Ip[k] the information of the prediction for sample k.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from ..utils import pinvchol, pinvchol2, _HALF_LOG_2PI
from .core import kf_predict, info_to_state, reduce_model
from .info import info_smoother_stationary
from .opts import KalmanOpts, process_kalman_inputs, process_fast_flag
from .smoother import SmootherResult


def info_update2(CtRinvC, CtRinvY, old_i, old_I):
    """Pure information-form measurement update (MATLAB: infoUpdate2).

    Returns (new_i, new_I) = (old_i + C'inv(R)y, old_I + C'inv(R)C).
    """
    return old_i + CtRinvY, old_I + CtRinvC


def _half_logpdet(M):
    """0.5 * log of the pseudo-determinant of a PSD matrix (product of the
    strictly positive eigenvalues)."""
    w = np.linalg.eigvalsh((M + M.T) / 2)
    w = w[w > 1e-12 * max(1.0, np.abs(w).max(initial=0.0))]
    return 0.5 * np.log(w).sum() if w.size else 0.0


def _info_update_v1(CtRinvC, CtRinvY, x, P, logdet_crc, inv_crc,
                    want_logl=True):
    """Information-form measurement update (port of infoUpdate.m).

    Returns (new_i, new_I, new_x, new_P, logL, old_I). MATLAB computes the
    log-determinants from the diagonals of (permuted-triangular) LDL factors;
    the Python pinvchol/pinvchol2 factors are eigendecomposition-based and not
    triangular, so the (pseudo-)log-determinants are evaluated directly from
    eigenvalues here."""
    chol_old_I, _, old_I = pinvchol2(P)
    old_I = chol_old_I @ chol_old_I.T
    new_I = old_I + CtRinvC
    new_i = old_I @ x + CtRinvY

    _, _, new_P = pinvchol2(new_I)
    new_x = new_P @ new_i
    logL = None
    if want_logl:
        # logdet of the innovation covariance of the reduced model:
        # logdet(newI) - logdet(oldI) + logdet(CtRinvC)
        logdetS = 2 * (_half_logpdet(new_I) - _half_logpdet(old_I)) + logdet_crc
        z = CtRinvY - CtRinvC @ x
        invS = inv_crc - new_P
        z2 = z @ (invS @ z)
        logL = -0.5 * z2 - 0.5 * logdetS - z.shape[0] * _HALF_LOG_2PI
    return new_i, new_I, new_x, new_P, logL, old_I


class InfoFilterV1Result(NamedTuple):
    X: np.ndarray         # (nx, N) filtered state estimates
    P: np.ndarray         # (N, nx, nx) filtered covariances
    Xp: np.ndarray        # (nx, N+1) one-step-ahead predictions
    Pp: np.ndarray        # (N+1, nx, nx) prediction covariances
    rejected: np.ndarray  # (N,) outlier-rejected samples (always False)
    logL: float           # total log-likelihood of the data
    Ip: np.ndarray        # (N, nx, nx) information of the prediction at k
    I: np.ndarray         # (N, nx, nx) information of the update at k


def info_filter_stationary_v1(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None,
                              U=None,
                              opts: KalmanOpts | None = None) -> InfoFilterV1Result:
    """Hybrid information/covariance stationary Kalman filter
    (MATLAB: statInfoFilter; deprecated upstream in favor of statInfoFilter2,
    ported as info_filter_stationary).

    Uses the information form for the measurement update and the covariance
    form for the prediction. Improper priors are NOT properly supported (the
    covariance-form prediction cannot propagate infinite variances); a
    warning is issued as in MATLAB.
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
    if opts.outlier_flag:
        warnings.warn("Sample rejection not implemented for information filter")
    if M < N:
        warnings.warn("Fast mode not implemented in information filter")

    Xp = np.full((nx, N + 1), np.nan)
    X = np.full((nx, N), np.nan)
    Pp = np.full((N + 1, nx, nx), np.nan)
    P = np.full((N, nx, nx), np.nan)
    Ip = np.full((N, nx, nx), np.nan)
    I = np.full((N, nx, nx), np.nan)
    rejected = np.zeros(N, dtype=bool)
    logL = np.full(N, np.nan)

    prev_x = x0.copy()
    prev_P = P0.copy()
    Xp[:, 0] = x0
    Pp[0] = P0

    Y_D = Y - D @ U
    BU = B @ U

    # Precompute reduced model (always, as in MATLAB):
    CtRinvC, _, CtRinvY, _, logl_margin, _ = reduce_model(C, R, Y_D)
    _, _, inv_crc = pinvchol(CtRinvC)
    logdet_crc = 2 * _half_logpdet(CtRinvC)

    do_logl = not opts.no_logl
    for i in range(N):
        y = CtRinvY[:, i]
        if not np.isnan(y).any():
            _, this_I, prev_x, prev_P, ll, prior_I = _info_update_v1(
                CtRinvC, y, prev_x, prev_P, logdet_crc, inv_crc,
                want_logl=do_logl)
            if do_logl:
                logL[i] = ll
        else:  # missing sample: no update; bookkeeping only
            chol_prior_I, _, _ = pinvchol2(prev_P)
            prior_I = chol_prior_I @ chol_prior_I.T
            this_I = prior_I
        X[:, i] = prev_x
        P[i] = prev_P
        Ip[i] = prior_I
        I[i] = this_I

        if i == 0 and np.isinf(np.diag(prev_P)).any():
            warnings.warn("Infinite covariance matrix at predict step, no "
                          "promises this will be handled well. Try using "
                          "info_filter_stationary or large, but finite, "
                          "variances.")
        prev_x, prev_P = kf_predict(A, Q, prev_x, prev_P, BU[:, i])
        Xp[:, i + 1] = prev_x
        Pp[i + 1] = prev_P

    if do_logl:
        total = float(np.nansum(logL + logl_margin))
    else:
        total = float("nan")
    return InfoFilterV1Result(X, P, Xp, Pp, rejected, total, Ip, I)


def info_smoother_stationary_v1(Y, A, C, Q, R, x0=None, P0=None, B=None,
                                D=None, U=None,
                                opts: KalmanOpts | None = None) -> SmootherResult:
    """Stationary information smoother with state-space outputs
    (MATLAB: statInfoSmoother).

    Thin wrapper around info_smoother_stationary (statInfoSmoother2) that
    additionally converts the filtered and predicted information pairs back
    to state estimates/covariances. As in MATLAB, Pt is not computed (None)
    and logL/rejected are NaN/None placeholders.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ny, N = Y.shape
    nx = A.shape[0]

    res = info_smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U, opts,
                                   want_states=True)
    Xf = np.zeros((nx, N))
    Pf = np.zeros((N, nx, nx))
    for i in range(N):
        Xf[:, i], Pf[i] = info_to_state(res.iif[:, i], res.If[i])
    Xp = np.zeros((nx, N + 1))
    Pp = np.zeros((N + 1, nx, nx))
    for i in range(N + 1):
        Xp[:, i], Pp[i] = info_to_state(res.ip[:, i], res.Ip[i])

    return SmootherResult(res.Xs, res.Ps, None, Xf, Pf, Xp, Pp, None,
                          float("nan"))

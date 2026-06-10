"""Square-root form Kalman filter/smoother.

Ports of statSqrtFilter, statSqrtSmoother, sqrtPredictUpdate, sqrtRTS.

Covariance square roots are kept as square upper factors F with F.T @ F == P
throughout (QR-based propagation, following Park & Kailath 1995). Unlike the
MATLAB sources, which return the Cholesky factors in the P/Ps outputs, the
public functions here return full covariance matrices so that the results are
drop-in compatible with filter_stationary / smoother_stationary.

Notes on deviations from MATLAB (see docstrings for details):
- statSqrtFilter seeds the recursion with chol(P)', chol(R)', chol(Q)' (lower
  factors), while the QR arrays require upper factors with F.T@F == X. This is
  only correct for diagonal matrices; here utils.cholcov2 (upper factors) is
  used, which is correct for any PSD matrix.
- statSqrtFilter produces NaN states when the first processed sample is
  missing; here the update is skipped (consistent with statKalmanFilter).
- statSqrtFilter leaves logL = NaN for steady-state (fast) samples; here the
  log-likelihood of those samples is computed as in filter_stationary.
- sqrtRTS's improper-prior branch references undefined variables in MATLAB
  (it cannot run); here it falls back to the covariance-form backward step,
  which handles infinite variances.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
from scipy.stats import chi2

from ..utils import cholcov2, logl_normal, _HALF_LOG_2PI
from .core import reduce_model, info_to_state, state_to_info
from .filter import FilterResult
from .smoother import SmootherResult, _back_step_rts
from .opts import KalmanOpts, process_kalman_inputs, process_fast_flag


def _qr_r(M):
    """R factor of a thin QR decomposition, with rows sign-normalized so the
    diagonal is non-negative (R.T @ R is invariant under row sign flips)."""
    R = np.linalg.qr(M, mode="r")
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return s[:, None] * R


def sqrt_predict_update(y, prev_x, cP, cR, cQ, A, C, b, rej_threshold=None):
    """Combined predict+update step of the square-root Kalman filter via QR
    (MATLAB: sqrtPredictUpdate; following Park & Kailath 1995).

    cP, cR, cQ are square factors with F.T @ F == P (resp. R, Q) for the
    *previous filtered* covariance P. The step first predicts through (A, Q, b)
    and then updates with observation y.

    Returns (new_x, new_cP, K, logL, rejected, cS) where new_cP is the factor
    of the new filtered covariance, K the Kalman gain, and cS the upper factor
    of the innovation covariance (cS.T @ cS == S).
    """
    ny = cR.shape[0]
    nx = cQ.shape[0]
    Mt = np.block([
        [cR, np.zeros((ny, nx))],
        [cP @ A.T @ C.T, cP @ A.T],
        [cQ @ C.T, cQ],
    ])
    R = _qr_r(Mt)
    cS = R[:ny, :ny]
    H = R[:ny, ny:]
    new_cP = R[ny:, ny:]

    x_pred = A @ prev_x + b
    K = scipy.linalg.solve_triangular(cS, H, lower=False).T
    innov = y - C @ x_pred
    iLy = scipy.linalg.solve_triangular(cS, innov, lower=False, trans="T")
    z2 = iLy @ iLy
    logL = (-0.5 * z2 - np.sum(np.log(np.abs(np.diag(cS))))
            - ny * _HALF_LOG_2PI)
    if rej_threshold is not None and z2 > rej_threshold:
        # Sample rejected: predict only.
        new_cP = _qr_r(np.vstack([cP @ A.T, cQ]))
        return x_pred, new_cP, K, logL, True, cS
    new_x = x_pred + K @ innov
    return new_x, new_cP, K, logL, False, cS


def _sqrt_predict_only(prev_x, cP, cQ, A, b):
    """Square-root prediction step (no measurement)."""
    return A @ prev_x + b, _qr_r(np.vstack([cP @ A.T, cQ]))


def _sqrt_filter(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None, U=None,
                 opts: KalmanOpts | None = None):
    """Square-root filter core. Returns (FilterResult, cPf) where cPf is the
    (N, nx, nx) stack of filtered-covariance factors (cPf[k].T@cPf[k]==P[k])."""
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
    cPf = np.full((N, nx, nx), np.nan)
    rejected = np.zeros(N, dtype=bool)

    prev_x = x0.copy()
    prev_P = P0.copy()
    Xp[:, 0] = x0
    Pp[0] = P0

    Y_D = Y - D @ U
    BU = B @ U

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
    else:
        logl_margin = 0.0

    # Improper (infinite-variance) prior: information-filter pass until all
    # uncertainties are finite (same approach as filter_stationary):
    first_ind = 0
    if np.isinf(np.diag(prev_P)).any():
        from .info import true_info_filter
        good = ~np.isnan(Y_D).any(axis=0)
        hit = np.nonzero(np.cumsum(good) >= nx)[0]
        n_samp = (hit[0] + 1) if hit.size else N
        CtRinvC2, _, CtRinvY2, _, _, _ = reduce_model(C, R, Y_D[:, :n_samp])
        i0, I0 = state_to_info(prev_x, prev_P)
        ii, I, ip, Ip = true_info_filter(CtRinvY2, CtRinvC2, A, Q,
                                         BU[:, :n_samp], i0, I0)
        for k in range(n_samp):
            logL[k] = -np.inf
            X[:, k], P[k] = info_to_state(ii[:, k], I[k])
            cPf[k] = cholcov2(P[k])
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

    cQt = cholcov2(Q)
    cRt = cholcov2(R)
    cPt = cholcov2(prev_P)
    eye_nx = np.eye(nx)
    zeros_nx = np.zeros((nx, nx))

    do_logl = not opts.no_logl
    K = None
    cS = None
    for i in range(first_ind, M):
        y = Y_D[:, i]
        if i == first_ind:
            # First processed step: update only (predict with A=I, Q=0, b=0,
            # since prev_x/cPt already are the prediction for this sample).
            At, cQi, b = eye_nx, zeros_nx, np.zeros(nx)
        else:
            At, cQi, b = A, cQt, BU[:, i - 1]
        if not np.isnan(y).any():
            prev_x, cPt, K, ll, rej, cS = sqrt_predict_update(
                y, prev_x, cPt, cRt, cQi, At, C, b, rej_threshold)
            rejected[i] = rej
            if do_logl:
                logL[i] = ll
        else:  # missing sample: predict only
            prev_x, cPt = _sqrt_predict_only(prev_x, cPt, cQi, At, b)
        X[:, i] = prev_x
        cPf[i] = cPt
        P[i] = cPt.T @ cPt
        Xp[:, i + 1] = A @ prev_x + BU[:, i]
        Pp[i + 1] = A @ P[i] @ A.T + Q

    if M < N:  # fast (steady-state) filtering for remaining steps
        if opts.outlier_flag:
            raise ValueError("KFfilter:outlierRejectFast - outlier rejection "
                             "is incompatible with fast mode.")
        prev_x = X[:, M - 1].copy()
        P_steady = P[M - 1]
        cP_steady = cPf[M - 1]
        K_steady = K
        G_steady = np.eye(nx) - K_steady @ C
        GBU_KY = G_steady @ BU[:, M - 1:N - 1] + K_steady @ Y_D[:, M:N]
        GA = G_steady @ A
        P[M:] = P_steady
        cPf[M:] = cP_steady
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
            # NOTE: MATLAB statSqrtFilter leaves these logL entries as NaN;
            # computed here for consistency with filter_stationary.
            iL = scipy.linalg.solve_triangular(cS, np.eye(ny), lower=False)
            innov = Y_D - C @ Xp[:, :-1]
            ll, _ = logl_normal(innov[:, M:], chol_inv_sigma=iL.T)
            logL[M:] = ll

    if do_logl:
        aux = logL + logl_margin
        total = np.nansum(aux[first_ind:])
    else:
        total = np.nan
    return FilterResult(X, P, Xp, Pp, rejected, float(total)), cPf


def sqrt_filter_stationary(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None,
                           U=None, opts: KalmanOpts | None = None) -> FilterResult:
    """Stationary Kalman filter in square-root form (MATLAB: statSqrtFilter).

    Same model, signature and return convention as filter_stationary; the
    covariance recursion is propagated through QR factor updates, which is
    numerically more robust. Note: unlike MATLAB (which returns Cholesky
    factors in P), full covariances are returned.
    """
    res, _ = _sqrt_filter(Y, A, C, Q, R, x0, P0, B, D, U, opts)
    return res


def _sqrt_rts(cpf, c_prev_ps, xp, xf, prev_xs, A, cQt, pf, pp, bu, iA):
    """Square-root Rauch-Tung-Striebel backward step (MATLAB: sqrtRTS).

    cpf: factor of the filtered covariance at this step; c_prev_ps: factor of
    the smoothed covariance at the next step. Returns
    (new_cPs, new_xs, new_Pt, Ht) with Ht = H.T (the transposed smoother gain).
    """
    nx = cQt.shape[0]
    if np.isinf(cpf).any() or not np.isfinite(pp).all():
        # Improper prior leaked into the filtered covariances. The MATLAB
        # branch for this case is broken (references undefined variables);
        # fall back to the covariance-form backward step, which handles
        # infinite variances:
        ps = c_prev_ps.T @ c_prev_ps
        new_ps, new_xs, new_pt, Hcov = _back_step_rts(
            pp, pf, ps, xp, xf, prev_xs, A, cQt, bu, iA)
        return cholcov2(new_ps), new_xs, new_pt, Hcov.T
    AcPf = cpf @ A.T
    G = AcPf.T @ AcPf + cQt.T @ cQt  # = Pp
    rhs = AcPf.T @ cpf
    try:
        Ht = np.linalg.solve(G, rhs)  # = H.T
    except np.linalg.LinAlgError:
        Ht = np.linalg.pinv(G) @ rhs
    M = np.block([
        [cQt, np.zeros((nx, nx))],
        [AcPf, cpf],
        [np.zeros((nx, nx)), c_prev_ps @ Ht],
    ])
    Rr = _qr_r(M)
    new_cPs = Rr[nx:, nx:]
    new_pt = (c_prev_ps.T @ c_prev_ps) @ Ht  # cov(x[k+1], x[k]) = Ps' H.T
    new_xs = xf + Ht.T @ (prev_xs - xp)
    return new_cPs, new_xs, new_pt, Ht


def sqrt_smoother_stationary(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None,
                             U=None, opts: KalmanOpts | None = None) -> SmootherResult:
    """Stationary Kalman smoother in square-root form (MATLAB: statSqrtSmoother).

    Forward pass via the square-root filter, backward pass via the QR-based
    RTS recursion. Same return convention as smoother_stationary (full
    covariances, not factors).
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ny, N = Y.shape
    nx = A.shape[0]
    x0, P0, B, D, U, opts = process_kalman_inputs(nx, N, x0, P0, B, D, U, opts,
                                                  ny=ny)
    M = process_fast_flag(opts.fast_flag, A, N)
    BU = B @ U

    fres, cPf = _sqrt_filter(Y, A, C, Q, R, x0, P0, B, D, U,
                             opts.copy(fast_flag=M + 1))
    Xf, Pf, Xp, Pp, rejected, logL = fres

    Xs = np.full_like(Xf, np.nan)
    Ps = np.full_like(Pf, np.nan)
    Pt = np.full((N - 1, nx, nx), np.nan)
    prev_xs = Xf[:, N - 1].copy()
    c_prev_ps = cPf[N - 1].copy()
    Xs[:, N - 1] = prev_xs
    Ps[N - 1] = Pf[N - 1]

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
    cQt = cholcov2(Q)

    Ht = None
    new_pt = None
    # Exact smoothing for the last M1 samples:
    for i in range(N - 2, N - M1 - 2, -1):
        if i < 0:
            break
        xf = Xf[:, i]
        bu = BU[:, i]
        xp = A @ xf + bu
        c_prev_ps, prev_xs, new_pt, Ht = _sqrt_rts(
            cPf[i], c_prev_ps, xp, xf, prev_xs, A, cQt, Pf[i], Pp[i + 1], bu, iA)
        Xs[:, i] = prev_xs
        Pt[i] = new_pt
        Ps[i] = c_prev_ps.T @ c_prev_ps

    if n_fast > 0:
        # Steady-state smoothing for the middle samples, reusing the gain:
        H = Ht.T
        if np.any(np.abs(np.linalg.eigvals(H)) > 1):
            warnings.warn("statKS:unstableSmooth - unstable smoothing, "
                          "skipping the backward pass.")
            H = np.zeros_like(H)
        aux = Xf - H @ (A @ Xf + BU)
        for i in range(N - M1 - 2, M2 - 1, -1):
            prev_xs = aux[:, i] + H @ prev_xs
            Xs[:, i] = prev_xs
        Ps[M2:N - M1 - 1] = c_prev_ps.T @ c_prev_ps
        Pt[M2:N - M1 - 1] = new_pt

    # Exact smoothing for the first M2 samples:
    for i in range(M2 - 1, -1, -1):
        xf = Xf[:, i]
        bu = BU[:, i]
        xp = A @ xf + bu
        c_prev_ps, prev_xs, new_pt, Ht = _sqrt_rts(
            cPf[i], c_prev_ps, xp, xf, prev_xs, A, cQt, Pf[i], Pp[i + 1], bu, iA)
        Xs[:, i] = prev_xs
        Pt[i] = new_pt
        Ps[i] = c_prev_ps.T @ c_prev_ps

    return SmootherResult(Xs, Ps, Pt, Xf, Pf, Xp, Pp, rejected, logL)

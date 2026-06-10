"""Constrained Kalman filter/smoother.

Ports of statKalmanFilterConstrained, statKalmanSmootherConstrained,
constrain/filterStationary_wConstraint and constrain/circleConstraint.

Two constraint mechanisms are provided, mirroring the MATLAB sources:

- filter_stationary_constrained / smoother_stationary_constrained: hard
  (equality) constraints. After each measurement update, the state is
  projected onto the linearized constraint surface H x = e and the covariance
  is projected accordingly (plus a 1e-12*I conditioning term, as in MATLAB).
  `constr_fun(x)` must return (H, e) (a third output, if present, is ignored).

- filter_stationary_w_constraint: soft constraints. The constraint is treated
  as an extra pseudo-observation H x = e + s, s ~ N(0, S), applied as an
  independent Kalman update before the regular measurement update.
  `constr_fun(x)` must return (H, e, S).

Notes on deviations from MATLAB:
- statKalmanFilterConstrained calls KFupdate with 4 arguments, matching the
  legacy KFupdateEff/KFupdateAlt signature (CtRinvY, CtRinvC, x, P) rather
  than the current 5-argument KFupdate (it cannot run as committed). The
  intended reduced-model update (C = R = C'inv(R)C, y = C'inv(R)y) is used
  here via kf_update.
- filterStationary_wConstraint references helper functions (predict, updateKF,
  update_wOutlierRejection) that no longer exist in matlab-linsys; the
  equivalent kf_predict/kf_update steps are used here. Its outlier-rejection
  hook is not implemented (it was not functional in MATLAB either).
- These routines do not support improper (infinite) priors; the MATLAB
  default P0 = 1e8*I is kept.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..utils import cholcov, cholcov2
from .core import kf_update, kf_predict
from .filter import FilterResult
from .smoother import SmootherResult
from .opts import process_kalman_inputs


def circle_constraint(x):
    """Linearized unit-circle constraint |x| = 1 (MATLAB: circleConstraint).

    Returns (H, e, S) such that H x = e + s with s ~ N(0, S) approximates the
    constraint in a neighborhood of x.
    """
    x = np.asarray(x, dtype=float).ravel()
    xn = x / np.linalg.norm(x)
    H = xn[None, :]
    e = np.array([1.0])
    S = np.array([[0.01]])
    return H, e, S


def _process_constrained_inputs(Y, A, C, Q, R, x0, P0, B, D, U):
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    ny, N = Y.shape
    nx = A.shape[0]
    if P0 is None:
        P0 = 1e8 * np.eye(nx)  # MATLAB default (no improper-prior support)
    x0, P0, B, D, U, _ = process_kalman_inputs(nx, N, x0, P0, B, D, U, None,
                                               ny=ny)
    return Y, A, C, Q, R, x0, P0, B, D, U, ny, nx, N


def filter_stationary_constrained(Y, A, C, Q, R, x0=None, P0=None, B=None,
                                  D=None, U=None, outlier_rejection=False,
                                  constr_fun=None) -> FilterResult:
    """Stationary Kalman filter with hard state constraints
    (MATLAB: statKalmanFilterConstrained).

    constr_fun(x) -> (H, e[, S]) gives the linearized constraint H x = e
    around x (S, if returned, is ignored: the constraint is enforced exactly
    by projection). With constr_fun=None the result equals the unconstrained
    filter. Returns a FilterResult with logL = NaN (not computed, as in
    MATLAB). Default prior is P0 = 1e8*I (improper priors unsupported).
    """
    (Y, A, C, Q, R, x0, P0, B, D, U,
     ny, nx, N) = _process_constrained_inputs(Y, A, C, Q, R, x0, P0, B, D, U)

    Xp = np.full((nx, N + 1), np.nan)
    X = np.full((nx, N), np.nan)
    Pp = np.full((N + 1, nx, nx), np.nan)
    P = np.full((N, nx, nx), np.nan)
    rejected = np.zeros(N, dtype=bool)

    tol = 1e-8
    prev_x = x0.copy()
    prev_P = P0.copy()
    Xp[:, 0] = x0
    Pp[0] = P0

    # Pre-compute reduced-model quantities:
    CtRinv = C.T @ np.linalg.pinv(R, rcond=tol)
    CtRinvC = CtRinv @ C
    Y_D = Y - D @ U
    CtRinvY = CtRinv @ Y_D
    BU = B @ U

    if outlier_rejection:
        warnings.warn("Outlier rejection not implemented")

    eye_nx = np.eye(nx)
    for i in range(N):
        ciry = CtRinvY[:, i]
        if not np.isnan(ciry).any():  # NaN measurement: skip update
            prev_x, prev_P = kf_update(CtRinvC, CtRinvC, ciry,
                                       prev_x, prev_P)[:2]
        if constr_fun is not None:
            # Enforce the linearized constraint H x = e by projection:
            out = constr_fun(prev_x)
            H = np.atleast_2d(np.asarray(out[0], dtype=float))
            e = np.atleast_1d(np.asarray(out[1], dtype=float))
            pinvH = np.linalg.pinv(H)
            prev_x = prev_x - pinvH @ (H @ prev_x - e)
            cP = cholcov2(prev_P)
            aux = (eye_nx - pinvH @ H) @ cP.T
            prev_P = aux @ aux.T + 1e-12 * eye_nx  # numerical conditioning
        X[:, i] = prev_x
        P[i] = prev_P
        prev_x, prev_P = kf_predict(A, Q, prev_x, prev_P, BU[:, i])
        Xp[:, i + 1] = prev_x
        Pp[i + 1] = prev_P

    return FilterResult(X, P, Xp, Pp, rejected, float("nan"))


def smoother_stationary_constrained(Y, A, C, Q, R, x0=None, P0=None, B=None,
                                    D=None, U=None, outlier_rejection=False,
                                    constr_fun=None) -> SmootherResult:
    """Stationary Kalman smoother with hard state constraints
    (MATLAB: statKalmanSmootherConstrained).

    Forward pass via filter_stationary_constrained, backward pass via the
    standard RTS recursion (the constraint is NOT re-enforced backwards, as
    in MATLAB). Returns a SmootherResult with logL = NaN.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    N = Y.shape[1]
    nx = A.shape[0]

    fres = filter_stationary_constrained(Y, A, C, Q, R, x0, P0, B, D, U,
                                         outlier_rejection, constr_fun)
    Xf, Pf, Xp, Pp, rejected, _ = fres

    Xs = Xf.copy()
    Ps = Pf.copy()
    Pt = np.full((N - 1, nx, nx), np.nan)
    prev_xs = Xf[:, N - 1].copy()
    prev_ps = Pf[N - 1].copy()

    for i in range(N - 2, -1, -1):
        xf, pf = Xf[:, i], Pf[i]
        xp, pp = Xp[:, i + 1], Pp[i + 1]

        AP = A @ pf
        # newK = (A pf)' / pp; pp symmetric so solve directly:
        newK = np.linalg.solve(pp, AP).T
        new_pt = prev_ps @ newK.T  # cov(x[k+1], x[k])
        sPs, _ = cholcov(prev_ps)
        Kps = newK @ sPs.T
        sPr, _ = cholcov(newK @ AP)  # = H Pp H'
        new_ps = Kps @ Kps.T + pf - sPr.T @ sPr
        prev_xs = xf + newK @ (prev_xs - xp)

        Xs[:, i] = prev_xs
        Pt[i] = new_pt
        prev_ps = new_ps
        Ps[i] = prev_ps

    return SmootherResult(Xs, Ps, Pt, Xf, Pf, Xp, Pp, rejected, float("nan"))


def filter_stationary_w_constraint(Y, A, C, Q, R, x0=None, P0=None, B=None,
                                   D=None, U=None, constr_fun=None) -> FilterResult:
    """Stationary Kalman filter with soft (stochastic) state constraints
    (MATLAB: constrain/filterStationary_wConstraint).

    constr_fun(xp) -> (H, e, S) defines the pseudo-observation H x = e + s,
    s ~ N(0, S), evaluated at the predicted state and applied as an extra
    Kalman update before the regular measurement update.

    NOTE: this variant uses the MATLAB predict-first convention: Xp/Pp have N
    entries and Xp[:, k] is the one-step-ahead prediction of x[k] (x0, P0 act
    as the estimate at time -1). logL is not computed (NaN).
    """
    (Y, A, C, Q, R, x0, P0, B, D, U,
     ny, nx, N) = _process_constrained_inputs(Y, A, C, Q, R, x0, P0, B, D, U)

    Xp = np.full((nx, N), np.nan)
    X = np.full((nx, N), np.nan)
    Pp = np.full((N, nx, nx), np.nan)
    P = np.full((N, nx, nx), np.nan)
    rejected = np.zeros(N, dtype=bool)

    prev_x = x0.copy()
    prev_P = P0.copy()
    BU = B @ U
    Y_D = Y - D @ U

    for i in range(N):
        prev_x, prev_P = kf_predict(A, Q, prev_x, prev_P, BU[:, i])
        Xp[:, i] = prev_x
        Pp[i] = prev_P

        if constr_fun is not None:
            H, e, S = constr_fun(prev_x)
            H = np.atleast_2d(np.asarray(H, dtype=float))
            if H.shape[0]:
                e = np.atleast_1d(np.asarray(e, dtype=float))
                S = np.atleast_2d(np.asarray(S, dtype=float))
                prev_x, prev_P = kf_update(H, S, e, prev_x, prev_P)[:2]

        y = Y_D[:, i]
        if not np.isnan(y).any():
            prev_x, prev_P = kf_update(C, R, y, prev_x, prev_P)[:2]
        X[:, i] = prev_x
        P[i] = prev_P

    return FilterResult(X, P, Xp, Pp, rejected, float("nan"))

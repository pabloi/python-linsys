"""Information-form Kalman filter and smoother.

Ports of trueStatInfoFilter, statInfoFilter2, statInfoSmoother2.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import scipy.linalg

from ..utils import cholcov, pinvchol, pinvchol2
from .core import info_to_state, state_to_info, reduce_model
from .opts import KalmanOpts, process_kalman_inputs, process_fast_flag


class InfoFilterResult(NamedTuple):
    ii: np.ndarray   # (nx, N) updated information states
    I: np.ndarray    # (N, nx, nx) updated information matrices
    ip: np.ndarray   # (nx, N+1) predicted information states
    Ip: np.ndarray   # (N+1, nx, nx) predicted information matrices
    X: np.ndarray | None = None    # (nx, N) state estimates (if requested)
    P: np.ndarray | None = None    # (N, nx, nx) state covariances


def true_info_filter(Y, CtRinvC, A, Q, BU, i0, I0, slow_samples=None):
    """Information filter core (MATLAB: trueStatInfoFilter).

    Y here is the reduced observation C' inv(R) y[k] (columns). Returns
    (ii, I, ip, Ip)."""
    nx = A.shape[0]
    N = Y.shape[1]
    if slow_samples is None:
        slow_samples = N
    prev_i = np.asarray(i0, dtype=float).ravel().copy()
    prev_I = np.atleast_2d(I0).copy()

    ey = np.eye(nx)
    ciQ, _, iQ = pinvchol2(Q)
    invertible_Q = np.linalg.norm(Q @ iQ - ey) < 1e-9
    iA = np.linalg.pinv(A)
    invertible_A = np.linalg.norm(A @ iA - ey) < 1e-9
    iQA = iQ @ A
    ciQA = ciQ.T @ A
    AtiQA = ciQA.T @ ciQA

    ip = np.full((nx, N + 1), np.nan)
    ii = np.full((nx, N), np.nan)
    Ip = np.full((N + 1, nx, nx), np.nan)
    I = np.full((N, nx, nx), np.nan)
    ip[:, 0] = prev_i
    Ip[0] = prev_I

    for k in range(slow_samples):
        y = Y[:, k]
        if not np.isnan(y).any():  # update (trivial in information form)
            prev_I = prev_I + CtRinvC
            prev_i = prev_i + y
        ii[:, k] = prev_i
        I[k] = prev_I

        # Predict:
        bu = BU[:, k]
        if invertible_Q:
            if invertible_A:
                cI, _ = cholcov(prev_I)
                iAcI = iA.T @ cI.T
                cP = scipy.linalg.cholesky(iQ + iAcI @ iAcI.T, lower=False)
                iAicP = iA @ np.linalg.inv(cP)
                b = cI @ iAicP
                iAcIb = iAcI @ b
                prev_I = iAcI @ iAcI.T - iAcIb @ iAcIb.T
                prev_i = iQA @ (iAicP @ iAicP.T) @ prev_i + prev_I @ bu
            else:
                chol_auxP, _, auxP = pinvchol(prev_I + AtiQA)
                HH = ciQA @ chol_auxP
                prev_I = ciQ @ (ey - HH @ HH.T) @ ciQ.T
                prev_i = iQA @ auxP @ prev_i + prev_I @ bu
        elif np.all(Q == 0):
            if invertible_A:
                cI, _ = cholcov(prev_I)
                iAcI = iA.T @ cI.T
                prev_I = iAcI @ iAcI.T
                prev_i = iA.T @ prev_i + prev_I @ bu
            else:
                x, P = info_to_state(prev_i, prev_I)
                prev_i, prev_I = state_to_info(A @ x + bu, A @ P @ A.T + Q)
        else:
            # Q is PSD but singular: fall back to state-space prediction, then
            # regularize Q for invertibility on subsequent steps.
            x, P = info_to_state(prev_i, prev_I)
            prev_i, prev_I = state_to_info(A @ x + bu, A @ P @ A.T + Q)
            warnings.warn("trueStatInfoFilter:nonInvQ - Q not invertible with "
                          "improper priors; regularizing Q for invertibility.")
            Q = Q + 1e-7 * np.eye(nx)
            invertible_Q = True
            ciQ, _, iQ = pinvchol(Q)
            iQA = iQ @ A
            ciQA = ciQ.T @ A
            AtiQA = ciQA.T @ ciQA
        ip[:, k + 1] = prev_i
        Ip[k + 1] = prev_I

    if slow_samples < N:  # fast (steady-state) update
        old_I = prev_I + CtRinvC
        _, old_P = info_to_state(prev_i, old_I)
        Ip[slow_samples + 1:N + 1] = prev_I
        I[slow_samples:N] = old_I
        for k in range(slow_samples, N):
            y = Y[:, k]
            if not np.isnan(y).any():
                prev_i = prev_i + y
            ii[:, k] = prev_i
            old_x = old_P @ prev_i
            prev_x = A @ old_x + BU[:, k]
            prev_i = prev_I @ prev_x
            ip[:, k + 1] = prev_i

    return ii, I, ip, Ip


def info_filter_stationary(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None,
                           U=None, opts: KalmanOpts | None = None,
                           want_states=True) -> InfoFilterResult:
    """Stationary information filter (MATLAB: statInfoFilter2). Same model and
    signature as filter_stationary; returns information-form estimates."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ny, N = Y.shape
    nx = A.shape[0]
    x0, P0, B, D, U, opts = process_kalman_inputs(nx, N, x0, P0, B, D, U, opts,
                                                  ny=ny)
    M = process_fast_flag(opts.fast_flag, A, N)
    if opts.outlier_flag:
        warnings.warn("Sample rejection not implemented for information filter")
    if M < N:
        warnings.warn("Fast mode not implemented in information filter")

    Y_D = Y - D @ U
    BU = B @ U
    CtRinvC, _, CtRinvY, _, _, _ = reduce_model(C, R, Y_D)
    i0, I0 = state_to_info(x0, P0)
    ii, I, ip, Ip = true_info_filter(CtRinvY, CtRinvC, A, Q, BU, i0, I0)
    X = P = None
    if want_states:
        X = np.full((nx, N), np.nan)
        P = np.full((N, nx, nx), np.nan)
        for k in range(N):
            X[:, k], P[k] = info_to_state(ii[:, k], I[k])
    return InfoFilterResult(ii, I, ip, Ip, X, P)


class InfoSmootherResult(NamedTuple):
    is_: np.ndarray  # (nx, N) smoothed information states
    Is: np.ndarray   # (N, nx, nx) smoothed information matrices
    iif: np.ndarray  # filtered (updated) information states
    If: np.ndarray
    ip: np.ndarray
    Ip: np.ndarray
    Xs: np.ndarray | None = None
    Ps: np.ndarray | None = None


def info_smoother_stationary(Y, A, C, Q, R, x0=None, P0=None, B=None, D=None,
                             U=None, opts: KalmanOpts | None = None,
                             want_states=True) -> InfoSmootherResult:
    """Stationary information smoother via independent forward and backward
    information-filter passes plus a merge (MATLAB: statInfoSmoother2)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    ny, N = Y.shape
    nx = A.shape[0]
    x0, P0, B, D, U, opts = process_kalman_inputs(nx, N, x0, P0, B, D, U, opts,
                                                  ny=ny)
    M = process_fast_flag(opts.fast_flag, A, N)

    Y_D = Y - D @ U
    BU = B @ U
    CtRinvC, _, CtRinvY, _, _, _ = reduce_model(C, R, Y_D)
    iA = np.linalg.pinv(A)
    cQ, _ = cholcov(Q)
    iAcQ = iA @ cQ.T
    i0, I0 = state_to_info(x0, P0)

    # Forward pass:
    iif, If, ip, Ip = true_info_filter(CtRinvY, CtRinvC, A, Q, BU, i0, I0, M)
    # Backward pass: the same filter run on time-reversed data with inverse
    # dynamics and an uninformative prior.
    BU_rev = -iA @ np.fliplr(np.concatenate(
        [np.zeros((BU.shape[0], 1)), BU[:, :-1]], axis=1))
    ifb, Ib, _, _ = true_info_filter(np.fliplr(CtRinvY), CtRinvC, iA,
                                     iAcQ @ iAcQ.T, BU_rev,
                                     np.zeros(nx), np.zeros((nx, nx)), M)
    # Merge: smoothed information is predicted-forward + filtered-backward
    Is = Ip[:N] + Ib[::-1]
    is_ = ip[:, :N] + np.fliplr(ifb)

    Xs = Ps = None
    if want_states:
        Xs = np.full((nx, N), np.nan)
        Ps = np.full((N, nx, nx), np.nan)
        for k in range(N):
            Xs[:, k], Ps[k] = info_to_state(is_[:, k], Is[k])
    return InfoSmootherResult(is_, Is, iif, If, ip, Ip, Xs, Ps)

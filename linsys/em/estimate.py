"""M-step parameter estimation for EM (port of matlab-linsys/EM/estimateParams.m).

See Cheng & Sabes 2006, Ghahramani & Hinton 1996, Shumway & Stoffer 1982.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..utils import cholcov, rob_cov
from .opts import EMOpts, process_em_opts


class ParamEstimate(NamedTuple):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x0: "np.ndarray | list[np.ndarray]"
    P0: "np.ndarray | list[np.ndarray]"


def _mrdivide(M, O):
    """MATLAB M/O (solve X @ O = M), with a least-squares fallback for
    singular/ill-conditioned O."""
    O = np.atleast_2d(np.asarray(O, dtype=float))
    M = np.atleast_2d(np.asarray(M, dtype=float))
    try:
        return np.linalg.solve(O.T, M.T).T
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(O.T, M.T, rcond=None)[0].T


class _Suff(NamedTuple):
    """Sufficient statistics (MATLAB: computeRelevantMatrices outputs)."""
    yx: np.ndarray   # sum y x'   (NaN samples removed)
    yu: np.ndarray   # sum y u'   (NaN samples removed)
    xx: np.ndarray   # sum x x'   (NaN samples removed)
    uu: np.ndarray   # sum u u'   (NaN samples removed)
    xu: np.ndarray   # sum x u'   (NaN samples removed)
    SP: np.ndarray   # sum P      (NaN samples removed)
    SPt: np.ndarray  # sum cov(x[k+1], x[k])
    xx_: np.ndarray  # sum_{k<N} x x'
    uu_: np.ndarray  # sum_{k<N} u u'
    xu_: np.ndarray  # sum_{k<N} x u'
    xx1: np.ndarray  # sum x[k+1] x[k]'
    xu1: np.ndarray  # sum x[k+1] u[k]'
    SP_: np.ndarray  # sum_{k<N} P[k]
    S_P: np.ndarray  # sum_{k>0} P[k]


def _relevant_matrices_single(Y, X, U, P, Pt) -> _Suff:
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    X = np.atleast_2d(np.asarray(X, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    P = np.asarray(P, dtype=float)
    Pt = np.asarray(Pt, dtype=float)

    # Sums for A, B estimation (states are defined for all samples, even when
    # the output is missing):
    xu_ = X[:, :-1] @ U[:, :-1].T
    uu_ = U[:, :-1] @ U[:, :-1].T
    xx_ = X[:, :-1] @ X[:, :-1].T
    SP_ = P[:-1].sum(axis=0)
    S_P = P[1:].sum(axis=0)
    SPt = Pt.sum(axis=0)
    xu1 = X[:, 1:] @ U[:, :-1].T
    xx1 = X[:, 1:] @ X[:, :-1].T

    # Remove samples with missing output data for C, D, R estimation:
    if np.isnan(Y).any():
        idx = ~np.isnan(Y).any(axis=0)
        Y = Y[:, idx]
        X = X[:, idx]
        U = U[:, idx]
        P = P[idx]
    SP = P.sum(axis=0)
    xu = X @ U.T
    uu = U @ U.T
    xx = X @ X.T
    yx = Y @ X.T
    yu = Y @ U.T
    return _Suff(yx, yu, xx, uu, xu, SP, SPt, xx_, uu_, xu_, xx1, xu1, SP_, S_P)


def _relevant_matrices(Y, X, U, P, Pt) -> _Suff:
    """Sufficient statistics; multiple realizations (lists) are summed."""
    if isinstance(X, list):
        parts = [_relevant_matrices_single(y, x, u, p, pt)
                 for y, x, u, p, pt in zip(Y, X, U, P, Pt)]
        return _Suff(*[sum(getattr(p, f) for p in parts)
                       for f in _Suff._fields])
    return _relevant_matrices_single(Y, X, U, P, Pt)


def _residuals(Y, U, X, A, B, C, D):
    """Expected state/output residuals (MATLAB: computeResiduals).

    Returns (w, z): w (nx, N-1) state-transition residuals, z (ny, Nz)
    output residuals at non-missing samples. Multiple realizations are
    concatenated as extra samples."""
    if isinstance(X, list):
        parts = [_residuals(y, u, x, A, B, C, D) for y, u, x in zip(Y, U, X)]
        w = np.concatenate([p[0] for p in parts], axis=1)
        z = np.concatenate([p[1] for p in parts], axis=1)
        return w, z
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    X = np.atleast_2d(np.asarray(X, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    idx = ~np.isnan(Y).any(axis=0)
    z = Y - C @ X - D @ U
    z = z[:, idx]
    w = X[:, 1:] - A @ X[:, :-1] - B @ U[:, :-1]
    return w, z


def _estimate_init(X, P, A, Q):
    """Initial-condition estimate (MATLAB: estimateInit).

    x0 is the smoothed estimate of the first state. For P0, MATLAB uses Q
    (the smoothed P[0] would decrease monotonically over EM iterations and
    converge to 0, so its sum with Q converges to Q)."""
    x0 = np.asarray(X)[:, 0].copy()
    P0 = Q.copy()
    return x0, P0


def estimate_params(Y, U, X, P, Pt, opts: EMOpts | None = None) -> ParamEstimate:
    """M-step of EM for an LTI state-space model (MATLAB: estimateParams).

    Y: (ny, N) output (NaN columns = missing); U: (nu, N) input;
    X: (nx, N) smoothed states; P: (N, nx, nx) smoothed state covariances;
    Pt: (N-1, nx, nx) smoothed transition covariances cov(x[k+1], x[k]).
    All of Y/U/X/P/Pt may instead be lists (multiple realizations with equal
    dimensions but possibly different N); then x0, P0 are lists too.

    opts must be a processed EMOpts (as inside em()); if None or unprocessed,
    defaults are filled here. Honors fix_a/fix_b/fix_c/fix_d/fix_q/fix_r/
    fix_x0/fix_p0, ind_b/ind_d, diag_a, stable_a, diag_r, spherical_r,
    min_q/min_r and robust_flag (robust Q estimation via utils.rob_cov).
    """
    is_list = isinstance(X, list)
    if opts is None or opts.Niter is None:  # unprocessed opts: fill defaults
        nu = (U[0] if is_list else np.atleast_2d(U)).shape[0]
        ny = (Y[0] if is_list else np.atleast_2d(Y)).shape[0]
        nx = (X[0] if is_list else np.atleast_2d(X)).shape[0]
        opts = process_em_opts(opts, nu, nx, ny)

    s = _relevant_matrices(Y, X, U, P, Pt)
    nx = s.xx.shape[0]
    ny = s.yx.shape[0]
    nu = s.uu.shape[0]

    # --- Estimate A, B ---
    ib = opts.ind_b
    xu_ = s.xu_[:, ib]
    xu1 = s.xu1[:, ib]
    uu_ = s.uu_[np.ix_(ib, ib)]
    B = np.zeros((nx, nu))
    if not opts.diag_a or opts.fix_a is not None:  # fixed and/or full A
        if opts.fix_a is None and opts.fix_b is None:
            O = np.block([[s.SP_ + s.xx_, xu_], [xu_.T, uu_]])
            AB = _mrdivide(np.hstack([s.SPt + s.xx1, xu1]), O)
            # In absence of state uncertainty reduces to [A,B] = X+/[X;U]
            A = AB[:, :nx]
            B[:, ib] = AB[:, nx:]
        elif opts.fix_a is None:  # only A estimated
            B = opts.fix_b.copy()
            A = _mrdivide(s.SPt + s.xx1 - B[:, ib] @ xu_.T, s.SP_ + s.xx_)
        elif opts.fix_b is None:  # only B estimated
            A = opts.fix_a.copy()
            B[:, ib] = _mrdivide(xu1 - A @ xu_, uu_)
        else:
            A = opts.fix_a.copy()
            B = opts.fix_b.copy()
    else:  # diagonal A enforced (heuristic, as in MATLAB)
        if opts.fix_b is None:
            O = np.block([[np.diag(np.diag(s.SP_ + s.xx_)), xu_],
                          [xu_.T, uu_]])
            AB = _mrdivide(np.hstack([np.diag(np.diag(s.SPt + s.xx1)), xu1]), O)
            A = AB[:, :nx]
            B[:, ib] = AB[:, nx:]
        else:
            B = opts.fix_b.copy()
            # NOTE: the MATLAB source returns a column vector here (likely a
            # bug); we build the intended diagonal matrix.
            a = np.diag(s.SPt + s.xx1 - B[:, ib] @ xu_.T) / np.diag(s.SP_ + s.xx_)
            A = np.diag(a)

    # Enforce stability if required:
    if opts.stable_a and opts.fix_a is None:
        if is_list:
            n_samp = max(np.atleast_2d(y).shape[1] for y in Y)
        else:
            n_samp = np.atleast_2d(Y).shape[1]
        th = 1 - 1 / (3 * n_samp)  # above this: practically unstable
        w, V = np.linalg.eig(A)
        idx = np.abs(w) > th
        if idx.any():
            w[idx] = th * w[idx] / np.abs(w[idx])
            A = np.real(V @ np.diag(w) @ np.linalg.inv(V))
            # Re-estimate B given the new A:
            if opts.fix_b is None:
                B[:, ib] = _mrdivide(xu1 - A @ xu_, uu_)
            if np.any(np.abs(np.linalg.eigvals(A)) > (1 - 1 / (4 * n_samp))):
                raise ValueError("estimate_params: A is still unstable")

    # --- Estimate C, D ---
    idd = opts.ind_d
    xu = s.xu[:, idd]
    uu = s.uu[np.ix_(idd, idd)]
    yu = s.yu[:, idd]
    D = np.zeros((ny, nu))
    if opts.fix_c is None and opts.fix_d is None:
        O = np.block([[s.SP + s.xx, xu], [xu.T, uu]])
        CD = _mrdivide(np.hstack([s.yx, yu]), O)
        # In absence of state uncertainty reduces to [C,D] = Y/[X;U]
        C = CD[:, :nx]
        D[:, idd] = CD[:, nx:]
    elif opts.fix_c is None:  # only C estimated
        D = opts.fix_d.copy()
        C = _mrdivide(s.yx - D[:, idd] @ xu.T, s.SP + s.xx)
    elif opts.fix_d is None:  # only D estimated
        C = opts.fix_c.copy()
        D[:, idd] = _mrdivide(yu - C @ xu, uu)
    else:
        C = opts.fix_c.copy()
        D = opts.fix_d.copy()

    # --- Estimate Q, R --- (Shumway & Stoffer 1982 adaptation)
    w, z = _residuals(Y, U, X, A, B, C, D)

    if opts.fix_q is None:
        aux, _ = cholcov(s.SP_)  # also enforces symmetry
        Aa = A @ aux.T
        Nw = w.shape[1]
        APt = A @ s.SPt.T
        Q2 = (s.S_P - (APt + APt.T) + Aa @ Aa.T) / Nw
        if not opts.robust_flag:
            Q1 = (w @ w.T) / Nw  # covariance of expected residuals
        else:
            # Robust estimation avoids overestimating Q due to 'outlier'
            # observations (breaks the non-decreasing logL guarantee).
            Q1 = rob_cov(w)
        Q = Q1 + Q2
        # Regularize: enforce minimum diagonal value
        q = np.diag(Q).copy()
        q[q < opts.min_q] = opts.min_q
        Q = Q.copy()
        np.fill_diagonal(Q, q)
        cQ, _ = cholcov(Q)
        Q = cQ.T @ cQ  # enforce PSD
    else:
        Q = opts.fix_q.copy()

    if opts.fix_r is None:
        aux, _ = cholcov(s.SP)  # enforce symmetry
        Ca = C @ aux.T
        Nz = z.shape[1]
        R1 = (z @ z.T) / Nz
        R2 = (Ca @ Ca.T) / Nz
        R = R1 + R2
        if opts.diag_r:
            R = np.diag(np.diag(R))
        if opts.spherical_r:
            R = np.eye(ny) * (np.trace(R) / ny)
        r = np.diag(R).copy()
        r[r < opts.min_r] = opts.min_r  # avoid ill-conditioning
        R = R.copy()
        np.fill_diagonal(R, r)
    else:
        R = opts.fix_r.copy()

    # --- Estimate x0, P0 ---
    if is_list:
        pairs = [_estimate_init(x, p, A, Q) for x, p in zip(X, P)]
        x0 = [p[0] for p in pairs]
        P0 = [p[1] for p in pairs]
        if opts.fix_x0 is not None:
            x0 = [opts.fix_x0.copy() for _ in x0]
        if opts.fix_p0 is not None:
            P0 = [opts.fix_p0.copy() for _ in P0]
    else:
        x0, P0 = _estimate_init(X, P, A, Q)
        if opts.fix_x0 is not None:
            x0 = opts.fix_x0.copy()
        if opts.fix_p0 is not None:
            P0 = opts.fix_p0.copy()

    return ParamEstimate(A, B, C, D, Q, R, x0, P0)

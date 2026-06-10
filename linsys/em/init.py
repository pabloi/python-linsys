"""Initial parameter guess for EM (port of matlab-linsys/EM/initEM.m)."""
from __future__ import annotations

import numpy as np

from ..utils import substitute_nans
from .estimate import ParamEstimate, estimate_params
from .opts import EMOpts, process_em_opts


def _init_guess(Y, U, nx, opts, rng) -> "np.ndarray | list[np.ndarray]":
    """Initial state guess from uncentered PCA of (Y - D U)
    (MATLAB: initEM>initGuessOld)."""
    if isinstance(Y, list):
        Xc = _init_guess(np.concatenate(Y, axis=1), np.concatenate(U, axis=1),
                         nx, opts, rng)
        splits = np.cumsum([np.atleast_2d(y).shape[1] for y in Y])[:-1]
        return [x.copy() for x in np.split(Xc, splits, axis=1)]
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    ny, N = Y.shape
    idx = ~np.isnan(Y).any(axis=0)
    if opts.fix_d is not None:  # D provided, no need to estimate
        D = opts.fix_d
    elif U.shape[0] == 0:
        D = np.zeros((ny, 0))
    else:  # least squares: D = Y/U
        D = np.linalg.lstsq(U[:, idx].T, Y[:, idx].T, rcond=None)[0].T
    M = Y[:, idx] - D @ U[:, idx]
    # Uncentered PCA: principal directions = right singular vectors of M
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    X = 1e-5 * rng.standard_normal((nx, N))
    if nx <= ny:
        X[:, idx] = vt[:nx, :]
    else:  # more states than output dims: fill what we can
        X[:ny, idx] = vt
    if not idx.all():  # interpolate states at missing samples
        t = np.arange(N)
        for i in range(nx):
            X[i, ~idx] = np.interp(t[~idx], t[idx], X[i, idx])
    # Fix scaling (WLOG): each state has norm 100
    X = 1e2 * X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    return X


def _init_cov(X, U, P=None, Pt=None):
    """Plausible initial state covariances (MATLAB: initEM>initCov).

    Returns (P, Pt) with shapes (N, nx, nx) and (N-1, nx, nx). (MATLAB tiles
    Pt to N pages; we use N-1 for consistency with the smoother output --
    only their sum is ever used, so this is a negligible difference in an
    initialization heuristic.)"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    N = X.shape[1]
    if P is None:
        dX = np.diff(X, axis=1).T  # (N-1, nx)
        if U.size and not np.all(U == 0):
            Ue = U[:, :N - 1].T  # (N-1, nu)
            # Projection orthogonal to the input:
            dX = dX - Ue @ np.linalg.lstsq(Ue, dX, rcond=None)[0]
        Px = 0.1 * (dX.T @ dX) / N
        P = np.tile(Px, (N, 1, 1))
        Pt = np.tile(0.2 * np.diag(np.diag(Px)), (N - 1, 1, 1))
    elif Pt is None:
        P = np.asarray(P, dtype=float)
        Pt = 0.2 * P[:N - 1]
    return P, Pt


def init_em(Y, U, x_guess, opts: EMOpts | None = None, P_guess=None,
            Pt_guess=None, rng=None) -> ParamEstimate:
    """Initialization of parameters for the EM search (MATLAB: initEM).

    Y: (ny, N) output (or list of such for multiple realizations);
    U: (nu, N) input (or list); x_guess: either an initial guess of the
    states (nx, N) array (or list), or an integer = number of states (the
    guess is then built from uncentered PCA of Y - (Y/U) U, with NaNs
    linearly interpolated first).

    Returns a ParamEstimate (A, B, C, D, Q, R, x0, P0); x0/P0 are lists when
    the data is a list. `rng` seeds the small random component of the PCA
    state guess (MATLAB uses the global generator).
    """
    is_list = isinstance(Y, list)
    if opts is None or opts.Niter is None:
        nu = (U[0] if is_list else np.atleast_2d(U)).shape[0]
        ny = (Y[0] if is_list else np.atleast_2d(Y)).shape[0]
        if x_guess is None:
            raise ValueError("init_em: x_guess must be a state guess (nx, N) "
                             "or an integer number of states")
        nx = int(x_guess) if np.isscalar(x_guess) else \
            (x_guess[0] if isinstance(x_guess, list)
             else np.atleast_2d(x_guess)).shape[0]
        opts = process_em_opts(opts, nu, ny=ny, nx=nx)
    rng = rng if isinstance(rng, np.random.Generator) else \
        np.random.default_rng(rng)

    if x_guess is None:
        raise ValueError("init_em: x_guess must be a state guess (nx, N) "
                         "or an integer number of states")
    if np.isscalar(x_guess):  # x_guess is the model order
        nx = int(x_guess)
        if is_list:
            Y2 = [substitute_nans(np.atleast_2d(y).T).T for y in Y]
        else:
            Y2 = substitute_nans(np.atleast_2d(Y).T).T
        X = _init_guess(Y2, U, nx, opts, rng)
    else:
        X = [np.atleast_2d(np.asarray(x, dtype=float)) for x in x_guess] \
            if isinstance(x_guess, list) else \
            np.atleast_2d(np.asarray(x_guess, dtype=float))

    # Initialize covariances to plausible values:
    if is_list:
        Pg = P_guess if P_guess is not None else [None] * len(Y)
        Ptg = Pt_guess if Pt_guess is not None else [None] * len(Y)
        pairs = [_init_cov(x, u, p, pt) for x, u, p, pt in zip(X, U, Pg, Ptg)]
        P = [p[0] for p in pairs]
        Pt = [p[1] for p in pairs]
    else:
        P, Pt = _init_cov(X, U, P_guess, Pt_guess)

    # Initial guesses of A, B, C, D, Q, R, x0, P0 via one M-step:
    return estimate_params(Y, U, X, P, Pt, opts)

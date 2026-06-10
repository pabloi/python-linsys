"""EM algorithm for LTI state-space identification (port of matlab-linsys/EM/EM.m)."""
from __future__ import annotations

import time
import warnings
from typing import NamedTuple

import numpy as np

from ..kalman import KalmanOpts, smoother_stationary
from .estimate import estimate_params
from .init import init_em
from .opts import EMOpts, process_em_opts


class EMResult(NamedTuple):
    """Result of em(). Multi-realization inputs make X/P/Pt/x0/P0 lists."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    X: "np.ndarray | list"   # (nx, N) smoothed states of the best model
    P: "np.ndarray | list"   # (N, nx, nx) smoothed covariances
    Pt: "np.ndarray | list"  # (N-1, nx, nx) smoothed transition covariances
    x0: "np.ndarray | list"
    P0: "np.ndarray | list"
    logL: float              # best log-likelihood found
    logl: np.ndarray         # per-iteration logL history (NaN-padded)
    msg: str                 # stopping reason
    out_log: "dict | None"   # extra log info (opts.log_flag)


def _kalman_opts(opts: EMOpts) -> KalmanOpts:
    return KalmanOpts(fast_flag=opts.fast_flag,
                      outlier_flag=opts.outlier_reject)


def _estimate_states(Y, A, C, Q, R, x0, P0, B, D, U, kopts):
    """E-step: smoothed states/covariances and logL (MATLAB: EM>estimateStates).

    For lists (multiple realizations) the logL of all realizations is summed."""
    if isinstance(Y, list):
        res = [smoother_stationary(y, A, C, Q, R, xi, pi, B, D, u, kopts)
               for y, xi, pi, u in zip(Y, x0, P0, U)]
        X = [r.Xs for r in res]
        P = [r.Ps for r in res]
        Pt = [r.Pt for r in res]
        l = float(sum(r.logL for r in res))
    else:
        r = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U, kopts)
        X, P, Pt, l = r.Xs, r.Ps, r.Pt, float(r.logL)
    return X, P, Pt, l


def _has_nan(X):
    if isinstance(X, list):
        return any(np.isnan(np.sum(x)) for x in X)
    return bool(np.isnan(np.sum(X)))


def _has_complex(X):
    if isinstance(X, list):
        return any(np.iscomplexobj(x) and np.any(np.imag(x) != 0) for x in X)
    return np.iscomplexobj(X) and bool(np.any(np.imag(X) != 0))


def _taus(A):
    """Time constants of A (for verbose printing)."""
    ev = np.sort(np.abs(np.linalg.eigvals(A)))
    with np.errstate(divide="ignore"):
        return -1.0 / np.log(ev)


def _check_stopping(l, k, opts, logl, drop_count, best_logl, X1):
    """Stopping logic (MATLAB: EM>checkStopping). k is the 1-based iteration.

    Returns (logl, drop_count, break_flag, msg). Updates logl[k] = l."""
    break_flag = False
    msg = ""
    if _has_complex(X1):
        msg = "Complex states detected, stopping."
        break_flag = True
    elif _has_nan(X1):
        msg = "States are NaN, stopping."
        break_flag = True

    logl[k] = l
    delta = l - logl[k - 1]
    improvement = delta >= 0
    logl_100_ago = logl[max(k - 100, 0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        target_rel_improvement_100 = ((l - logl_100_ago)
                                      / (opts.target_logl - logl_100_ago))
    below_target = max(l, best_logl) < opts.target_logl
    rel_improvement_last_100 = l - logl_100_ago

    # Warning conditions:
    if not improvement:
        # Drops in logL may happen with fast (steady-state) filtering and NaN
        # samples, with enforced-stable A, or from numerical issues.
        if abs(delta) > 1e-2:
            warnings.warn(f"EM:logLdrop - logL decreased at iteration {k}, "
                          f"drop = {delta}")
            drop_count += 1
    else:
        drop_count = 0

    # Failure conditions:
    if np.imag(l) != 0:
        msg = ("Complex logL, probably ill-conditioned matrices involved. "
               "Stopping.")
        break_flag = True

    # Early stopping (to avoid wasting time):
    nny = len(opts.include_output_idx)
    if (k > 100 and below_target
            and target_rel_improvement_100 < opts.target_tol
            and not opts.robust_flag):
        msg = "Unlikely to reach target value. Stopping."
        break_flag = True
    elif (k > 100 and rel_improvement_last_100 / nny < opts.convergence_tol
            and not opts.robust_flag):
        msg = "Increase is within tolerance (local max). Stopping."
        break_flag = True
    elif k == opts.Niter - 1:
        msg = "Max number of iterations reached. Stopping."
        break_flag = True
    elif drop_count > 10:
        msg = ("log-L dropped 10 consecutive times. Possibly ill-conditioned "
               "solution. Stopping.")
        break_flag = True
    return logl, drop_count, break_flag, msg


def em(Y, U, x_guess, opts: EMOpts | None = None, P_guess=None, Pt_guess=None,
       rng=None) -> EMResult:
    """EM identification of an LTI state-space model (MATLAB: EM).

    Y: (ny, N) output data; U: (nu, N) input data; x_guess: number of states
    (int) or initial guess of the state trajectories (nx, N). Y, U (and
    x_guess/P_guess/Pt_guess) may be lists of arrays for multiple
    realizations of the same system (logL is summed across realizations).

    Alternates E-steps (Kalman smoothing via kalman.smoother_stationary) and
    M-steps (estimate_params) keeping the best (highest-logL) model found.

    Deviations from MATLAB: returns x0/P0 too (MATLAB does not); the final
    exact (non-fast) state re-estimation uses the best parameters instead of
    the last iterate (the MATLAB code mixes best params with last-iterate
    states); final/per-iteration printing requires opts.verbose (default
    False).
    """
    is_list = isinstance(Y, list)
    if opts is not None and opts.log_flag:
        t_start = time.perf_counter()

    # Process options:
    if is_list:
        nu = np.atleast_2d(U[0]).shape[0]
        ny = np.atleast_2d(Y[0]).shape[0]
        Y = [np.atleast_2d(np.asarray(y, dtype=float)) for y in Y]
        U = [np.atleast_2d(np.asarray(u, dtype=float)) for u in U]
        any_nan = any(np.isnan(y).any() for y in Y)
    else:
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        U = np.atleast_2d(np.asarray(U, dtype=float))
        nu, ny = U.shape[0], Y.shape[0]
        any_nan = np.isnan(Y).any()
    if np.isscalar(x_guess):
        nx = int(x_guess)
    else:
        nx = (np.atleast_2d(x_guess[0]) if isinstance(x_guess, list)
              else np.atleast_2d(x_guess)).shape[0]
    opts = process_em_opts(opts, nu, nx, ny)
    if opts.fast_flag != 0 and any_nan:
        warnings.warn("EM:fastAndLoose - requested fast filtering but data "
                      "contains NaNs. Smoothing will be approximate and logL "
                      "is not guaranteed to be non-decreasing.")
    elif opts.fast_flag not in (0, 1):
        warnings.warn("EM:fastFewSamples - requested an exact number of "
                      "samples for fast filtering; this is an approximation "
                      "unless the slowest time-constant is much smaller.")

    out_log = {"opts": opts} if opts.log_flag else None

    # --- Init ---
    if is_list:
        Yred = [y[opts.include_output_idx, :] for y in Y]
    else:
        Yred = Y[opts.include_output_idx, :]
    A1, B1, C1, D1, Q1, R1, x01, P01 = init_em(Yred, U, x_guess, opts,
                                               P_guess, Pt_guess, rng=rng)
    kopts = _kalman_opts(opts)
    with warnings.catch_warnings():
        if opts.fast_flag != 0:
            warnings.filterwarnings("ignore", message=".*stat.*")
        X1, P1, Pt1, best_logl = _estimate_states(Yred, A1, C1, Q1, R1,
                                                  x01, P01, B1, D1, U, kopts)

    # Log-likelihood history & current best solution:
    logl = np.full(opts.Niter, np.nan)
    logl[0] = best_logl
    A, B, C, D, Q, R = A1, B1, C1, D1, Q1, R1
    x0, P0, X, P, Pt = x01, P01, X1, P1, Pt1
    if opts.target_logl is None:
        opts.target_logl = logl[0]

    # --- EM loop ---
    break_flag = False
    msg = ""
    if opts.verbose:
        print(f"Iter = 1, target logL = {opts.target_logl:.8g}, current "
              f"logL = {best_logl:.8g}, tau = {_taus(A)}")
    drop_count = 0
    disp_step = 100
    k = 0
    if opts.log_flag:
        out_log["eigs"] = []
        out_log["run_time"] = []
    for k in range(1, opts.Niter):  # MAIN LOOP
        # E-step: distribution of latent variables given current params.
        # M-step: parameters maximizing the expected data likelihood.
        if opts.log_flag:
            out_log["eigs"].append(np.sort(np.linalg.eigvals(A1)))
            out_log["run_time"].append(time.perf_counter() - t_start)
            t_start = time.perf_counter()

        # E-step:
        try:
            with warnings.catch_warnings():
                if opts.fast_flag != 0:
                    warnings.filterwarnings("ignore", message=".*stat.*")
                X1, P1, Pt1, l = _estimate_states(Yred, A1, C1, Q1, R1,
                                                  x01, P01, B1, D1, U, kopts)
        except (np.linalg.LinAlgError, ValueError) as e:
            # Ill-conditioned iterate (e.g. unstable A with Q ~ 0). MATLAB
            # has an (inactive) try-catch for this; we stop and keep the
            # best model found so far.
            msg = f"E-step failed ({e}), stopping."
            warnings.warn("EM:EstepFail - " + msg)
            break_flag = True
            break

        # Check stop conditions:
        with warnings.catch_warnings():
            if opts.fast_flag != 0 and any_nan:
                warnings.filterwarnings("ignore", message="EM:logLdrop.*")
            logl, drop_count, break_flag, msg = _check_stopping(
                l, k, opts, logl, drop_count, best_logl, X1)
        if l > best_logl:  # improvement: keep as best
            A, B, C, D, Q, R = A1, B1, C1, D1, Q1, R1
            x0, P0, X, P, Pt = x01, P01, X1, P1, Pt1
            best_logl = l

        # Print some info:
        if opts.verbose and k % disp_step == 0:
            last_change = l - logl[k - disp_step]
            print(f"Iter = {k}, dlogL = {last_change:.3g}, over target = "
                  f"{l - opts.target_logl:.3g}, tau = {_taus(A1)}")
        if break_flag and not opts.robust_flag:
            break

        # M-step:
        A1, B1, C1, D1, Q1, R1, x01, P01 = estimate_params(
            Yred, U, X1, P1, Pt1, opts)

    # --- Housekeeping ---
    if opts.fast_flag != 0:
        # Recompute optimal states & logL exactly (no fast/steady-state
        # shortcuts). NOTE: MATLAB uses the *last-iterate* params here, which
        # can overwrite the best-model states; we use the best params.
        try:
            X, P, Pt, best_logl = _estimate_states(
                Yred, A, C, Q, R, x0, P0, B, D, U, kopts.copy(fast_flag=0))
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn("EM:exactPassFail - exact final smoothing failed "
                          f"({e}); returning fast-mode states/logL.")

    if opts.log_flag:
        out_log["break_flag"] = break_flag
        out_log["msg"] = msg
        out_log["best_logl"] = best_logl
        out_log["logl"] = logl
    if opts.verbose:
        print(msg)
        print(f"Finished. Number of iterations = {k}, logL = "
              f"{best_logl:.8g}, tau = {_taus(A)}")

    # If some outputs were excluded, restore full-size C, R, D:
    nny = len(opts.include_output_idx)
    if nny < ny:
        Rfull = np.diag(np.full(ny, np.inf))
        Rfull[np.ix_(opts.include_output_idx, opts.include_output_idx)] = R
        R = Rfull
        Cfull = np.zeros((ny, nx))
        Cfull[opts.include_output_idx, :] = C
        C = Cfull
        # Recompute most likely D for the excluded outputs (least squares,
        # using non-NaN samples; MATLAB uses all samples):
        Yall = np.concatenate(Y, axis=1) if is_list else Y
        Uall = np.concatenate(U, axis=1) if is_list else U
        good = ~np.isnan(Yall).any(axis=0)
        Dfull = np.linalg.lstsq(Uall[:, good].T, Yall[:, good].T,
                                rcond=None)[0].T
        Dfull[opts.include_output_idx, :] = D
        D = Dfull

    return EMResult(A, B, C, D, Q, R, X, P, Pt, x0, P0, float(best_logl),
                    logl, msg, out_log)

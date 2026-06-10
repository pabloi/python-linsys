"""EM with multiple random restarts (port of matlab-linsys/EM/randomStartEM.m)."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import median_filter

from ..kalman import KalmanOpts, smoother_stationary
from ..utils import fwd_sim
from .em import EMResult, em
from .opts import EMOpts, process_em_opts


def _guess(nx, Y, U, opts, rng):
    """Random plausible state trajectories to restart EM from
    (MATLAB: randomStartEM>guess)."""
    if isinstance(U, list):
        y = np.concatenate(Y, axis=1)
        u = np.concatenate(U, axis=1)
    else:
        y, u = np.atleast_2d(Y), np.atleast_2d(U)
    y = y[opts.include_output_idx, :]
    ny, N = y.shape
    nu = u.shape[0]
    if opts.fix_a is None:
        # WLOG: diagonal A with log-uniform time-constants in [1, N/2]
        A1 = np.diag(np.exp(-1.0 / np.exp(np.log(N / 2) * rng.random(nx))))
    else:
        A1 = opts.fix_a
    B1 = np.ones((nx, nu)) if opts.fix_b is None else opts.fix_b  # WLOG
    if opts.fix_q is None:
        Q1 = (abs(rng.standard_normal()) + 1e-4) * np.eye(nx)  # needs PSD
    else:
        Q1 = opts.fix_q
    C1 = rng.standard_normal((ny, nx)) / ny if opts.fix_c is None \
        else opts.fix_c  # WLOG
    D1 = rng.standard_normal((ny, nu)) if opts.fix_d is None else opts.fix_d
    x0 = opts.fix_x0  # None by default
    P0 = opts.fix_p0
    _, x_sim = fwd_sim(u, A1, B1, np.zeros((1, nx)), np.zeros((1, nu)),
                       x0=x0, Q=Q1, R=None, rng=rng)
    if opts.fix_r is None:
        z = y - C1 @ x_sim[:, :-1] - D1 @ u
        z = z[:, ~np.isnan(z).any(axis=0)]
        R1 = z @ z.T / z.shape[1] + C1 @ Q1 @ C1.T  # reasonable estimate
    else:
        R1 = opts.fix_r
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # uninformative prior, fast mode
        x_guess = smoother_stationary(y, A1, C1, Q1, R1, x0, P0, B1, D1, u,
                                      KalmanOpts(fast_flag=opts.fast_flag)).Xs
    x_guess = np.nan_to_num(x_guess, nan=0.0)
    # Median-filter (window 9, zero-padded) to avoid very ugly estimates:
    x_guess = median_filter(x_guess, size=(1, 9), mode="constant", cval=0.0)
    if isinstance(U, list):
        splits = np.cumsum([np.atleast_2d(yy).shape[1] for yy in Y])[:-1]
        x_guess = [x.copy() for x in np.split(x_guess, splits, axis=1)]
    return x_guess


def random_start_em(Y, U, nx, opts: EMOpts | None = None,
                    rng=None) -> EMResult:
    """EM with random restarts, keeping the best logL (MATLAB: randomStartEM).

    Runs a short (100-iteration) EM from the default (PCA) initialization to
    set a benchmark, then opts.Nreps full EM runs from random state guesses,
    then refines the best solution (a fast pass when allowed, followed by a
    patient exact pass), each refinement controlled by the refine_* options.

    Y/U may be lists (multiple realizations). `rng` seeds all random draws
    (MATLAB uses the global generator).
    """
    rng = rng if isinstance(rng, np.random.Generator) else \
        np.random.default_rng(rng)
    if isinstance(U, list):
        nu = np.atleast_2d(U[0]).shape[0]
        ny = np.atleast_2d(Y[0]).shape[0]
    else:
        nu = np.atleast_2d(U).shape[0]
        ny = np.atleast_2d(Y).shape[0]

    base = EMOpts() if opts is None else opts
    opt_r = process_em_opts(base, nu, nx, ny)  # used only within this function
    verbose = opt_r.verbose

    # Rep 0: very fast evaluation of the default initialization (benchmark):
    if verbose:
        print("Starting rep 0 (short one)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="EM:logLdrop.*")
        warnings.filterwarnings("ignore", message="EM:fastAndLoose.*")
        best = em(Y, U, nx, base.copy(Niter=100), rng=rng)
    best_ll = best.logL

    opt2 = base.copy(target_logl=best_ll)
    last_success = 0
    for i in range(1, opt_r.Nreps + 1):
        if verbose:
            print(f"Starting rep {i}. Best logL so far = {best_ll:.8g} "
                  f"(rep = {last_success})")
        x_guess = _guess(nx, Y, U, opt_r, rng)
        res = em(Y, U, x_guess, opt2, rng=rng)
        if res.logL > best_ll:
            best = res
            best_ll = res.logL
            opt2 = opt2.copy(target_logl=best_ll)
            last_success = i
            if verbose:
                print(f"---- Success, best logL = {best_ll:.8g} "
                      f"(rep = {last_success}) ----")

    # Refinement of the best solution found:
    if (not opt_r.disable_refine and opt_r.refine_fast_flag
            and opt_r.fast_flag != 0):
        if verbose:
            print(f"Refining solution... (fast) Best logL so far = "
                  f"{best_ll:.8g}")
        opt_f = opt2.copy(
            Niter=opt_r.refine_max_iter,
            convergence_tol=opt_r.refine_tol / opt_r.fast_refine_tol_factor,
            target_tol=1e-4, fast_flag=50, target_logl=best_ll)
        res = em(Y, U, best.X, opt_f, P_guess=best.P, Pt_guess=best.Pt,
                 rng=rng)
        if res.logL > best_ll:
            best = res
            best_ll = res.logL
        elif verbose:
            print("Fast refining did not work (?)")

    if not opt_r.disable_refine:
        if verbose:
            print(f"Refining solution... (patient mode) Best logL so far = "
                  f"{best_ll:.8g}")
        opt_p = opt2.copy(Niter=opt_r.refine_max_iter, fast_flag=0,
                          convergence_tol=opt_r.refine_tol,
                          target_logl=best_ll)
        res = em(Y, U, best.X, opt_p, P_guess=best.P, Pt_guess=best.Pt,
                 rng=rng)
        if res.logL > best_ll:
            best = res
            best_ll = res.logL

    if verbose:
        print(f"End. Best logL = {best_ll:.8g}")
    return best

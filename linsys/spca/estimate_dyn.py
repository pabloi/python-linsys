"""Dynamics estimation for sPCA (port of matlab-linsys/sPCA).

MATLAB sources: estimateDynv3b.m -> estimate_dyn (alias estimate_dyn_v3b),
estimateDynv4.m -> estimate_dyn_v4.

Given a multivariate time series X (d, N), fits matrices J, V, K such that

    Xh[:, k+1] = J Xh[:, k],   Xh[:, 0] = 1,   X ~ V Xh + K

with J diagonal with real poles in (0, 1) (exponentially decaying states).
The fit is performed by nonlinear least squares over the time constants
tau (J = diag(exp(-1/tau))), projecting X onto the span of the decays.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.optimize

__all__ = ["EstimateDynResult", "estimate_dyn", "estimate_dyn_v3b",
           "estimate_dyn_v4"]

# Regularization for the analytic E@E.T computation: avoids solutions with
# (numerically ill-conditioned) double poles. 1e-2 keeps poles ~30 apart,
# 1e-4 ~4 apart (MATLAB comment).
_ALPHA = 1e-3
# MATLAB uses lb = 0 on tau, which makes the residual NaN if the bound is
# ever evaluated (0/0 in the decay computation); use a tiny positive bound.
_TAU_MIN = 1e-6


class EstimateDynResult(NamedTuple):
    """J: (order, order) diagonal; Xh: states incl. constant row when
    null_k is False; V: (d, order); K: (d, 1) or (d, 0); resnorm: residual
    sum of squares (MATLAB calls this output r2, misleadingly)."""
    J: np.ndarray
    Xh: np.ndarray
    V: np.ndarray
    K: np.ndarray
    resnorm: float


def _decays(tau, NN, null_k):
    """E (order(+1), NN): exponential decays exp(-t/tau), plus a row of ones
    when null_k is False (constant term)."""
    E = np.exp(-np.arange(NN)[None, :] / tau[:, None])
    if not null_k:
        E = np.vstack([E, np.ones((1, NN))])
    return E


def _comp_eet(e_tau, NN, null_k):
    """Analytic computation of E @ E.T (+ regularization) for the decay
    matrix, avoiding the O(order*N) product (MATLAB: compEEt)."""
    aN = e_tau ** NN
    M = (1.0 - np.outer(aN, aN)) / (1.0 - np.outer(e_tau, e_tau)) \
        + _ALPHA * np.eye(e_tau.size)
    if not null_k:
        E1 = (1.0 - aN) / (1.0 - e_tau)
        M = np.block([[M, E1[:, None]], [E1[None, :], np.array([[float(NN)]])]])
    return M


def _projector(tau, NN, null_k):
    """I - E.T inv(E E.T) E: projector onto the orthogonal complement of the
    decay subspace (MATLAB: projector)."""
    E = _decays(tau, NN, null_k)
    e_tau = np.exp(-1.0 / tau)
    EEt = _comp_eet(e_tau, NN, null_k)
    return np.eye(NN) - np.linalg.solve(EEt, E).T @ E


def estimate_dyn(X, real_poles_only=True, null_k=False, j0=2, rng=None):
    """Fit decaying-exponential linear dynamics to a time series
    (MATLAB: estimateDynv3b).

    X: (d, N) time series, columns are time samples.
    real_poles_only: only True is implemented (as in MATLAB).
    null_k: if True, no constant term K is included in the fit.
    j0: scalar model order (optimization restarted from 10 random
        initializations of the time constants), or an (order, order) initial
        guess for J (single optimization started from its eigenvalues).
    rng: seed or numpy Generator for the random initializations.

    Returns EstimateDynResult(J, Xh, V, K, resnorm). Note Xh includes the
    constant row (all ones) as its last row when null_k is False, so that
    X ~ [V K] @ Xh.

    Deviation from MATLAB: the source has a bug where the best residual is
    never updated across restarts (`bestRes` is not refreshed inside the
    loop), so later restarts are compared against the first one only; here
    the best solution so far is tracked correctly.
    """
    if not real_poles_only:
        raise NotImplementedError(
            "estimate_dyn: complex/double poles not implemented (as in "
            "MATLAB)")
    X = np.atleast_2d(np.asarray(X, dtype=float))
    NN = X.shape[1]
    rng = rng if isinstance(rng, np.random.Generator) \
        else np.random.default_rng(rng)

    j0_arr = np.asarray(j0, dtype=float)
    if j0_arr.size == 1 and j0_arr.item() >= 1:
        order = int(j0_arr.item())
        # Random initialization (MATLAB: randi(NN)+|randn|)
        t0 = rng.integers(1, NN + 1, order) + np.abs(
            rng.standard_normal(order))
        reps = 10
    else:
        order = j0_arr.shape[0]
        w = np.linalg.eigvals(j0_arr)
        t0 = -1.0 / np.log(np.real(w))
        reps = 1

    lb = _TAU_MIN
    ub = 5.0 * NN  # limits time constants so the slowest pole is not a line
    t0 = np.clip(np.asarray(t0, dtype=float), lb, ub)

    def residual(tau):
        return (X @ _projector(tau, NN, null_k)).ravel()

    def solve(t0):
        sol = scipy.optimize.least_squares(
            residual, t0, bounds=(lb, ub), xtol=1e-12, ftol=1e-14,
            gtol=1e-14, max_nfev=100000)
        return sol.x, 2.0 * sol.cost  # cost = 0.5 * sum of squares

    best_tau, best_res = solve(t0)
    for _ in range(1, reps):
        t0 = np.clip(NN * rng.random(order), lb, ub)
        tau, res = solve(t0)
        if res < best_res:
            best_tau, best_res = tau, res

    tau = best_tau
    Xh = _decays(tau, NN, null_k)
    J = np.diag(np.exp(-1.0 / tau))

    # Linear regression for the loadings: X ~ [V K] @ Xh
    VK = np.linalg.lstsq(Xh.T, X.T, rcond=None)[0].T
    V = VK[:, :order]
    K = VK[:, order:]  # (d, 1) if a constant was included, else (d, 0)
    return EstimateDynResult(J, Xh, V, K, best_res)


# Documented alias mapping the MATLAB version name explicitly:
estimate_dyn_v3b = estimate_dyn


def estimate_dyn_v4(X, real_poles_only=True, U=None, j0=2, rng=None):
    """Dynamics estimation with an input term (MATLAB: estimateDynv4).

    Fits Xh[:, k+1] = J Xh[:, k] + K u[k]. As in the MATLAB source, only the
    cases U == None/empty/all-zero (delegates with null_k=True) and U
    constant over time (delegates with null_k=False) are implemented; an
    arbitrary time-varying U raises NotImplementedError.

    Deviation from MATLAB: the constant-U cases delegate to estimate_dyn
    (the port of estimateDynv3b, with multiple random restarts) instead of
    the older single-start estimateDynv3, which lives in sPCA/old/ and is
    not ported.
    """
    if U is None or np.size(U) == 0 or np.all(np.asarray(U) == 0):
        return estimate_dyn(X, real_poles_only, null_k=True, j0=j0, rng=rng)
    U = np.atleast_2d(np.asarray(U, dtype=float))
    if np.all(U == U[:, :1]):  # constant (non-zero) input
        return estimate_dyn(X, real_poles_only, null_k=False, j0=j0, rng=rng)
    raise NotImplementedError(
        "estimate_dyn_v4: arbitrary time-varying inputs are not implemented "
        "(as in MATLAB)")

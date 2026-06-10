"""Kalman filter/smoother options (port of processKalmanOpts / processFastFlag)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class KalmanOpts:
    """Options for the stationary Kalman filter/smoother.

    fast_flag: 0/False = exact filtering (default); 1/True = auto-select the
    number of exact samples before assuming steady-state; any other integer =
    use that many exact samples.
    """
    fast_flag: int = 0
    outlier_flag: bool = False
    no_reduce_flag: bool = False
    no_logl: bool = False
    ind_d: np.ndarray | None = None  # columns of U that feed D (default: all)
    ind_b: np.ndarray | None = None  # columns of U that feed B (default: all)

    def copy(self, **kw):
        return replace(self, **kw)


def process_kalman_inputs(nx, N, x0=None, P0=None, B=None, D=None, U=None,
                          opts=None, ny=1):
    """Fill in defaults for optional filter arguments (MATLAB: processKalmanOpts).

    Returns (x0, P0, B, D, U, opts) as arrays of consistent shape.
    Default initial condition is x0 = 0 with infinite (improper) covariance.
    """
    if x0 is None:
        x0 = np.zeros(nx)
    x0 = np.asarray(x0, dtype=float).ravel()
    if P0 is None:
        P0 = np.diag(np.full(nx, np.inf))  # improper flat prior
    P0 = np.atleast_2d(np.asarray(P0, dtype=float))
    if U is None:
        nu = 0 if B is None else np.atleast_2d(B).shape[1]
        U = np.zeros((nu, N))
    if isinstance(U, list):
        nu = U[0].shape[0]
        if any(u.shape[0] != nu for u in U):
            raise ValueError("Multiple inputs given but with inconsistent sizes")
    else:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        nu = U.shape[0]
    if B is None:
        B = np.zeros((nx, nu))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if B.shape[1] != nu:
        if B.size == 0:
            warnings.warn("B was empty but U was not. Replacing B with 0")
            B = np.zeros((nx, nu))
        else:
            raise ValueError("Incompatible sizes of B, U")
    if D is None:
        D = np.zeros((ny, nu))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    if D.shape[1] != nu:
        if D.size == 0:
            warnings.warn("D was empty but U was not. Replacing D with 0")
            D = np.zeros((ny, nu))
        else:
            raise ValueError("Incompatible sizes of D, U")
    if opts is None:
        opts = KalmanOpts()
    if opts.fast_flag and opts.outlier_flag:
        warnings.warn("Requested fast mode AND outlier rejection, which is not "
                      "possible. Disabling fast mode.")
        opts = opts.copy(fast_flag=0)
    return x0, P0, B, D, U, opts


def process_fast_flag(fast_flag, A, N):
    """Number of samples M to filter exactly before steady-state shortcuts
    (MATLAB: processFastFlag)."""
    ev = np.abs(np.linalg.eigvals(A))
    if not fast_flag or int(fast_flag) >= N:
        M = N
    elif fast_flag == 1 or fast_flag is True:
        with np.errstate(divide="ignore"):
            tau = -1.0 / np.log(ev)
        M1 = int(np.ceil(3 * np.max(tau))) if np.all(ev < 1) else N
        M = min(max(M1, 20), N)
    else:
        M = min(int(np.ceil(abs(fast_flag))), N)
        with np.errstate(divide="ignore"):
            tau = -1.0 / np.log(ev[ev < 1])
        M1 = int(np.ceil(3 * np.max(tau))) if tau.size else N
        if M < N and M < M1:
            warnings.warn("statKSfast:fewSamples - number of samples for fast "
                          "filtering was provided, but system time-constants "
                          "indicate more are needed")
    if M < N and np.any(ev > 1):
        warnings.warn("statKSfast:unstable - steady-state (fast) filtering on "
                      "an unstable system would diverge. Doing exact filtering.")
        M = N
    return M

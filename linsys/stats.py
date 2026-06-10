"""Data log-likelihood under a linear state-space model
(port of matlab-linsys/misc: dataLogLikelihood.m, logLincomplete.m,
logLComplete.m).

All functions accept lists of (Y, U, ...) for multiple realizations of the
same system (MATLAB cell-array inputs).
"""
from __future__ import annotations

import warnings

import numpy as np

from .utils import cholcov, logl_normal
from .kalman import filter_stationary, KalmanOpts

__all__ = ["data_log_likelihood", "logl_incomplete", "logl_complete"]


def _as2d(M):
    return np.atleast_2d(np.asarray(M, dtype=float))


def _check_r(R):
    """Warn if R is not PD (MATLAB does the same via mycholcov)."""
    R = _as2d(R)
    _, r = cholcov(R)
    if r != R.shape[0]:
        warnings.warn("R is not PD, this can end badly")


def _r_plus_cpc(C, R, P):
    """R + C P C', summed through Cholesky factors so the result is PSD
    (MATLAB: RplusCPC in logLincomplete.m)."""
    cP1, _ = cholcov(P)
    CcP = C @ cP1.T
    cR, r = cholcov(R)
    S = cR.T @ cR + CcP @ CcP.T
    _, rs = cholcov(S)
    if rs < C.shape[0]:
        warnings.warn("dataLogL:nonPDcov - R+C*P*C^t was not positive "
                      "definite. LogL is not defined. Regularizing to move "
                      "forward.")
        S = S + 1e-11 * np.eye(C.shape[0])
    return S


def _logl_exact(z, Pp, C, R):
    """Sum over samples of log N(z_k; 0, R + C Pp_k C')."""
    total = 0.0
    for i in range(z.shape[1]):
        ll, _ = logl_normal(z[:, i], _r_plus_cpc(C, R, Pp[i]))
        total += ll
    return float(total)


def _logl_approx(z, Pp, C, R):
    """Approximate logL using the median (steady-state proxy) uncertainty."""
    mPP = np.median(Pp, axis=0)
    ll, _ = logl_normal(z, _r_plus_cpc(C, R, mPP))
    return float(np.sum(ll))


def _logl_max(z):
    """Maximum achievable logL over all (fixed) innovation covariances:
    attained at P = sample covariance of z.

    Note: the MATLAB ``logLopt`` in dataLogLikelihood.m omits the factor N
    (it returns the per-sample value), which makes it incomparable with the
    other methods; the factor is restored here so that
    ``max >= approx`` holds exactly and ``max ~>= exact`` in practice.
    """
    d, N = z.shape
    S = (z @ z.T) / N
    w = np.linalg.eigvalsh((S + S.T) / 2)
    with np.errstate(divide="ignore"):
        logdetS = float(np.sum(np.log(w)))
    return -0.5 * N * (d + d * np.log(2 * np.pi) + logdetS)


def _fast_split(z, A):
    """Number of leading samples to treat exactly in 'fast' mode."""
    N = z.shape[1]
    ev = np.abs(np.linalg.eigvals(np.atleast_2d(A)))
    ev = np.clip(ev, np.finfo(float).tiny, 1 - 1e-15)
    M1 = int(np.ceil(3 * np.max(-1.0 / np.log(ev))))
    return min(max(M1, 20), N)


def _logl_fast(z, Pp, C, R, A):
    M = _fast_split(z, A)
    if M >= z.shape[1]:
        return _logl_exact(z, Pp, C, R)
    return (_logl_exact(z[:, :M], Pp[:M], C, R)
            + _logl_approx(z[:, M:], Pp[M:], C, R))


def _innovations(Y, U, C, D, Xp, Pp):
    """One-step-ahead residuals and matching prediction covariances,
    NaN samples removed."""
    pred_y = C @ Xp[:, :-1] + D @ U
    z = Y - pred_y
    idx = ~np.isnan(Y).any(axis=0)
    return z[:, idx], Pp[:-1][idx]


def data_log_likelihood(Y, U, A, B, C, D, Q, R, x0=None, P0=None,
                        method="exact"):
    """Log-likelihood of data under a model (MATLAB: dataLogLikelihood).

    Two modes, as in MATLAB:

    - ``x0`` is None or a single initial-state guess (and ``P0`` its
      covariance): the Kalman filter is run and its **exact** total logL is
      returned regardless of ``method`` (with a warning if a different method
      was requested), because the filter computes it at no extra cost.
    - ``x0``/``P0`` contain the full one-step-ahead predictions
      (``Xp`` (nx, N+1) and ``Pp`` (N+1, nx, nx), e.g. from a previous filter
      pass): the 'incomplete-data' likelihood
      p({y} | params) [Albert & Shadmehr 2017, eq. A1.25] is computed from
      the innovations with the requested ``method``:

      - 'exact': sum of log N(z_k; 0, R + C Pp_k C') (slowest).
      - 'approx': uses the median Pp as a steady-state proxy.
      - 'fast': 'exact' for the first M samples (M from the slowest time
        constant of A, at least 20), 'approx' for the rest.
      - 'max': maximum achievable logL over innovation covariances
        (P = sample covariance of the innovations); an upper bound for
        'approx' (and, in practice, for 'exact'). Note: the MATLAB version
        omits a factor N here; this port restores it (see ``_logl_max``).

    ``Y``/``U`` may be lists (multiple realizations): the total logL is the
    sum over realizations (``x0``/``P0`` may be lists too, or shared).
    Returns a float (total log-likelihood, NOT normalized per sample).
    """
    _check_r(R)
    if isinstance(Y, (list, tuple)):
        if not isinstance(x0, (list, tuple)):
            return float(sum(
                data_log_likelihood(y, u, A, B, C, D, Q, R, x0, P0, method)
                for y, u in zip(Y, U)))
        return float(sum(
            data_log_likelihood(y, u, A, B, C, D, Q, R, xx, pp, method)
            for y, u, xx, pp in zip(Y, U, x0, P0)))

    Y = _as2d(Y)
    U = _as2d(U)
    x0a = None if x0 is None else np.asarray(x0, dtype=float)
    if x0a is None or x0a.ndim <= 1 or x0a.shape[1] <= 1:
        # True initial-state guess: run the filter, which gives the exact logL
        res = filter_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U,
                                opts=KalmanOpts(fast_flag=False))
        if method != "exact":
            warnings.warn("dataLogL:ignoreMethod - method requested was not "
                          "'exact', but returning exact log-likelihood "
                          "anyway, because it's faster.")
        return float(res.logL)

    # Full one-step-ahead predictions were provided (Xp, Pp):
    Xp, Pp = x0a, np.asarray(P0, dtype=float)
    C, D, R = _as2d(C), _as2d(D), _as2d(R)
    z, Ppz = _innovations(Y, U, C, D, Xp, Pp)
    if method == "approx":
        return _logl_approx(z, Ppz, C, R)
    elif method == "exact":
        return _logl_exact(z, Ppz, C, R)
    elif method == "max":
        return _logl_max(z)
    elif method == "fast":
        return _logl_fast(z, Ppz, C, R, _as2d(A))
    raise ValueError(f"data_log_likelihood: unknown method '{method}'")


def logl_incomplete(Y, U, A, B, C, D, Q, R, x0=None, P0=None,
                    method="approx"):
    """Incomplete-data log-likelihood p({y} | params), normalized per sample
    and per output dimension (MATLAB: logLincomplete).

    Same methods as :func:`data_log_likelihood` ('approx' is the default, as
    'exact' is slow). If ``x0``/``P0`` are omitted, a near-uninformative
    prior is used (with a warning). If they contain the full predicted states
    ``Xp`` (nx, N+1) / ``Pp`` (N+1, nx, nx), filtering is skipped (useful
    inside EM).

    Note: the MATLAB source sets the default prior covariance to
    ``1e8*zeros(...)`` which is all zeros (an apparent typo for ``1e8*eye``);
    this port uses ``1e8 * I``.

    ``Y``/``U``/``x0``/``P0`` may be lists; the result is the sample-size
    weighted average of the per-realization values.
    """
    _check_r(R)
    if isinstance(Y, (list, tuple)):
        vals = [logl_incomplete(y, u, A, B, C, D, Q, R, xx, pp, method)
                for y, u, xx, pp in zip(Y, U, x0, P0)]
        sizes = np.array([np.atleast_2d(y).shape[1] for y in Y], dtype=float)
        return float(np.dot(vals, sizes) / sizes.sum())

    Y, U = _as2d(Y), _as2d(U)
    A = _as2d(A)
    if x0 is None or P0 is None:
        warnings.warn("logLincomplete:noPriorGiven - no prior was provided. "
                      "Assuming an uninformative prior.")
        x0 = np.zeros(A.shape[0])
        P0 = 1e8 * np.eye(A.shape[0])
    x0a = np.asarray(x0, dtype=float)
    if x0a.ndim <= 1 or x0a.shape[1] == 1:
        P0a = np.asarray(P0, dtype=float)
        if P0a.ndim == 3:
            P0a = P0a[0]
        res = filter_stationary(Y, A, C, Q, R, x0=x0a.ravel(), P0=P0a,
                                B=B, D=D, U=U,
                                opts=KalmanOpts(fast_flag=False))
        Xp, Pp = res.Xp, res.Pp
    else:
        Xp, Pp = x0a, np.asarray(P0, dtype=float)

    C, D, R = _as2d(C), _as2d(D), _as2d(R)
    z, Ppz = _innovations(Y, U, C, D, Xp, Pp)
    d, N = z.shape
    if method == "approx":
        total = _logl_approx(z, Ppz, C, R)
    elif method == "exact":
        total = _logl_exact(z, Ppz, C, R)
    elif method == "max":
        total = _logl_max(z)
    elif method == "fast":
        total = _logl_fast(z, Ppz, C, R, A)
    else:
        raise ValueError(f"logl_incomplete: unknown method '{method}'")
    return total / (N * d)


def logl_complete(Y, U, A, B, C, D, Q, R, X):
    """Complete-data log-likelihood p({y}, {x} | params), normalized per
    sample and per output dimension (MATLAB: logLcomplete in logLComplete.m).

    ``X`` is the (nx, N+1) state sequence (one more sample than Y, so that
    state residuals ``w[k] = x[k+1] - A x[k] - B u[k]`` are defined for all
    k); an (nx, N) X is also accepted (the last transition is then dropped).

    Notes on the MATLAB source (faithfulness deviations):
    - MATLAB computes the state residual input term as ``B*U(1:end-1)``
      (linear indexing: only correct for single-input systems); this port
      uses ``B @ U[:, :-1]``.
    - The MATLAB cell branch references undefined variables (``X0``, ``P0``)
      and would error; lists are supported here (sample-size weighted
      average, as in logLincomplete).
    - Rank-deficient Q/R are handled through the pseudo-inverse
      (pseudo-likelihood on the support of the covariance).

    ``method`` has no effect in the MATLAB source (the switch does not
    exist); it is not accepted here.
    """
    _check_r(R)
    if isinstance(Y, (list, tuple)):
        vals = [logl_complete(y, u, A, B, C, D, Q, R, x)
                for y, u, x in zip(Y, U, X)]
        sizes = np.array([np.atleast_2d(y).shape[1] for y in Y], dtype=float)
        return float(np.dot(vals, sizes) / sizes.sum())

    Y, U, X = _as2d(Y), _as2d(U), _as2d(X)
    A, B, C, D = _as2d(A), _as2d(B), _as2d(C), _as2d(D)
    Q, R = _as2d(Q), _as2d(R)
    N = Y.shape[1]
    z = Y - (C @ X[:, :N] + D @ U)
    idx = ~np.isnan(Y).any(axis=0)
    z = z[:, idx]
    nT = min(X.shape[1] - 1, N)  # transitions available
    w = X[:, 1:nT + 1] - (A @ X[:, :nT] + B @ U[:, :nT])

    llz, _ = logl_normal(z, R)
    llw, _ = logl_normal(w, Q)
    d, n2 = z.shape
    return float((np.sum(llz) + np.sum(llw)) / (n2 * d))

"""Cross-validated EM (port of matlab-linsys/EM/CV_EM.m).

The MATLAB CV_EM is stale: it calls randomStartEM with a signature that no
longer exists, passes the full-rate input U together with the decimated
output, and references helpers (`upsample`, `canonizev2`) that are not in
the repository. This port keeps the documented intent: folds are built by
regularly-interleaved decimation (1-out-of-nfolds samples per fold), a model
is fit to each fold at the decimated rate, converted back to the full sample
rate, and the states are re-estimated on the full data (with the held-out
samples as missing). Models are returned in canonical form so they can be
compared across folds.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

from ..kalman import KalmanOpts, smoother_stationary
from ..utils import canonize, cholcov
from .opts import EMOpts
from .random_start import random_start_em


def _upsample_dynamics(A, B, Q, nfolds):
    """Convert dynamics fitted at 1-every-nfolds samples to per-sample rate.

    Solves x[k+nfolds] = A x[k] + B u + w, w~N(0,Q) for the per-sample
    (A1, B1, Q1) assuming constant input within each stride:
    A = A1^nfolds, B = (sum_j A1^j) B1, Q = sum_j A1^j Q1 A1'^j.
    """
    nx = A.shape[0]
    A1 = np.real(scipy.linalg.fractional_matrix_power(A, 1.0 / nfolds))
    powers = [np.linalg.matrix_power(A1, j) for j in range(nfolds)]
    S = sum(powers)
    B1 = np.linalg.solve(S, B)
    K = sum(np.kron(p, p) for p in powers)
    Q1 = np.linalg.solve(K, Q.ravel()).reshape(nx, nx)
    Q1 = (Q1 + Q1.T) / 2
    cQ, _ = cholcov(Q1)
    Q1 = cQ.T @ cQ  # enforce PSD
    return A1, B1, Q1


def cv_em(Y, U, nx, nfolds=2, opts: EMOpts | None = None, rng=None):
    """Cross-validated EM via interleaved decimation folds (MATLAB: CV_EM).

    Y: (ny, N) output; U: (nu, N) input; nx: number of states; nfolds:
    number of interleaved folds. Each fold is fit with random_start_em on
    Y[:, i::nfolds] (and the correspondingly decimated input), the dynamics
    are converted to the full sample rate, and the states re-estimated by
    smoothing the full-rate data with the held-out samples set to NaN.

    Returns a list of nfolds dicts with keys A, B, C, D, Q, R, X, logL
    (model in 'canonical' form, X at the full sample rate).
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    rng = rng if isinstance(rng, np.random.Generator) else \
        np.random.default_rng(rng)
    models = []
    for i in range(nfolds):
        Yred = Y[:, i::nfolds]
        Ured = U[:, i::nfolds]
        res = random_start_em(Yred, Ured, nx, opts, rng=rng)

        # Transform A, B, Q to describe per-sample dynamics:
        A1, B1, Q1 = _upsample_dynamics(res.A, res.B, res.Q, nfolds)

        # Re-estimate states at the full rate (held-out samples are missing):
        Y2 = np.full_like(Y, np.nan)
        Y2[:, i::nfolds] = Yred
        sres = smoother_stationary(Y2, A1, res.C, Q1, res.R, None, None,
                                   B1, res.D, U, KalmanOpts())

        # Use a canonical form so models can be compared across folds:
        A2, B2, C2, X2, _, Q2, _ = canonize(A1, B1, res.C, X=sres.Xs, Q=Q1,
                                            method="canonical")
        models.append({"A": A2, "B": B2, "C": C2, "D": res.D, "Q": Q2,
                       "R": res.R, "X": X2, "logL": res.logL})
    return models

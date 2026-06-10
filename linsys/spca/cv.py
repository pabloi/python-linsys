"""Cross-validated sPCA (port of matlab-linsys/sPCA/crossval/CVsPCA.m)."""
from __future__ import annotations

import numpy as np
import scipy.linalg

from .spca import SPCAModel, _chng_init_state, spca

__all__ = ["cv_spca"]


def cv_spca(Y, dyn_order=2, force_pcs=False, null_bd=False,
            output_under_rank=0, n_folds=2, rng=None):
    """Cross-validated sPCA (MATLAB: CVsPCA).

    Folds take 1-out-of-n_folds time samples, regularly interleaved
    (fold f uses Y[:, f::n_folds]). Each fold's model is re-expressed at the
    original (per-sample) time scale: J -> J**(1/n_folds), B rescaled so the
    n_folds-step composition matches the fold dynamics, the initial state
    re-defined so all folds impose the same initial condition at the first
    (true) sample, and X re-computed for every sample.

    Y: (ny, N), columns are time samples (transposed vs. MATLAB).
    Returns a list of n_folds SPCAModel.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    N = Y.shape[1]
    models = []
    for f in range(n_folds):
        model = spca(Y[:, f::n_folds], dyn_order, force_pcs, null_bd,
                     output_under_rank, rng=rng)
        # Per-sample dynamics instead of per-n_folds-samples:
        J = np.real(scipy.linalg.fractional_matrix_power(model.J,
                                                         1.0 / n_folds))
        I = np.eye(J.shape[0])
        Jn = np.linalg.matrix_power(J, n_folds)  # == fold-scale J
        B = np.linalg.solve(I - Jn, (I - J) @ model.B)
        # Re-define the initial state so all folds impose the same initial
        # condition for the first (true) sample, even if not fitting it:
        Jf = np.linalg.matrix_power(J, f)
        new_x0 = Jf @ model.X[:, :1] \
            + np.linalg.solve(I - J, (I - Jf) @ B)
        B, D, X = _chng_init_state(J, B, model.C, model.D, model.X, new_x0)

        # Re-compute X to get a state estimate for every sample:
        n_fold_samples = X.shape[1]
        width = int(np.ceil(N / n_folds)) * n_folds
        Xn = np.full((X.shape[0], width), np.nan)
        for k in range(n_folds):
            cols = k + n_folds * np.arange(n_fold_samples)
            if k == f:
                Xn[:, cols] = X
            else:
                Jk = np.linalg.matrix_power(J, k - f)
                Xn[:, cols] = Jk @ X \
                    + np.linalg.solve(I - J, (I - Jk) @ B)
        X = Xn[:, :N]

        models.append(SPCAModel(C=model.C, J=J, X=X, B=B, D=D, r2=model.r2))
    return models

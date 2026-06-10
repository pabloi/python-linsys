"""Model-selection helpers (port of matlab-linsys/misc: bicaic.m,
foldSplit.m, bestPairedMatch.m)."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["BICAICResult", "bic_aic", "fold_split", "best_paired_match"]


class BICAICResult(NamedTuple):
    BIC: float
    AIC: float
    BICalt: float


def _get(model, *names):
    for n in names:
        if n in model:
            return np.atleast_2d(np.asarray(model[n], dtype=float))
    raise KeyError(f"model must contain one of {names}")


def bic_aic(model, Y, logL) -> BICAICResult:
    """BIC/AIC-style information criteria for a fitted model
    (MATLAB: bicaic).

    Parameters
    ----------
    model : dict with keys 'J' (or 'A'), 'B', 'C', 'D', 'Q', 'R'.
    Y : (ny, N) data the model was fit to (only N is used).
    logL : total log-likelihood of the fit.

    Returns
    -------
    BICAICResult(BIC, AIC, BICalt). NOTE: these follow the MATLAB sign
    convention ``BIC = 2*logL - log(N)*k`` and ``AIC = 2*logL - 2*k``
    (HIGHER is better), which is the negative of the textbook definitions.

    The number of free parameters k is counted heuristically as the number
    of nonzero entries (exact zeros are presumed fixed, not free):
    ``k = nnz(J) + (nnz(B) - nx) + nnz(C) + nnz(D) + nnz(triu(Q)) +
    nnz(triu(R))`` (one column of B can be arbitrarily scaled, hence the
    ``- nx``). ``BICalt`` additionally counts the states as parameters of a
    matrix-factorization problem: ``k_alt = k + nx*N - nnz(triu(Q)) -
    nnz(triu(R)) - nnz(J) - (nnz(B) - nx)``.
    (The MATLAB source also contains a parametric count which it immediately
    overwrites with this nonzero-based one; only the effective version is
    ported.)
    """
    J = _get(model, "J", "A")
    B = _get(model, "B")
    C = _get(model, "C")
    D = _get(model, "D")
    Q = _get(model, "Q")
    R = _get(model, "R")
    M = J.shape[0]
    N = np.atleast_2d(Y).shape[1]

    Na = int(np.count_nonzero(J))
    Nb = int(np.count_nonzero(B)) - M
    Nc = int(np.count_nonzero(C))
    Nd = int(np.count_nonzero(D))
    Nq = int(np.count_nonzero(np.triu(Q)))
    Nr = int(np.count_nonzero(np.triu(R)))
    k = Na + Nb + Nc + Nd + Nq + Nr
    k_alt = k + M * N - Nq - Nr - Na - Nb

    BIC = 2 * logL - np.log(N) * k
    BICalt = 2 * logL - np.log(N) * k_alt
    AIC = 2 * logL - 2 * k
    return BICAICResult(float(BIC), float(AIC), float(BICalt))


def fold_split(data, n_folds, axis=1):
    """Split data into interleaved folds for cross-validation
    (MATLAB: foldSplit).

    Fold i keeps samples ``i, i+n_folds, i+2*n_folds, ...`` along ``axis``
    and replaces every other sample with NaN, so each fold has the same
    shape as the original data and can be fed directly to NaN-aware system
    identification (EM, Kalman filtering).

    NOTE: the MATLAB version folds along the FIRST dimension (callers pass
    transposed, time-by-channel data). Per PORTING.md, time runs along
    columns here, so the default is ``axis=1``; pass ``axis=0`` for the
    MATLAB behavior.

    Returns a list of ``n_folds`` arrays.
    """
    data = np.asarray(data, dtype=float)
    folded = []
    for i in range(n_folds):
        f = np.full_like(data, np.nan)
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(i, None, n_folds)
        f[tuple(sl)] = data[tuple(sl)]
        folded.append(f)
    return folded


def best_paired_match(vec1, vec2):
    """Greedy pairing of two value vectors (MATLAB: bestPairedMatch).

    Heuristically minimizes ``norm(vec1[ind1] - vec2[ind2])`` by assigning,
    for each element of ``vec2`` in order, the closest not-yet-matched
    element of ``vec1``. Useful to match eigenvalues/time-constants between
    two fitted models.

    Returns
    -------
    (ind1, ind2): integer arrays (0-based; MATLAB returns 1-based) with
    ``len(ind1) == len(vec1)`` and ``ind2 == arange(len(vec2))``.
    ``ind1[i]`` is the index into ``vec1`` matched with ``vec2[i]`` for
    ``i < len(vec2)``; any remaining entries of ``ind1`` list the unmatched
    elements of ``vec1``.
    """
    v1 = np.asarray(vec1, dtype=float).ravel()
    v2 = np.asarray(vec2, dtype=float).ravel()
    not_matched = np.ones(v1.size, dtype=bool)
    ind1 = np.full(v1.size, -1, dtype=int)
    i = 0
    while i < v2.size and not_matched.any():
        dif = np.abs(v1 - v2[i])
        best = np.min(dif[not_matched])
        closest = np.nonzero((dif == best) & not_matched)[0][0]
        ind1[i] = closest
        not_matched[closest] = False
        i += 1
    leftovers = np.nonzero(not_matched)[0]
    ind1[ind1 == -1] = leftovers
    ind2 = np.arange(v2.size)
    return ind1, ind2

"""Viterbi decoding for discrete HMMs (port of matlab-linsys/discrHMM/viterbi.m).

Matrix orientation (see :mod:`linsys.hmm.helpers`): the transition matrix
``T`` is **column-stochastic**, ``T[i, j] = p(x[k+1] = i | x[k] = j)``
(columns sum to 1, as enforced by MATLAB's ``columnNormalize``), and the
emission matrix ``O`` is (M, D) column-stochastic, ``O[m, d] = p(y=m | x=d)``.

``nonStatViterbi.m`` is NOT ported: the MATLAB source raises an error
unconditionally ("This function has not been tested") and is dead code.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from .helpers import column_normalize

__all__ = ["ViterbiResult", "viterbi"]


class ViterbiResult(NamedTuple):
    opt_seq: np.ndarray  # (N,) most likely state sequence (ints in 0..D-1)
    logL: float          # joint log-likelihood log p(y_{0..N-1}, x*_{0..N-1})


def viterbi(observations, transition_matrix, emission_matrix,
            prior_p=None) -> ViterbiResult:
    """Most likely state sequence given observations (MATLAB: viterbi).

    Parameters
    ----------
    observations : (N,) int array of observed symbols in ``0..M-1``
        (MATLAB uses 1-based symbols).
    transition_matrix : (D, D) column-stochastic,
        ``T[i, j] = p(x[k+1]=i | x[k]=j)``. Columns are normalized if needed.
    emission_matrix : (M, D) column-stochastic, ``O[m, d] = p(y=m | x=d)``.
    prior_p : optional (D,) prior over the initial state (uniform if omitted,
        with a warning, as in MATLAB).

    Returns
    -------
    ViterbiResult(opt_seq, logL): the MAP state sequence (0-based) and its
    joint log-likelihood.
    """
    obs = np.asarray(observations).ravel().astype(int)
    N = obs.size
    T = column_normalize(np.asarray(transition_matrix, dtype=float))
    O = column_normalize(np.asarray(emission_matrix, dtype=float))
    D = T.shape[0]
    if prior_p is None:
        warnings.warn("Prior not given, using uniform prior.")
        prior_p = np.ones(D) / D
    prior_p = np.asarray(prior_p, dtype=float).ravel()

    with np.errstate(divide="ignore"):  # log(0) = -inf is fine here
        lE = np.log(O).T  # (D, M): lE[:, m] = log p(y=m | x=.)
        lT = np.log(T)
        lp0 = np.log(prior_p)

    # optimal_logl[d] = max over paths ending in state d of the joint logL
    optimal_logl = lp0 + lE[:, obs[0]]
    mle_prev = np.zeros((D, N - 1), dtype=int) if N > 1 else \
        np.zeros((D, 0), dtype=int)
    for i in range(1, N):
        # aux[i_next, j_prev] = logl[j_prev] + lT[i_next, j_prev] + lE[i_next]
        aux = optimal_logl[None, :] + lT + lE[:, obs[i]][:, None]
        mle_prev[:, i - 1] = np.argmax(aux, axis=1)
        optimal_logl = np.max(aux, axis=1)

    opt_seq = np.empty(N, dtype=int)
    opt_seq[N - 1] = int(np.argmax(optimal_logl))
    logL = float(optimal_logl[opt_seq[N - 1]])
    for k in range(N - 2, -1, -1):
        opt_seq[k] = mle_prev[opt_seq[k + 1], k]
    return ViterbiResult(opt_seq, logL)

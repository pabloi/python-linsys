"""Parameter estimation (EM) for discrete HMMs
(port of matlab-linsys/discrHMM/HMM_EM.m and HMMmatrixEstim.m).

Matrix orientation (see :mod:`linsys.hmm.helpers`): transition matrix ``T``
is (D, D) **column-stochastic** (``T[i, j] = p(next=i | curr=j)``, columns
sum to 1); observation matrix ``O`` is (M, D) column-stochastic. State
posteriors are (D, N) with columns as time samples (note: MATLAB's
``HMMmatrixEstim`` indexes ``stateDistrHistory`` as (N, D), rows = time,
which is inconsistent with what ``HMMstationaryInference`` returns — see
the bug notes in the port report; here the (D, N) convention is used
throughout).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .helpers import column_normalize, hmm_logl
from .inference import hmm_stationary_inference

__all__ = ["hmm_matrix_estim", "HMMEMResult", "hmm_em"]


def hmm_matrix_estim(state_distr, observations, n_symbols=None):
    """M-step estimate of the transition/observation matrices
    (MATLAB: HMMmatrixEstim).

    Faithful port of the MATLAB estimator, which approximates the pairwise
    posterior by the product of the smoothed marginals:
    ``joint[i, j] ~ mean_k p(x[k+1]=i | Y) p(x[k]=j | Y)`` (this is NOT the
    exact Baum-Welch xi-statistic; it is exact only when the posteriors are
    nearly deterministic).

    Parameters
    ----------
    state_distr : (D, N) smoothed state posteriors, columns are time samples.
    observations : (N,) int array of observed symbols in ``0..M-1``.
    n_symbols : optional number of symbols M (default ``max(observations)+1``,
        matching MATLAB's ``max(observationHistory)``).

    Returns
    -------
    (transition_matrix, observation_matrix): (D, D) and (M, D), both
    column-stochastic.
    """
    p = np.asarray(state_distr, dtype=float)
    D, N = p.shape
    obs = np.asarray(observations).ravel().astype(int)
    if obs.size != N:
        raise ValueError("observations must have one entry per column of "
                         "state_distr")
    # joint[i, j] = sum_k p[i, k+1] * p[j, k] / (N-1)
    joint = p[:, 1:] @ p[:, :-1].T / (N - 1)
    transition_matrix = column_normalize(joint)

    if n_symbols is None:
        n_symbols = int(obs.max()) + 1
    joint_obs = np.zeros((n_symbols, D))
    for m in range(n_symbols):
        joint_obs[m, :] = p[:, obs == m].sum(axis=1)
    observation_matrix = column_normalize(joint_obs)
    return transition_matrix, observation_matrix


class HMMEMResult(NamedTuple):
    observation_matrix: np.ndarray  # (M, D) column-stochastic
    transition_matrix: np.ndarray   # (D, D) column-stochastic
    state_distr: np.ndarray         # (D, N) smoothed posteriors at the optimum
    logl: np.ndarray                # (n_iter,) log-likelihood per iteration


def hmm_em(observations, p0, observation_matrix=None, transition_matrix=None,
           n_symbols=None, max_iter=100, tol=1e-8, rng=None) -> HMMEMResult:
    """Baum-Welch-style EM fit of a discrete HMM (MATLAB: HMM_EM).

    E-step: forward-backward smoothing (:func:`hmm_stationary_inference`);
    M-step: :func:`hmm_matrix_estim` (the marginal-product approximation of
    the MATLAB source).

    Deviations from the MATLAB original (which is an unfinished draft):
    - MATLAB initializes the matrices with ``randn`` (which yields negative
      "probabilities"); here random initialization draws uniform positive
      entries and column-normalizes.
    - MATLAB hard-codes 2 iterations (a debugging leftover, ``while i<2%100``)
      and computes no likelihood; here EM runs up to ``max_iter`` iterations
      and stops early when the log-likelihood improves by less than ``tol``.

    Parameters
    ----------
    observations : (N,) int array of observed symbols in ``0..M-1``.
    p0 : (D,) initial-state distribution (held fixed, as in MATLAB).
    observation_matrix : optional (M, D) initial guess.
    transition_matrix : optional (D, D) initial guess.
    n_symbols : number of symbols M (required if observation_matrix is None
        and you want more symbols than appear in the data).
    max_iter, tol : iteration cap and minimum logL improvement.
    rng : seed or Generator for the random initialization.

    Returns an :class:`HMMEMResult`. ``logl[k]`` is the log-likelihood of the
    parameters available at the start of iteration k (non-decreasing up to
    the M-step approximation).
    """
    obs = np.asarray(observations).ravel().astype(int)
    p0 = np.asarray(p0, dtype=float).ravel()
    D = p0.size
    if n_symbols is None:
        n_symbols = (int(obs.max()) + 1 if observation_matrix is None
                     else np.asarray(observation_matrix).shape[0])
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    if observation_matrix is None:
        observation_matrix = column_normalize(rng.uniform(.1, 1., (n_symbols, D)))
    else:
        observation_matrix = column_normalize(
            np.asarray(observation_matrix, dtype=float))
    if transition_matrix is None:
        transition_matrix = column_normalize(rng.uniform(.1, 1., (D, D)))
    else:
        transition_matrix = column_normalize(
            np.asarray(transition_matrix, dtype=float))

    logl_hist = []
    state_distr = None
    for _ in range(max_iter):
        logl = hmm_logl(obs, observation_matrix, transition_matrix, p0)
        logl_hist.append(logl)
        # E-step:
        res = hmm_stationary_inference(obs, observation_matrix,
                                       transition_matrix, p0, smooth=True)
        state_distr = res.p_smoothed
        # M-step:
        transition_matrix, observation_matrix = hmm_matrix_estim(
            state_distr, obs, n_symbols=n_symbols)
        if len(logl_hist) > 1 and logl_hist[-1] - logl_hist[-2] < tol:
            break
    return HMMEMResult(observation_matrix, transition_matrix, state_distr,
                       np.asarray(logl_hist))

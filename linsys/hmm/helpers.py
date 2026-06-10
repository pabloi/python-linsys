"""Single-step HMM operations and small utilities (port of matlab-linsys/discrHMM).

Conventions for all ``linsys.hmm`` modules
------------------------------------------
- States are integers ``0..D-1`` and observation symbols are integers
  ``0..M-1`` (MATLAB used 1-based indices for both).
- The observation (emission) matrix ``O`` has shape ``(M, D)`` and is
  **column-stochastic**: ``O[m, d] = p(y = m | x = d)``, each column sums
  to 1 (MATLAB normalizes it with ``columnNormalize``).
- The transition matrix ``T`` has shape ``(D, D)`` and is
  **column-stochastic**: ``T[i, j] = p(x[k+1] = i | x[k] = j)``, i.e. the
  *next* state indexes rows and the *previous* state indexes columns. The
  prediction step is therefore ``p_next = T @ p`` (verified against
  ``HMMpredict.m``; ``columnNormalize`` makes columns sum to 1).
- A distribution over states is a 1-D ``(D,)`` vector. Time series of
  distributions are ``(D, N)`` arrays with columns as time samples,
  following PORTING.md.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "column_normalize", "hmm_update", "hmm_back_update", "hmm_predict",
    "hmm_logl", "discretize_obs", "linear_transition_matrix",
]


def column_normalize(p):
    """Normalize columns of ``p`` to sum to 1 (MATLAB: columnNormalize).

    Works on 1-D vectors (normalizes the whole vector) and 2-D arrays
    (normalizes each column independently).
    """
    p = np.asarray(p, dtype=float)
    return p / p.sum(axis=0)


def hmm_update(prior_state_distr, obs_given_state_distr):
    """Measurement update of the discrete-state (HMM) filter (MATLAB: HMMupdate).

    Implements ``p(x_k | y_k) ∝ p(y_k | x_k) p(x_k)``.

    Parameters
    ----------
    prior_state_distr : (D,) array, p(x_k | y_{0..k-1}).
    obs_given_state_distr : (D,) array, the row ``O[m, :]`` of the
        observation matrix for the observed symbol m (likelihood per state).

    Returns
    -------
    (D,) array, the UNNORMALIZED posterior (like the MATLAB code, which
    leaves normalization to the caller since it does not affect the MAP).
    Raises ValueError if the posterior is identically zero.
    """
    updated = np.asarray(obs_given_state_distr, dtype=float) * prior_state_distr
    if updated.sum() == 0:
        raise ValueError("hmm_update: observation has p=0 for all states "
                         "with p>0. Impossible update.")
    return updated


def hmm_predict(prior_state_distr, next_given_curr_distr):
    """Prediction step of the discrete-state filter (MATLAB: HMMpredict).

    Implements ``p(x_{k+1}) = sum_j p(x_{k+1} | x_k = j) p(x_k = j)``,
    i.e. ``T @ p`` with T column-stochastic. Output is not normalized.
    """
    return np.asarray(next_given_curr_distr, dtype=float) @ prior_state_distr


def hmm_back_update(next_state_smoothed_distr, curr_state_distr,
                    next_given_curr_distr, next_state_predicted_distr):
    """Backward (smoothing) step of the discrete-state smoother
    (MATLAB: HMMbackUpdate).

    Implements
    ``p(x_k | Y) = p(x_k | y_{0..k}) * sum_i T[i, :] p(x_{k+1}=i | Y) / p(x_{k+1}=i | y_{0..k})``.

    Parameters
    ----------
    next_state_smoothed_distr : (D,) p(x_{k+1} | all data) [smoothed next state]
    curr_state_distr : (D,) p(x_k | y_{0..k}) [filtered current state]
    next_given_curr_distr : (D, D) column-stochastic transition matrix
    next_state_predicted_distr : (D,) p(x_{k+1} | y_{0..k}) [predicted next state]

    Returns the NORMALIZED smoothed distribution p(x_k | all data).
    """
    tol = 1e-15
    innov = next_state_smoothed_distr / (next_state_predicted_distr + tol)
    smoothed = curr_state_distr * (np.asarray(next_given_curr_distr).T @ innov)
    s = smoothed.sum()
    if s == 0:
        raise ValueError("hmm_back_update: inconsistent successive states "
                         "during smoothing. Impossible update.")
    return smoothed / s


def hmm_logl(observations, p_obs_given_state, p_state_given_prev,
             p_state_initial=None):
    """Log-likelihood of an observation sequence under an HMM (MATLAB: HMMlogL).

    Note: the MATLAB ``HMMlogL.m`` is a non-functional stub (it calls a
    nonexistent function and never assigns its output). This implements the
    standard scaled forward recursion:
    ``logL = sum_k log p(y_k | y_{0..k-1})``.

    Parameters
    ----------
    observations : (N,) int array of symbols in ``0..M-1`` (one per time step).
    p_obs_given_state : (M, D) column-stochastic observation matrix.
    p_state_given_prev : (D, D) column-stochastic transition matrix.
    p_state_initial : optional (D,) prior on the initial state. Uniform if
        omitted.

    Returns
    -------
    float, the log-likelihood (``-inf`` if the sequence is impossible).
    """
    O = column_normalize(p_obs_given_state)
    T = column_normalize(p_state_given_prev)
    obs = np.asarray(observations).ravel().astype(int)
    D = T.shape[0]
    if p_state_initial is None:
        p = np.ones(D) / D
    else:
        p = column_normalize(np.asarray(p_state_initial, dtype=float).ravel())
    logl = 0.0
    for m in obs:
        py = O[m, :] @ p  # p(y_k | y_{0..k-1})
        if py <= 0:
            return -np.inf
        logl += np.log(py)
        p = T @ ((O[m, :] * p) / py)  # update + predict, normalized
    return logl


def discretize_obs(observations, nbins=100, value_range=None):
    """Discretize continuous observations into bins (MATLAB: discretizeObs).

    Returns, for each observation, its bin index in ``0..nbins-1`` (the
    MATLAB version returns 1-based bins ``1..Nbins``). Values outside
    ``value_range`` are saturated to the first/last bin.

    Parameters
    ----------
    observations : array-like of continuous values.
    nbins : number of bins (default 100).
    value_range : optional (lo, hi) pair; defaults to (min, max) of the data.

    Returns an int64 array of the same shape (MATLAB returns uint8/uint16
    and errors above 65535 bins; both restrictions are dropped here).
    """
    obs = np.asarray(observations, dtype=float)
    if value_range is None or np.size(value_range) != 2:
        value_range = (np.min(obs), np.max(obs))
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi == lo:
        return np.zeros(obs.shape, dtype=np.int64)
    bins = np.ceil(nbins * (obs - lo) / (hi - lo)).astype(np.int64) - 1
    return np.clip(bins, 0, nbins - 1)


def linear_transition_matrix(n, a, q, b):
    """Input-dependent transition-matrix generator (MATLAB: linearTransitionMatrix).

    Faithful port: returns a callable ``T(u)`` with
    ``T(u)[i, j] = exp((x_i - a*x_j - b*u) / (2*sqrt(q)))`` where
    ``x = 1..n`` (kept 1-based to reproduce MATLAB values exactly; the
    difference is a constant factor that vanishes under column
    normalization).

    Note: the returned matrix is NOT normalized (callers should apply
    :func:`column_normalize`), and the MATLAB formula looks like an
    unfinished Gaussian kernel (no squaring of the residual); it is ported
    as-is.
    """
    x = np.arange(1, n + 1, dtype=float)
    return lambda u: np.exp((x[:, None] - a * x[None, :] - b * np.asarray(u))
                            / (2.0 * np.sqrt(q)))

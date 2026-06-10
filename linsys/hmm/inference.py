"""Forward-backward inference for discrete HMMs (port of matlab-linsys/discrHMM).

Ports of HMMstationaryInference, HMMstationaryInferenceAlt and
HMMnonStationaryInferenceAlt. Implemented in the Kalman-smoother style
(filter forward, then a backward correction pass); functionally equivalent
to the forward-backward algorithm.

See ``linsys.hmm.helpers`` for the matrix orientation conventions
(column-stochastic ``O`` (M, D) and ``T`` (D, D); state/observation indices
are 0-based; time runs along columns).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .helpers import column_normalize, hmm_update, hmm_predict, hmm_back_update

__all__ = [
    "HMMInferenceResult",
    "hmm_stationary_inference",
    "hmm_stationary_inference_alt",
    "hmm_nonstationary_inference_alt",
]


class HMMInferenceResult(NamedTuple):
    """Posterior state distributions; columns are time samples.

    p_predicted : (D, N+1), column k is p(x_k | y at times < k). Column 0 is
        the prior; column N is the one-step-ahead prediction past the data.
    p_updated : (D, N), column k is the filtered p(x_k | y at times <= k).
    p_smoothed : (D, N) or None, column k is p(x_k | all data). None when
        smoothing was not requested (MATLAB skips it when nargout < 3).

    Note: unlike MATLAB HMMstationaryInference (which can return unnormalized
    columns), all returned columns are normalized to sum to 1.
    """
    p_predicted: np.ndarray
    p_updated: np.ndarray
    p_smoothed: Optional[np.ndarray]


def _matrix_getter(spec, input_):
    """Return a function ``get(k) -> column-normalized matrix at time k``.

    ``spec`` may be a constant matrix or a callable of the input row
    ``input_[k]``. Callables are re-evaluated only when the input value
    changes between consecutive requests (as in the MATLAB code).
    """
    if not callable(spec):
        M = column_normalize(np.asarray(spec, dtype=float))
        return lambda k: M
    cache = {"u": None, "M": None}

    def get(k):
        u = np.atleast_1d(input_[k])
        if cache["M"] is None or not np.array_equal(cache["u"], u):
            cache["M"] = column_normalize(np.asarray(spec(input_[k]),
                                                     dtype=float))
            cache["u"] = u
        return cache["M"]

    return get


def _group_obs_by_time(observations, observation_times, n_times):
    """List of observation-symbol arrays, one per time step 0..n_times-1."""
    obs = np.asarray(observations).ravel().astype(int)
    times = np.asarray(observation_times).ravel().astype(int)
    if obs.size != times.size:
        raise ValueError("observations and observation_times must have the "
                         "same length")
    if times.size and times.min() < 0:
        raise ValueError("Observation times before initial condition!")
    if times.size and times.max() >= n_times:
        raise ValueError("Observations outside the input range")
    order = np.argsort(times, kind="stable")
    obs, times = obs[order], times[order]
    starts = np.searchsorted(times, np.arange(n_times + 1))
    return [obs[starts[i]:starts[i + 1]] for i in range(n_times)]


def _forward_backward(obs_by_time, n_times, get_obs_matrix, get_trans_matrix,
                      p_state_initial, smooth):
    """Shared engine for all three inference variants."""
    D = get_trans_matrix(0).shape[0]
    if p_state_initial is None:
        p0 = np.ones(D) / D
    else:
        p0 = np.asarray(p_state_initial, dtype=float).ravel()

    p_predicted = np.empty((D, n_times + 1))
    p_updated = np.empty((D, n_times))
    p = p0
    p_predicted[:, 0] = p
    for k in range(n_times):
        O = get_obs_matrix(k)
        for m in obs_by_time[k]:
            p = hmm_update(p, O[m, :])
        p_updated[:, k] = p
        p = hmm_predict(p, get_trans_matrix(k))
        p_predicted[:, k + 1] = p
        p = p / p.sum()  # prevent underflow of all states
    p_updated = column_normalize(p_updated)
    p_predicted = column_normalize(p_predicted)

    p_smoothed = None
    if smooth:
        p_smoothed = np.empty((D, n_times))
        p = p_updated[:, n_times - 1]
        p_smoothed[:, n_times - 1] = p
        for k in range(n_times - 2, -1, -1):
            p = hmm_back_update(p, p_updated[:, k], get_trans_matrix(k),
                                p_predicted[:, k + 1])
            p_smoothed[:, k] = p
    return HMMInferenceResult(p_predicted, p_updated, p_smoothed)


def hmm_stationary_inference(observations, p_obs_given_state,
                             p_state_given_prev, p_state_initial=None,
                             smooth=True):
    """Forward-backward inference with one observation per time step
    (MATLAB: HMMstationaryInference).

    Parameters
    ----------
    observations : (N,) int array of symbols in ``0..M-1``, one per step.
    p_obs_given_state : (M, D) observation matrix, ``O[m, d] = p(y=m | x=d)``.
        Columns are normalized to sum to 1 if they do not.
    p_state_given_prev : (D, D) transition matrix,
        ``T[i, j] = p(x[k+1]=i | x[k]=j)`` (column-stochastic).
    p_state_initial : optional (D,) initial-state prior (default uniform).
    smooth : skip the backward pass when False (p_smoothed is None).

    Returns an :class:`HMMInferenceResult`.
    """
    obs = np.asarray(observations).ravel().astype(int)
    n_times = obs.size
    return _forward_backward(
        [obs[k:k + 1] for k in range(n_times)], n_times,
        _matrix_getter(p_obs_given_state, None),
        _matrix_getter(p_state_given_prev, None),
        p_state_initial, smooth)


def hmm_stationary_inference_alt(observations, observation_times,
                                 p_obs_given_state, p_state_given_prev,
                                 p_state_initial=None, smooth=True):
    """Forward-backward inference with arbitrary observation times
    (MATLAB: HMMstationaryInferenceAlt).

    Allows zero, one or multiple observations per time step. Times are
    0-based integers (MATLAB used 1-based); the initial prior corresponds
    to time 0 and the chain runs through ``max(observation_times)``.

    Parameters are as in :func:`hmm_stationary_inference`, plus
    ``observation_times``: (N,) int array, same length as ``observations``.
    """
    times = np.asarray(observation_times).ravel().astype(int)
    if times.size == 0:
        raise ValueError("at least one observation is required")
    n_times = int(times.max()) + 1
    return _forward_backward(
        _group_obs_by_time(observations, times, n_times), n_times,
        _matrix_getter(p_obs_given_state, None),
        _matrix_getter(p_state_given_prev, None),
        p_state_initial, smooth)


def hmm_nonstationary_inference_alt(observations, observation_times, input,
                                    p_obs_given_state, p_state_given_prev,
                                    p_state_initial=None, smooth=True):
    """Forward-backward inference with input-dependent matrices
    (MATLAB: HMMnonStationaryInferenceAlt).

    ``p_obs_given_state`` and ``p_state_given_prev`` may each be either a
    constant matrix or a callable ``f(u)`` of the input at one time step,
    returning the matrix to use at that step (re-evaluated only when the
    input changes). The transition matrix evaluated at time k governs the
    transition from k to k+1.

    Parameters
    ----------
    observations, observation_times : as in
        :func:`hmm_stationary_inference_alt` (times are 0-based).
    input : (N,) or (N, nu) array; its length defines the time span (the
        chain runs over times ``0..N-1`` regardless of observation times).
    p_obs_given_state : (M, D) matrix or callable ``f(u) -> (M, D)``.
    p_state_given_prev : (D, D) matrix or callable ``f(u) -> (D, D)``.
    p_state_initial : optional (D,) initial-state prior (default uniform).
    smooth : skip the backward pass when False.

    Returns an :class:`HMMInferenceResult` with N time columns.
    """
    u = np.asarray(input)
    n_times = u.shape[0]
    return _forward_backward(
        _group_obs_by_time(observations, observation_times, n_times), n_times,
        _matrix_getter(p_obs_given_state, u),
        _matrix_getter(p_state_given_prev, u),
        p_state_initial, smooth)

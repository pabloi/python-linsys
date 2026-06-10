"""Discrete hidden Markov models (port of matlab-linsys/discrHMM).

Conventions (binding for the whole package):

- States are ``0..D-1`` and observation symbols ``0..M-1`` (0-based; MATLAB
  used 1-based indices).
- The transition matrix ``T`` (D, D) is **column-stochastic**:
  ``T[i, j] = p(x[k+1] = i | x[k] = j)`` — columns sum to 1, as enforced by
  MATLAB's ``columnNormalize`` (sums over dim 1) and used by ``HMMpredict``
  (``p_next = T @ p``).
- The observation matrix ``O`` (M, D) is column-stochastic:
  ``O[m, d] = p(y = m | x = d)``.
- Distributions over states are (D,) vectors; time series of distributions
  are (D, N) arrays with columns as time samples (see PORTING.md).
"""
from .helpers import (column_normalize, hmm_update, hmm_back_update,
                      hmm_predict, hmm_logl, discretize_obs,
                      linear_transition_matrix)
from .inference import (HMMInferenceResult, hmm_stationary_inference,
                        hmm_stationary_inference_alt,
                        hmm_nonstationary_inference_alt)
from .viterbi import ViterbiResult, viterbi
from .fit import hmm_matrix_estim, HMMEMResult, hmm_em

__all__ = [
    "column_normalize", "hmm_update", "hmm_back_update", "hmm_predict",
    "hmm_logl", "discretize_obs", "linear_transition_matrix",
    "HMMInferenceResult", "hmm_stationary_inference",
    "hmm_stationary_inference_alt", "hmm_nonstationary_inference_alt",
    "ViterbiResult", "viterbi",
    "hmm_matrix_estim", "HMMEMResult", "hmm_em",
]

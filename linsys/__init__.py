"""
python-linsys: Linear Dynamical System Identification Toolbox

A Python port of the MATLAB matlab-linsys toolbox for linear system
identification, Kalman filtering/smoothing, and EM-based estimation.
"""

from .linear_system import LinearSystem
from .kalman import (
    kalman_filter,
    kalman_smoother,
    kalman_predict,
    kalman_update,
)
from .em import em_identify, em_step
from .subspace import subspace_id
from .utils import (
    simulate,
    log_likelihood,
    hankel_matrix,
)

__version__ = "0.1.0"
__all__ = [
    "LinearSystem",
    "kalman_filter",
    "kalman_smoother",
    "kalman_predict",
    "kalman_update",
    "em_identify",
    "em_step",
    "subspace_id",
    "simulate",
    "log_likelihood",
    "hankel_matrix",
]

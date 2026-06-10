"""linsys: linear dynamical systems toolbox (Python port of matlab-linsys).

Model:
    x[k+1] = A x[k] + B u[k] + w[k],  w ~ N(0, Q)
    y[k]   = C x[k] + D u[k] + z[k],  z ~ N(0, R)

Data layout: columns of Y/U/X are time samples; covariance stacks are
(N, nx, nx). See PORTING.md.
"""
from . import utils
from . import kalman
from . import subspace
from . import spca
from . import em
from . import hmm
from . import stats
from . import model_selection
from . import misc_helpers
from . import viz
from .utils import fwd_sim, canonize
from .stats import data_log_likelihood, logl_incomplete, logl_complete

__version__ = "0.1.0"

__all__ = ["utils", "kalman", "subspace", "spca", "em", "hmm", "stats",
           "model_selection", "misc_helpers", "viz", "fwd_sim", "canonize",
           "data_log_likelihood", "logl_incomplete", "logl_complete",
           "__version__"]

"""Visualization of models and fits (port of matlab-linsys/misc/viz).

matplotlib is imported lazily inside the plotting functions (falling back
to the Agg backend when no GUI backend is available), so importing this
package does not require matplotlib. All functions return matplotlib Figure
objects and never call ``plt.show()``.

Models are plain dicts with keys 'A' (or 'J'), 'B', 'C', 'D', 'Q', 'R' and
an optional 'name' (the linsys model classes are not ported yet).

Not ported (legacy code, skipped per PORTING.md): legacy_modelCompare.m,
legacy_vizDataLikelihood.m, legacy_vizSingleModel.m,
legacy_vizSingleModelMLMC.m.
"""
from .helpers import sort_c
from .plots import (viz_data_fit, viz_data_likelihood,
                    viz_cv_data_likelihood, viz_data_res, viz_models,
                    viz_hmm_inference)

__all__ = ["sort_c", "viz_data_fit", "viz_data_likelihood",
           "viz_cv_data_likelihood", "viz_data_res", "viz_models",
           "viz_hmm_inference"]

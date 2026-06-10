"""Shared helpers for the viz package (port of matlab-linsys/misc/viz).

matplotlib is imported lazily (inside :func:`get_plt`) so that the core
package has no hard matplotlib dependency.
"""
from __future__ import annotations

import numpy as np

from ..kalman import filter_stationary, smoother_stationary
from ..utils import fwd_sim

__all__ = ["get_plt", "sort_c", "model_mats", "model_name", "one_ahead_output",
           "data_projections", "red_blue_cmap"]


def get_plt():
    """Lazy matplotlib.pyplot import, falling back to the Agg backend
    (headless-safe). Returns the pyplot module."""
    import matplotlib
    try:
        import matplotlib.pyplot as plt
    except Exception:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    return plt


def red_blue_cmap():
    """Red-white-blue colormap as defined in the MATLAB viz functions."""
    from matplotlib.colors import LinearSegmentedColormap
    n = 100
    t = np.arange(n)[:, None] / n
    ex1, ex2, mid = np.array([1., 0, 0]), np.array([0., 0, 1.]), np.ones(3)
    seg = np.vstack([ex1 * (1 - t) + mid * t, mid[None, :],
                     ex2 * t + mid * (1 - t)])
    # MATLAB applies flipud(map): blue = negative, red = positive
    return LinearSegmentedColormap.from_list("linsys_rwb", seg[::-1])


def model_mats(model):
    """Normalize a model dict: returns (A, B, C, D, Q, R) float arrays.
    Accepts 'J' as an alias for 'A' (MATLAB canonical models)."""
    m = dict(model)
    A = m.get("A", m.get("J"))
    if A is None:
        raise KeyError("model must contain 'A' (or 'J')")
    def g(k, default=None):
        v = m.get(k, default)
        return None if v is None else np.atleast_2d(np.asarray(v, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = g("C")
    return A, g("B"), C, g("D"), g("Q"), g("R")


def model_name(model, i):
    return dict(model).get("name", f"model {i + 1}")


def one_ahead_output(model, Y, U):
    """One-step-ahead (Kalman-predicted) output and the smoother result."""
    A, B, C, D, Q, R = model_mats(model)
    sres = smoother_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
    yhat = C @ sres.Xp[:, :-1] + D @ U
    return yhat, sres


def deterministic_sim(model, U, x0):
    """Noise-free forward simulation from x0. Returns (Yd, Xd)."""
    A, B, C, D, _, _ = model_mats(model)
    out, state = fwd_sim(U, A, B, C, D, x0=x0)
    return out, state


def filter_logl(model, Y, U, x0=None, P0=None):
    A, B, C, D, Q, R = model_mats(model)
    return filter_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U).logL


def data_projections(model, Y, U):
    """Least-squares projection of the data onto the model states:
    X ~ argmin || Y - D U - C X || (MATLAB: getDataProjections, simplified)."""
    A, B, C, D, _, _ = model_mats(model)
    Z = Y - D @ U
    good = ~np.isnan(Z).any(axis=0)
    X = np.full((C.shape[1], Y.shape[1]), np.nan)
    X[:, good] = np.linalg.lstsq(C, Z[:, good], rcond=None)[0]
    return X


def sort_c(ref_c, all_c):
    """Match the columns of each C in ``all_c`` to the most-similar column
    of ``ref_c`` (MATLAB: misc/viz/sortC.m).

    Similarity is the absolute cosine between columns; pairs are assigned
    greedily (best pair first, neither column can be re-selected).

    Parameters
    ----------
    ref_c : (ny, nref) reference matrix (typically the largest model's C).
    all_c : list of (ny, n_i) matrices, n_i <= nref expected.

    Returns
    -------
    list of int arrays: ``out[i][k]`` is the column of ``ref_c`` matched to
    column k of ``all_c[i]`` (0-based; MATLAB is 1-based). If a model has
    more columns than ``ref_c``, leftovers get sequential indices past
    nref - 1 (MATLAB leaves NaN there).
    """
    ref = np.real(np.asarray(ref_c, dtype=complex)).astype(float)
    nref = ref.shape[1]
    ref_norm2 = np.sum(ref ** 2, axis=0)
    out = []
    for thisC in all_c:
        thisC = np.asarray(thisC, dtype=float)
        n_this = thisC.shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = 1 - np.abs(thisC.T @ ref) / np.sqrt(
                np.outer(np.sum(thisC ** 2, axis=0), ref_norm2))
        assign = np.full(n_this, -1, dtype=int)
        for _ in range(min(n_this, nref)):
            if np.isnan(dist).all():
                break
            ii, jj = np.unravel_index(np.nanargmin(dist), dist.shape)
            assign[ii] = jj
            dist[ii, :] = np.nan
            dist[:, jj] = np.nan
        # Leftovers (model larger than reference): sequential slots
        nxt = nref
        for k in range(n_this):
            if assign[k] < 0:
                assign[k] = nxt
                nxt += 1
        out.append(assign)
    return out

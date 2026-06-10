"""EM options (port of matlab-linsys/EM/processEMopts.m)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np


def _is_nan_flag(v):
    """True if v is the 'set to zero' special flag: NaN scalar, 'zero', or 0."""
    if isinstance(v, str):
        return v.lower() == "zero"
    if np.isscalar(v):
        try:
            return bool(np.isnan(v)) or v == 0
        except TypeError:
            return False
    return False


@dataclass
class EMOpts:
    """Options for EM system identification (MATLAB: processEMopts).

    Defaults mirror the MATLAB ones, except `verbose` which defaults to False
    in Python (MATLAB default is true).

    Special values: fix_q / fix_x0 / fix_p0 accept np.nan (or the string
    'zero', or 0) meaning "fix to zero" (for fix_p0, NaN means an improper
    infinite-variance prior, as in MATLAB). fix_x0 = NaN also implies
    fix_p0 = 0 (no uncertainty on a given initial state).
    """
    Niter: int | None = None        # default 1000*nx (set in process_em_opts)
    Nreps: int = 10                 # repetitions for random_start_em
    robust_flag: bool = False
    outlier_reject: bool = False
    fast_flag: int = 1              # statKF/KS auto-select fast samples
    convergence_tol: float = 5e-3   # min logL improvement per output dim / 100 iters
    target_tol: float = 1e-3        # min relative improvement towards target / 100 iters
    target_logl: float | None = None
    diag_a: bool = False
    spherical_r: bool = False
    diag_r: bool = False
    th_r: float = 0.0               # deprecated (must be 0)
    ind_d: np.ndarray | None = None  # columns of U that feed D (default: all)
    ind_b: np.ndarray | None = None  # columns of U that feed B (default: all)
    log_flag: bool = False
    fix_a: np.ndarray | None = None
    fix_b: np.ndarray | None = None
    fix_c: np.ndarray | None = None
    fix_d: np.ndarray | None = None
    fix_q: np.ndarray | None = None   # may be NaN/'zero'/0 -> zeros(nx, nx)
    fix_r: np.ndarray | None = None
    fix_x0: np.ndarray | None = None  # may be NaN/'zero'/0 -> zeros, with fix_p0 = 0
    fix_p0: np.ndarray | None = None  # may be NaN -> diag(inf) (improper prior)
    include_output_idx: np.ndarray | None = None  # default: all outputs
    stable_a: bool = False
    min_q: float = 0.0
    min_r: float = 1e-6
    verbose: bool = False           # NOTE: MATLAB default is true
    # random_start_em refinement stage:
    refine_tol: float = 1e-4
    refine_max_iter: int = 20000
    refine_fast_flag: bool = True
    fast_refine_tol_factor: float = 10.0
    disable_refine: bool = False

    def copy(self, **kw):
        return replace(self, **kw)


def process_em_opts(opts: EMOpts | None, nu, nx, ny) -> EMOpts:
    """Fill defaults and resolve special option values (MATLAB: processEMopts).

    Returns a new EMOpts; the input is not modified. Idempotent.
    """
    o = EMOpts() if opts is None else replace(opts)

    if o.Niter is None:
        o.Niter = 1000 * nx
    if o.ind_b is None:
        o.ind_b = np.arange(nu)
    else:
        o.ind_b = np.atleast_1d(np.asarray(o.ind_b, dtype=int))
    if o.ind_d is None:
        o.ind_d = np.arange(nu)
    else:
        o.ind_d = np.atleast_1d(np.asarray(o.ind_d, dtype=int))
    if o.include_output_idx is None:
        o.include_output_idx = np.arange(ny)
    else:
        idx = np.atleast_1d(np.asarray(o.include_output_idx))
        if idx.dtype == bool:
            if idx.size != ny:
                raise ValueError("EMopts:outputIdxListSizeMismatch - provided "
                                 "logical list of output indexes is inconsistent "
                                 "with output size.")
            idx = np.where(idx)[0]
        o.include_output_idx = idx.astype(int)

    # Sanity checks on possibly conflicting options:
    if o.fix_b is not None:
        o.ind_b = np.arange(nu)
        o.fix_b = np.atleast_2d(np.asarray(o.fix_b, dtype=float))
        if o.fix_b.shape != (nx, nu):
            raise ValueError("EMopts:providedBdimMismatch - provided B matrix "
                             "size is inconsistent with number of inputs or states.")
    if o.fix_a is not None:
        o.fix_a = np.atleast_2d(np.asarray(o.fix_a, dtype=float))

    nny = o.include_output_idx.size
    if o.fix_c is not None:
        o.fix_c = np.atleast_2d(np.asarray(o.fix_c, dtype=float))
        if o.fix_c.shape[1] != nx or o.fix_c.shape[0] not in (ny, nny):
            raise ValueError("processEMopts:fixCsize - incorrect fixed C matrix size.")
        if o.fix_c.shape[0] == ny:
            o.fix_c = o.fix_c[o.include_output_idx, :]
        elif nny != ny:
            warnings.warn("processEMopts:fixCsize - fixed C appears to be "
                          "already reduced to include_output_idx; proceeding.")
    if o.fix_d is not None:
        o.ind_d = np.arange(nu)
        o.fix_d = np.atleast_2d(np.asarray(o.fix_d, dtype=float))
        if o.fix_d.shape[1] != nu or o.fix_d.shape[0] not in (ny, nny):
            raise ValueError("processEMopts:fixDsize - incorrect fixed D matrix size.")
        if o.fix_d.shape[0] == ny:
            o.fix_d = o.fix_d[o.include_output_idx, :]
        elif nny != ny:
            warnings.warn("processEMopts:fixDsize - fixed D appears to be "
                          "already reduced to include_output_idx; proceeding.")
    if o.fix_r is not None and not _is_nan_flag(o.fix_r):
        o.fix_r = np.atleast_2d(np.asarray(o.fix_r, dtype=float))
        if (o.fix_r.shape[0] != o.fix_r.shape[1]
                or o.fix_r.shape[0] not in (ny, nny)):
            raise ValueError("processEMopts:fixRsize - incorrect fixed R matrix "
                             "size, should be square and of same size as output.")
        if o.fix_r.shape[0] == ny:
            o.fix_r = o.fix_r[np.ix_(o.include_output_idx, o.include_output_idx)]
        elif nny != ny:
            warnings.warn("processEMopts:fixRsize - fixed R appears to be "
                          "already reduced to include_output_idx; proceeding.")

    # Reinterpret special options:
    if o.fix_q is not None:
        if _is_nan_flag(o.fix_q):
            o.fix_q = np.zeros((nx, nx))
        else:
            o.fix_q = np.atleast_2d(np.asarray(o.fix_q, dtype=float))
    if o.fix_x0 is not None:
        if _is_nan_flag(o.fix_x0):
            o.fix_x0 = np.zeros(nx)
            o.fix_p0 = np.zeros((nx, nx))  # no uncertainty if initial state given
        else:
            o.fix_x0 = np.asarray(o.fix_x0, dtype=float).ravel()
    if o.fix_p0 is not None and not isinstance(o.fix_p0, np.ndarray):
        if np.isscalar(o.fix_p0) and np.isnan(o.fix_p0):
            o.fix_p0 = np.diag(np.full(nx, np.inf))  # max uncertainty
    if o.fix_p0 is not None:
        o.fix_p0 = np.atleast_2d(np.asarray(o.fix_p0, dtype=float))

    if o.th_r != 0:
        raise ValueError("EMopts: thR option was deprecated, must be 0.")
    return o

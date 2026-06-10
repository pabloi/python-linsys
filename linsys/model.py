"""Object-oriented model classes (port of matlab-linsys/linsys).

MATLAB -> Python class map:

- ``linsys.m``         -> :class:`LinSys`
- ``fittedLinsys.m``   -> :class:`FittedLinSys`
- ``dset.m``           -> :class:`DataSet`
- ``dataFit.m``        -> :class:`DataFit`
- ``stateEstimate.m``  -> :class:`StateEstimate`
- ``initCond.m``       -> :class:`InitCond`
- ``trainInfo.m``      -> :class:`TrainInfo`

Conventions (PORTING.md): columns are time samples (Y: (ny, N), U: (nu, N),
X: (nx, N)); covariance stacks are (N, nx, nx); lists replace MATLAB cell
arrays (multiple realizations of one system); NaN marks missing samples.

Notable deviations from MATLAB (besides snake_case):

- The :class:`LinSys` constructor takes matrices in the order
  ``A, B, C, D, Q, R`` (MATLAB ``linsys(A, C, R, B, D, Q)``).
- Methods that modify model parameters (``canonize``, ``transform``,
  ``scale``, ``shift_states``, ``em_refine``, ...) return new *plain*
  ``LinSys`` objects (in MATLAB, value semantics return the same class,
  which leaves stale fit metadata on ``fittedLinsys`` objects).
- ``DataFit`` stores the initial condition it was given (the MATLAB class
  declares ``initialCondition`` immutable and never assigns the constructor
  argument, so it is always the default - a bug, see module report).
- Random draws accept an ``rng`` argument (MATLAB uses the global stream).
- ``LinSys.id`` returns only the fitted model(s); the MATLAB second output
  (``outlog``) is available as ``FittedLinSys.training_log``.

Not ported (plot-only methods without a viz equivalent):

- ``linsys.vizSingleFit`` / ``linsys.vizSingleRes`` (hard-coded 12x15 image
  reshapes, specific to one dataset of the original paper),
- ``linsys.compareResiduals`` (bar plot) and ``fittedLinsys.compare``
  (figure of LRT/AIC/BIC bars; the numeric pieces are available via
  ``likelihood_ratio_test``/``BIC``/``AIC``/``AICc``),
- ``linsys.comparisonTable`` (its MATLAB body is entirely commented out).

``linsys.upsample``/``downsample`` raise ``NotImplementedError`` (they
``error('unimplemented')`` in MATLAB too). ``linsys.SSid`` methods 'SSEM'
(subspaceEMhybrid.m does not exist in the MATLAB repo) and 'subid'
(third-party ext/ code, excluded per PORTING.md) are not supported.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .utils import canonize as _canonize_fn
from .utils import fwd_sim
from .utils import transform as _transform_fn
from .kalman import (KalmanOpts, filter_stationary, reduce_model,
                     smoother_stationary)
from .stats import data_log_likelihood
from .model_selection import bic_aic as _bic_aic
from .model_selection import fold_split

__all__ = ["LinSys", "FittedLinSys", "DataSet", "DataFit", "StateEstimate",
           "InitCond", "TrainInfo", "KFilterResult", "KSmoothResult",
           "fit_linsys"]


def _as2d(M):
    return np.atleast_2d(np.asarray(M, dtype=float))


def _md5(*arrays):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# TrainInfo (trainInfo.m)
# ---------------------------------------------------------------------------
@dataclass
class TrainInfo:
    """Metadata of a model-fitting run (MATLAB: trainInfo). All optional."""
    set_hash: str = ""
    method: str = ""
    options: object = None


# ---------------------------------------------------------------------------
# StateEstimate (stateEstimate.m)
# ---------------------------------------------------------------------------
class StateEstimate:
    """Gaussian state-trajectory estimate (MATLAB: stateEstimate).

    state: (nx, N) means; covar: (N, nx, nx) covariances (a single (nx, nx)
    matrix is broadcast to all samples); lag_one_covar: (N-1, nx, nx)
    cov(x[k+1], x[k]), optional. Lists of arrays represent multiple
    realizations (MATLAB cells).
    """

    def __init__(self, state=None, covar=None, lag_one_covar=None):
        if isinstance(state, list):  # multiple realizations
            self.state = [_as2d(x) for x in state]
            self.covar = (None if covar is None
                          else [np.asarray(p, dtype=float) for p in covar])
            self.lag_one_covar = (None if lag_one_covar is None else
                                  [np.asarray(p, dtype=float)
                                   for p in lag_one_covar])
            return
        self.state = None if state is None else _as2d(state)
        self.covar = None
        self.lag_one_covar = None
        if covar is not None:
            P = np.asarray(covar, dtype=float)
            if P.ndim == 2:
                if P.shape[0] != P.shape[1]:
                    raise ValueError("stateEstimate:constructor - uncertainty "
                                     "matrix is not square")
                P = np.broadcast_to(P, (self.state.shape[1],) + P.shape).copy()
            if P.shape[1] != self.state.shape[0]:
                raise ValueError("stateEstimate:constructor - inconsistent "
                                 "state and uncertainty sizes")
            if P.shape[1] != P.shape[2]:
                raise ValueError("stateEstimate:constructor - uncertainty "
                                 "matrix is not square")
            if P.shape[0] != self.state.shape[1]:
                raise ValueError("stateEstimate:inconsistentCovarSize - "
                                 "covariance stack does not have the same "
                                 "number of samples as the state")
            # PSD check. NOTE: the MATLAB check (`any(diag(D))<0`) compares a
            # logical with 0 and can never fire; the intended check is done
            # here (finite samples only; warning, not an error).
            finite = np.isfinite(P).all(axis=(1, 2))
            if finite.any():
                try:
                    w = np.linalg.eigvalsh((P[finite]
                                            + P[finite].transpose(0, 2, 1)) / 2)
                    tol = 1e-10 * max(1.0, float(np.abs(w).max()))
                    if (w < -tol).any():
                        warnings.warn("stateEstimate:constructor - "
                                      "uncertainty matrix is not PSD")
                except np.linalg.LinAlgError:
                    pass
            self.covar = P
        if lag_one_covar is not None:
            Pt = np.asarray(lag_one_covar, dtype=float)
            if self.covar is not None and Pt.shape[0] != self.covar.shape[0] - 1:
                raise ValueError("stateEstimate:constructor - inconsistent "
                                 "covariance matrix sizes")
            self.lag_one_covar = Pt

    # -- properties -------------------------------------------------------
    @property
    def is_multiple(self):
        return isinstance(self.state, list)

    @property
    def nsamp(self):
        """Number of time samples (MATLAB: Nsamp)."""
        if self.is_multiple:
            return np.array([x.shape[1] for x in self.state])
        return self.state.shape[1]

    @property
    def order(self):
        if self.is_multiple:
            return self.state[0].shape[0]
        return self.state.shape[0]

    # -- methods ----------------------------------------------------------
    def get_sample(self, k) -> "InitCond":
        """Single sample as an initial condition (MATLAB: getSample;
        0-based index here)."""
        return InitCond(self.state[:, k], self.covar[k])

    def extract_single(self, i) -> "StateEstimate":
        """Extract realization i of a multiple estimate (MATLAB: extractSingle)."""
        if not self.is_multiple:
            raise ValueError("stateEstim object is not multiple, cannot "
                             "extract a single set")
        if i >= len(self.state):
            raise IndexError(f"stateEstim:extractSingle - index {i} larger "
                             f"than number of estimates ({len(self.state)})")
        return StateEstimate(self.state[i],
                             None if self.covar is None else self.covar[i])

    def marginalize(self, idx) -> "StateEstimate":
        """Marginal estimate of a subset of states (MATLAB: marginalize;
        0-based index/indices here)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        x = self.state[idx, :]
        P = None if self.covar is None else self.covar[:, idx][:, :, idx]
        Pt = None if self.lag_one_covar is None else \
            self.lag_one_covar[:, idx][:, :, idx]
        return StateEstimate(x, P, Pt)

    def plot(self, offset=0, prc=99.7, ax=None):
        """Plot state means with shaded CI bands (MATLAB: stateEstimate.plot).

        Returns the matplotlib Axes used."""
        from .viz.helpers import get_plt
        from scipy.stats import norm
        plt = get_plt()
        if not 0 <= prc <= 100:
            raise ValueError("prc must be a number between 0 and 100")
        if ax is None:
            ax = plt.gca()
        f = norm.ppf(1 - 0.5 * (1 - prc / 100.0))
        x = offset + np.arange(1, self.nsamp + 1)
        for i in range(self.order):
            y = self.state[i, :]
            line, = ax.plot(x, y, linewidth=2)
            if self.covar is not None:
                e = f * np.sqrt(self.covar[:, i, i])
                ax.fill_between(x, y - e, y + e, color=line.get_color(),
                                alpha=0.5, edgecolor="none", zorder=0)
        return ax


# ---------------------------------------------------------------------------
# InitCond (initCond.m)
# ---------------------------------------------------------------------------
class InitCond(StateEstimate):
    """Gaussian belief over the initial state x[0] (MATLAB: initCond).

    ``InitCond()`` (no arguments / empty x) represents the improper flat
    prior: ``state``/``covar`` are None and the Kalman functions use their
    defaults (x0 = 0, P0 = inf*I). With x given and P omitted, P defaults to
    the same improper inf*I. ``state`` is stored 1-D (nx,) per PORTING.md.
    Lists represent per-realization initial conditions (MATLAB cells).
    """

    def __init__(self, x=None, P=None):  # noqa: super().__init__ not used
        if isinstance(x, list):
            self.state = [None if xi is None
                          else np.asarray(xi, dtype=float).ravel() for xi in x]
            if P is None:
                P = [None] * len(self.state)
            self.covar = [None if p is None else _as2d(p) for p in P]
            self.lag_one_covar = None
            return
        self.lag_one_covar = None
        if x is None or np.size(x) == 0:
            self.state = None
            self.covar = None
            return
        x = np.asarray(x, dtype=float)
        if x.ndim > 1 and x.shape[1] > 1:
            raise ValueError("initCond:construct - initial condition "
                             "estimates must represent a single time-sample")
        self.state = x.ravel()
        if P is None:
            P = np.diag(np.full(self.state.size, np.inf))
        self.covar = _as2d(P)

    @property
    def nsamp(self):
        return 1

    @property
    def order(self):
        if self.is_multiple:
            s = self.state[0]
            return 0 if s is None else s.size
        return 0 if self.state is None else self.state.size

    def extract_single(self, i) -> "InitCond":
        if not self.is_multiple:
            raise ValueError("initCond is not multiple, cannot extract a "
                             "single set")
        return InitCond(self.state[i], self.covar[i])


# ---------------------------------------------------------------------------
# DataSet (dset.m)
# ---------------------------------------------------------------------------
class DataSet:
    """Input/output dataset for modeling (MATLAB: dset).

    in_: (nu, N) input; out: (ny, N) output (columns are time samples; NaN
    columns of out mark missing samples). Lists make a "multiple" dataset
    (several realizations); if only one of in_/out is a list, the other is
    replicated.
    """

    def __init__(self, in_, out):
        if isinstance(in_, list):
            self.in_ = [_as2d(u) for u in in_]
            if isinstance(out, list):
                self.out = [_as2d(y) for y in out]
            else:
                self.out = [_as2d(out) for _ in in_]
        elif isinstance(out, list):
            self.out = [_as2d(y) for y in out]
            self.in_ = [_as2d(in_) for _ in out]
        else:
            self.in_ = _as2d(in_)
            self.out = _as2d(out)
        if not self.is_multiple:
            if self.in_.shape[1] != self.out.shape[1]:
                raise ValueError("dset:constructor - inconsistent input and "
                                 "output sample sizes")
        else:
            if len(self.in_) != len(self.out):
                raise ValueError("dset:constructor - inconsistent number of "
                                 "inputs and outputs")
            for u, y in zip(self.in_, self.out):
                if u.shape[1] != y.shape[1]:
                    raise ValueError("dset:constructor - inconsistent input "
                                     "and output sample sizes")

    # -- properties -------------------------------------------------------
    @property
    def is_multiple(self):
        return isinstance(self.in_, list)

    @property
    def ninputs(self):
        if self.is_multiple:
            return np.array([u.shape[0] for u in self.in_])
        return self.in_.shape[0]

    @property
    def noutputs(self):
        if self.is_multiple:
            return np.array([y.shape[0] for y in self.out])
        return self.out.shape[0]

    @property
    def nsamp(self):
        if self.is_multiple:
            return np.array([u.shape[1] for u in self.in_])
        return self.in_.shape[1]

    @property
    def non_nan_samp(self):
        """Number of samples with no NaN output (MATLAB: nonNaNSamp)."""
        if self.is_multiple:
            return np.array([int(np.sum(~np.isnan(y).any(axis=0)))
                             for y in self.out])
        return int(np.sum(~np.isnan(self.out).any(axis=0)))

    @property
    def hash(self):
        """MD5 hash of the data (MATLAB uses the GetMD5 MEX; hashlib here)."""
        if self.is_multiple:
            return _md5(np.hstack(self.in_), np.hstack(self.out))
        return _md5(np.vstack([self.in_, self.out]))

    # -- methods ----------------------------------------------------------
    def _check_single(self, what):
        if self.is_multiple:
            raise NotImplementedError(f"DataSet.{what}: unimplemented for "
                                      "multiple datasets (as in MATLAB)")

    def get_data_projections(self, model):
        """Project output data onto the model states (MATLAB:
        getDataProjections). Returns (res, res_ls): pinv(C)-projection and
        the R-weighted least-squares projection."""
        self._check_single("get_data_projections")
        yd = self.out - model.D @ self.in_
        res = np.linalg.lstsq(model.C, yd, rcond=None)[0]
        CtRinvC, _, CtRinvY, _, _, _ = reduce_model(model.C, model.R, yd)
        res_ls = np.linalg.solve(CtRinvC, CtRinvY)
        return res, res_ls

    def reduce(self, exclude_idx) -> "DataSet":
        """Drop the given output rows (MATLAB: dset.reduce)."""
        self._check_single("reduce")
        exclude_idx = np.atleast_1d(exclude_idx)
        if exclude_idx.dtype == bool:
            exclude_idx = np.where(exclude_idx)[0]
        return DataSet(self.in_, np.delete(self.out, exclude_idx, axis=0))

    def split(self, breaks, return_as_multi_set=False):
        """Split along time at the given break points (MATLAB: dset.split).

        ``breaks`` lists the first (0-based) sample of each sub-set; the
        first set is presumed to start at 0 and the last to end at nsamp.
        Returns a list of DataSet (or one multiple DataSet)."""
        self._check_single("split")
        breaks = list(np.atleast_1d(np.asarray(breaks, dtype=int)))
        if breaks[0] != 0:
            breaks = [0] + breaks
        if breaks[-1] != self.nsamp:
            breaks = breaks + [self.nsamp]
        new_in = [self.in_[:, b0:b1] for b0, b1 in zip(breaks[:-1], breaks[1:])]
        new_out = [self.out[:, b0:b1] for b0, b1 in zip(breaks[:-1], breaks[1:])]
        if return_as_multi_set:
            return DataSet(new_in, new_out)
        return [DataSet(u, y) for u, y in zip(new_in, new_out)]

    def block_split(self, block_size, n_partitions):
        """Split into n_partitions by alternating blocks of block_size
        (MATLAB: blockSplit). Unassigned blocks become NaN; the trailing
        incomplete block is discarded; leading/trailing all-NaN samples of
        each partition are trimmed. Returns a list of DataSet."""
        self._check_single("block_split")
        n_blocks = self.nsamp // block_size
        last = n_blocks * block_size
        base_out = self.out.copy()
        base_out[:, last:] = np.nan
        multi = []
        for j in range(n_partitions):
            out_j = base_out.copy()
            for i in range(n_blocks):
                if i % n_partitions != j:
                    out_j[:, i * block_size:(i + 1) * block_size] = np.nan
            ok = ~np.isnan(out_j).all(axis=0)
            first, lastk = np.argmax(ok), len(ok) - np.argmax(ok[::-1])
            multi.append(DataSet(self.in_[:, first:lastk],
                                 out_j[:, first:lastk]))
        return multi

    def alternate(self, n):
        """n interleaved folds: fold i keeps samples i, i+n, ... and NaNs the
        rest (MATLAB: dset.alternate, via foldSplit)."""
        self._check_single("alternate")
        return [DataSet(self.in_, y) for y in fold_split(self.out, n, axis=1)]

    def extract_single(self, i) -> "DataSet":
        if not self.is_multiple:
            raise ValueError("dset object is not multiple, cannot extract a "
                             "single set")
        if i >= len(self.out):
            raise IndexError(f"dset:extractSingle - index {i} larger than "
                             f"number of dsets ({len(self.out)})")
        return DataSet(self.in_[i], self.out[i])

    def logL(self, mdl, init_c=None):
        """Log-likelihood of this data under one model (or list of models)."""
        if isinstance(mdl, (list, tuple)):
            return np.array([self.logL(m, init_c) for m in mdl])
        return mdl.logL(self, init_c)

    def flat_residuals(self):
        """Residuals of the static (flat) model (MATLAB: flatResiduals)."""
        self._check_single("flat_residuals")
        from .misc_helpers import get_flat_model
        fm = get_flat_model(self.out, self.in_)
        return self.out - fm.D @ self.in_

    def estimate_var(self):
        """Covariance estimate from consecutive sample differences at
        constant input; approximates R + 0.5*C*Q*C' near steady state
        (MATLAB: estimateVar)."""
        self._check_single("estimate_var")
        diff_u = np.diff(self.in_, axis=1)
        diff_y = np.diff(self.out, axis=1)
        diff_y = diff_y[:, (diff_u == 0).all(axis=0)]
        return 0.5 * (diff_y @ diff_y.T) / diff_y.shape[1]

    # -- plotting (delegates to linsys.viz) --------------------------------
    def viz_fit(self, models):
        """(MATLAB: dset.vizFit -> vizDataFit)"""
        self._check_single("viz_fit")
        from .viz import viz_data_fit
        return viz_data_fit(_as_model_dicts(models), self.out, self.in_)

    def viz_res(self, models):
        """(MATLAB: dset.vizRes -> vizDataRes)"""
        self._check_single("viz_res")
        from .viz import viz_data_res
        return viz_data_res(_as_model_dicts(models), self.out, self.in_)

    def compare_models(self, models):
        """(MATLAB: dset.compareModels -> vizDataLikelihood)"""
        self._check_single("compare_models")
        from .viz import viz_data_likelihood
        return viz_data_likelihood(_as_model_dicts(models),
                                   (self.out, self.in_))


def _as_model_dicts(models):
    if isinstance(models, LinSys) or isinstance(models, dict):
        models = [models]
    return [m.to_dict() if isinstance(m, LinSys) else m for m in models]


# ---------------------------------------------------------------------------
# results of the Kalman wrappers
# ---------------------------------------------------------------------------
class KFilterResult(NamedTuple):
    """Result of LinSys.kfilter (lists for multiple datasets)."""
    filtered: "StateEstimate | list"
    one_ahead: "StateEstimate | list"
    rejected: "np.ndarray | list"
    logL: "float | np.ndarray"


class KSmoothResult(NamedTuple):
    """Result of LinSys.ksmooth (lists for multiple datasets)."""
    smoothed: "StateEstimate | list"
    filtered: "StateEstimate | list"
    one_ahead: "StateEstimate | list"
    rejected: "np.ndarray | list"
    logL: "float | np.ndarray"


# ---------------------------------------------------------------------------
# LinSys (linsys.m)
# ---------------------------------------------------------------------------
class LinSys:
    """Linear time-invariant state-space model (MATLAB: linsys).

        x[k+1] = A x[k] + B u[k] + w[k],  w ~ N(0, Q)
        y[k]   = C x[k] + D u[k] + z[k],  z ~ N(0, R)

    NOTE: the constructor argument order is A, B, C, D, Q, R (the MATLAB
    constructor is ``linsys(A, C, R, B, D, Q)``).
    """

    def __init__(self, A, B, C, D, Q, R, name=""):
        A, B, C = _as2d(A), _as2d(B), _as2d(C)
        D, Q, R = _as2d(D), _as2d(Q), _as2d(R)
        if A.shape[0] != A.shape[1]:
            raise ValueError("A matrix is not square")
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q matrix is not square")
        if R.shape[0] != R.shape[1]:
            raise ValueError("R matrix is not square")
        if B.shape[0] != A.shape[0]:
            raise ValueError("A and B have inconsistent sizes")
        if C.shape[0] != D.shape[0]:
            raise ValueError("Inconsistent C and D")
        if C.shape[0] != R.shape[0]:
            raise ValueError("Inconsistent C and R")
        if Q.shape[0] != A.shape[0]:
            raise ValueError("A and Q have inconsistent sizes")
        if B.shape[1] != D.shape[1]:
            raise ValueError("Inconsistent input sizes")
        self.A, self.B, self.C, self.D, self.Q, self.R = A, B, C, D, Q, R
        self.name = name

    def __repr__(self):
        nm = f", name={self.name!r}" if self.name else ""
        return (f"{type(self).__name__}(order={self.order}, "
                f"ninputs={self.ninputs}, noutputs={self.noutputs}{nm})")

    # -- properties -------------------------------------------------------
    @property
    def order(self):
        """Model order = number of states (MATLAB: order)."""
        return self.A.shape[0]

    nx = order  # alias

    @property
    def ninputs(self):
        """(MATLAB: Ninput)"""
        return self.B.shape[1]

    @property
    def noutputs(self):
        """(MATLAB: Noutput)"""
        return self.C.shape[0]

    @property
    def eigenvalues(self):
        """Eigenvalues of A."""
        return np.linalg.eigvals(self.A)

    @property
    def time_constants(self):
        """Time constants tau = -1 / log|eig(A)| (in samples)."""
        with np.errstate(divide="ignore"):
            return -1.0 / np.log(np.abs(self.eigenvalues))

    @property
    def hash(self):
        """MD5 hash of all parameters (MATLAB uses the GetMD5 MEX)."""
        return _md5(np.block([[self.A, self.B, self.Q, self.C.T],
                              [self.C, self.D, self.C, self.R]]))

    def _with_params(self, A=None, B=None, C=None, D=None, Q=None, R=None):
        """New plain LinSys with some parameters replaced (keeps name)."""
        return LinSys(self.A if A is None else A, self.B if B is None else B,
                      self.C if C is None else C, self.D if D is None else D,
                      self.Q if Q is None else Q, self.R if R is None else R,
                      name=self.name)

    # -- model variants ----------------------------------------------------
    def deterministic(self) -> "LinSys":
        """Copy with Q = 0 (states evolve deterministically)."""
        return self._with_params(Q=np.zeros_like(self.Q))

    def noiseless(self) -> "LinSys":
        """Copy with R = 0 (no observation noise)."""
        return self._with_params(R=np.zeros_like(self.R))

    # -- fitting / filtering ------------------------------------------------
    def fit(self, dat_set, init_c=None, method=None) -> "DataFit":
        """Fit state estimates to a dataset (MATLAB: linsys.fit)."""
        return DataFit(self, dat_set, method, init_c)

    def kfilter(self, dat_set, init_c=None, opts=None) -> KFilterResult:
        """Kalman-filter a dataset (MATLAB: Kfilter; wraps
        kalman.filter_stationary). ``dat_set`` may be a DataSet, a multiple
        DataSet or a list of DataSet (results become lists)."""
        if isinstance(dat_set, list):
            ics = init_c if init_c is not None else [None] * len(dat_set)
            res = [self.kfilter(d, ic, opts) for d, ic in zip(dat_set, ics)]
            f, o, rej, ll = zip(*res)
            return KFilterResult(list(f), list(o), list(rej), np.asarray(ll))
        if dat_set.is_multiple:
            res = []
            for i in range(len(dat_set.out)):
                ic = InitCond() if init_c is None else \
                    (init_c.extract_single(i) if init_c.is_multiple else init_c)
                res.append(self.kfilter(dat_set.extract_single(i), ic, opts))
            f, o, rej, ll = zip(*res)
            return KFilterResult(list(f), list(o), list(rej), np.asarray(ll))
        if init_c is None:
            init_c = InitCond()
        r = filter_stationary(dat_set.out, self.A, self.C, self.Q, self.R,
                              x0=init_c.state, P0=init_c.covar, B=self.B,
                              D=self.D, U=dat_set.in_, opts=opts)
        return KFilterResult(StateEstimate(r.X, r.P),
                             StateEstimate(r.Xp, r.Pp), r.rejected, r.logL)

    def ksmooth(self, dat_set, init_c=None, opts=None) -> KSmoothResult:
        """Kalman-smooth a dataset (MATLAB: Ksmooth; wraps
        kalman.smoother_stationary)."""
        if isinstance(dat_set, list):
            ics = init_c if init_c is not None else [None] * len(dat_set)
            res = [self.ksmooth(d, ic, opts) for d, ic in zip(dat_set, ics)]
            s, f, o, rej, ll = zip(*res)
            return KSmoothResult(list(s), list(f), list(o), list(rej),
                                 np.asarray(ll))
        if dat_set.is_multiple:
            res = []
            for i in range(len(dat_set.out)):
                ic = InitCond() if init_c is None else \
                    (init_c.extract_single(i) if init_c.is_multiple else init_c)
                res.append(self.ksmooth(dat_set.extract_single(i), ic, opts))
            s, f, o, rej, ll = zip(*res)
            return KSmoothResult(list(s), list(f), list(o), list(rej),
                                 np.asarray(ll))
        if init_c is None:
            init_c = InitCond()  # improper prior: can be problematic
        r = smoother_stationary(dat_set.out, self.A, self.C, self.Q, self.R,
                                x0=init_c.state, P0=init_c.covar, B=self.B,
                                D=self.D, U=dat_set.in_, opts=opts)
        return KSmoothResult(StateEstimate(r.Xs, r.Ps, r.Pt),
                             StateEstimate(r.Xf, r.Pf),
                             StateEstimate(r.Xp, r.Pp), r.rejected, r.logL)

    # -- prediction / simulation --------------------------------------------
    def predict(self, state_e: StateEstimate, in_=None) -> StateEstimate:
        """Propagate every sample of a state estimate len(in_) steps into the
        future, all with the same input series (MATLAB: predict)."""
        if in_ is None:
            in_ = np.zeros((self.ninputs, 1))
        in_ = _as2d(in_)
        x = state_e.state.copy()
        P = state_e.covar.copy()
        for i in range(in_.shape[1]):
            x = self.A @ x + (self.B @ in_[:, i])[:, None]
            P = np.einsum("ij,njk,lk->nil", self.A, P, self.A) + self.Q
        return StateEstimate(x, P)

    def predict2(self, state_e: StateEstimate, in_, M) -> StateEstimate:
        """Predict each point of a consecutive state series M samples ahead
        using a single input series temporally aligned with the states
        (MATLAB: predict2).

        Sample j is propagated with inputs u[j], ..., u[j+M-1]. NOTE: the
        MATLAB source uses u[j+1], ..., u[j+M] (one sample late, despite its
        own "temporally aligned" comment), which makes NaheadOutput's
        "M-ahead" prediction at time k depend on u[k]; this port uses the
        correct alignment."""
        in_ = _as2d(in_)
        ns = state_e.nsamp
        nx = self.order
        x = np.zeros((nx, ns))
        P = np.zeros((ns, nx, nx))
        for j in range(min(in_.shape[1] - M + 1, ns)):
            xj = state_e.state[:, j].copy()
            Pj = state_e.covar[j].copy()
            for i in range(M):
                xj = self.A @ xj + self.B @ in_[:, i + j]
                Pj = self.A @ Pj @ self.A.T + self.Q
            x[:, j] = xj
            P[j] = Pj
        return StateEstimate(x, P)

    def simulate(self, in_, init_c=None, deterministic_flag=False,
                 noiseless_flag=False, rng=None):
        """Simulate a realization of the system (MATLAB: simulate; wraps
        utils.fwd_sim). Uncertainty of the initial condition is ignored.

        deterministic_flag: Q = 0 (no process noise); noiseless_flag: R = 0.
        ``rng`` seeds the noise (deviation: MATLAB uses the global stream).
        Returns (DataSet, StateEstimate) - the state has N+1 samples."""
        Q = np.zeros_like(self.Q) if deterministic_flag else self.Q
        R = np.zeros_like(self.R) if noiseless_flag else self.R
        x0 = None if init_c is None else init_c.state
        out, state = fwd_sim(in_, self.A, self.B, self.C, self.D, x0=x0,
                             Q=Q, R=R, rng=rng)
        return DataSet(in_, out), StateEstimate(state,
                                                np.zeros((self.order,
                                                          self.order)))

    # -- likelihood / residuals ----------------------------------------------
    def logL(self, dat_set, init_c=None):
        """Exact log-likelihood of a dataset under this model (MATLAB:
        linsys.logL; wraps stats.data_log_likelihood).

        NOTE: this is the TOTAL logL (the MATLAB comment claims "per sample,
        per dim" but dataLogLikelihood returns the total there too)."""
        if init_c is None:
            init_c = InitCond()
        return data_log_likelihood(dat_set.out, dat_set.in_, self.A, self.B,
                                   self.C, self.D, self.Q, self.R,
                                   init_c.state, init_c.covar, "exact")

    def residual(self, dat_set, method="det", init_c=None):
        """Output residuals of the model on a dataset (MATLAB: residual).

        method 'det': residual of the deterministic (Q = 0, R = 0)
        simulation; 'oneAhead': one-step-ahead prediction residuals. If no
        initial condition is given, an appropriate one is estimated (MLE)."""
        model = self
        if init_c is None:
            if method == "oneAhead":
                dfit = self.fit(dat_set, None, "KS")
            elif method == "det":
                # Q=0 makes the initial condition the MLE one for a
                # deterministic system (KF used: KS struggles with Q=0).
                model = self.deterministic()
                dfit = model.fit(dat_set, None, "KF")
            else:
                raise ValueError(f"residual: unknown method '{method}'")
            init_c = dfit.state_estim.get_sample(0)
        dfit = model.fit(dat_set, init_c, "KF")
        if method == "oneAhead":
            return dfit.one_ahead_residual
        elif method == "det":
            return dfit.deterministic_residual
        raise ValueError(f"residual: unknown method '{method}'")

    # -- transformations -----------------------------------------------------
    def canonize(self, method=None):
        """Transform to a canonical representation (MATLAB: canonize; wraps
        utils.canonize). Returns (new LinSys, V)."""
        method = "canonical" if method is None else method
        A, B, C, _, V, Q, _ = _canonize_fn(self.A, self.B, self.C, None,
                                           self.Q, None, method=method)
        return self._with_params(A=A, B=B, C=C, Q=Q), V

    def transform(self, V) -> "LinSys":
        """Similarity transform x' = V x (MATLAB: transform; wraps
        utils.transform). Returns a new LinSys."""
        A, B, C, Q, _, _ = _transform_fn(np.asarray(V, dtype=float), self.A,
                                         self.B, self.C, self.Q)
        return self._with_params(A=A, B=B, C=C, Q=Q)

    def scale(self, k) -> "LinSys":
        """Scale states by k (scalar or per-state vector) (MATLAB: scale)."""
        k = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        if k.size == 1:
            k = np.full(self.order, k[0])
        elif k.size != self.order:
            raise ValueError("scale: k must be scalar or have one entry per "
                             "state")
        return self.transform(np.diag(k))

    def shift_states(self, K) -> "LinSys":
        """Accommodate shifted states x' = x + K u (MATLAB: shiftStates)."""
        K = _as2d(K)
        I = np.eye(self.order)
        return self._with_params(B=self.B + (I - self.A) @ K,
                                 D=self.D - self.C @ K)

    def mle_shift(self, dat_set):
        """MLE state shift to accommodate a dataset (MATLAB: mleShift).
        Returns (shifted LinSys, K). Columns of B that are all zero keep a
        zero shift. Uses scipy.optimize.minimize (MATLAB: fminunc)."""
        from scipy.optimize import minimize
        nx, nu = self.order, self.ninputs
        zero_cols = np.all(self.B == 0, axis=0)
        m = nu - int(zero_cols.sum())

        def neg_logl(kvec):
            K = np.zeros((nx, nu))
            K[:, ~zero_cols] = kvec.reshape(nx, m)
            return -self.shift_states(K).logL(dat_set)

        res = minimize(neg_logl, np.zeros(nx * m), method="BFGS")
        K = np.zeros((nx, nu))
        K[:, ~zero_cols] = res.x.reshape(nx, m)
        return self.shift_states(K), K

    def em_refine(self, dat_set) -> "LinSys":
        """Refine parameters by EM starting from this model's smoothed states
        (MATLAB: EMrefine). Returns a new (plain) LinSys."""
        from .em import EMOpts, em
        ks = self.ksmooth(dat_set)
        sm = ks.smoothed
        opts = EMOpts(ind_b=np.where(~np.all(self.B == 0, axis=0))[0],
                      ind_d=np.where(~np.all(self.D == 0, axis=0))[0],
                      Niter=1000, convergence_tol=1e-5, fast_flag=0)
        if isinstance(self, FittedLinSys) and self.train_options is not None:
            opts.include_output_idx = getattr(self.train_options,
                                              "include_output_idx", None)
        if isinstance(sm, list):
            st = [s.state for s in sm]
            P = [s.covar for s in sm]
            Pt = [s.lag_one_covar for s in sm]
        else:
            st, P, Pt = sm.state, sm.covar, sm.lag_one_covar
        r = em(dat_set.out, dat_set.in_, st, opts, P_guess=P, Pt_guess=Pt)
        return LinSys(r.A, r.B, r.C, r.D, r.Q, r.R, name=self.name)

    def upsample(self, n_factor):
        raise NotImplementedError("unimplemented (also in MATLAB)")

    def downsample(self, n_factor):
        raise NotImplementedError("unimplemented (also in MATLAB)")

    # -- output-space surgery -------------------------------------------------
    def pad(self, pad_idx, Dpad=None, Cpad=None, Rpad=None) -> "LinSys":
        """Expand the output by padding C, D (default 0) and R (default
        infinite variance) at the given new-output indices (MATLAB: pad)."""
        pad_idx = np.atleast_1d(pad_idx)
        if pad_idx.dtype == bool:
            pad_idx = np.where(pad_idx)[0]
        pad_idx = pad_idx.astype(int)
        if len(np.unique(pad_idx)) != len(pad_idx):
            raise ValueError("pad: pad_idx must not contain repeats")
        n_new = len(pad_idx)
        new_size = n_new + self.noutputs
        if pad_idx.min() < 0 or pad_idx.max() >= new_size:
            raise ValueError("pad: pad_idx out of range")
        Cpad = np.zeros((n_new, self.order)) if Cpad is None else _as2d(Cpad)
        Dpad = np.zeros((n_new, self.ninputs)) if Dpad is None else _as2d(Dpad)
        Rpad = np.full(n_new, np.inf) if Rpad is None else \
            np.asarray(Rpad, dtype=float).ravel()
        old = np.ones(new_size, dtype=bool)
        old[pad_idx] = False
        newC = np.empty((new_size, self.order))
        newD = np.empty((new_size, self.ninputs))
        newR = np.zeros((new_size, new_size))
        newC[pad_idx, :], newC[old, :] = Cpad, self.C
        newD[pad_idx, :], newD[old, :] = Dpad, self.D
        newR[pad_idx, pad_idx] = Rpad
        newR[np.ix_(old, old)] = self.R
        return self._with_params(C=newC, D=newD, R=newR)

    def exclude_output(self, exclude_idx) -> "LinSys":
        """Drop output rows (MATLAB: excludeOutput)."""
        exclude_idx = np.atleast_1d(exclude_idx)
        if exclude_idx.dtype == bool:
            exclude_idx = np.where(exclude_idx)[0]
        return self._with_params(
            C=np.delete(self.C, exclude_idx, axis=0),
            D=np.delete(self.D, exclude_idx, axis=0),
            R=np.delete(np.delete(self.R, exclude_idx, axis=0),
                        exclude_idx, axis=1))

    def reduce(self, dat_set=None):
        """Equivalent model with output reduced to nx dimensions (MATLAB:
        reduce; wraps kalman.reduce_model). Infinite-variance outputs are
        dropped first. Returns (new LinSys, reduced DataSet or None)."""
        exc = np.isinf(np.diag(self.R))
        this = self.exclude_output(exc)
        if dat_set is None:
            Y = np.zeros((this.noutputs, 1))
        else:
            Y = dat_set.out[~exc, :]
        Cn, Rn, Yn, _, _, Dn = reduce_model(this.C, this.R, Y, this.D)
        new = LinSys(this.A, this.B, Cn, Dn, this.Q, Rn, name=self.name)
        red = None if dat_set is None else DataSet(dat_set.in_, Yn)
        return new, red

    # -- stochastic/deterministic decomposition --------------------------------
    def noise_covar(self, N):
        """Covariance of the stochastic state component after N steps
        (MATLAB: noiseCovar)."""
        M = self.Q.copy()
        for _ in range(N - 1):
            M = self.A @ M @ self.A.T + self.Q
        return M

    def det_predict(self, N=None):
        """Deterministic state after N steps (infinite horizon if N is None)
        from x = 0 under a step on the first input (MATLAB: detPredict)."""
        I = np.eye(self.order)
        b1 = self.B[:, 0]
        if N is not None:
            return np.linalg.solve(I - self.A,
                                   (I - np.linalg.matrix_power(self.A, N)) @ b1)
        return np.linalg.solve(I - self.A, b1)

    def snr(self, N):
        """SNR-like estimate per state after N steps (MATLAB: SNR)."""
        X = self.det_predict(N)
        M = self.noise_covar(N)
        return X ** 2 / np.diag(M)

    # -- model comparison helpers ------------------------------------------
    def bic_aic(self, dat_set, logL=None):
        """BIC/AIC information criteria of this model on a dataset (wraps
        model_selection.bic_aic; no MATLAB linsys equivalent). ``logL``
        defaults to ``self.logL(dat_set)``. Higher is better (MATLAB sign
        convention, see model_selection.bic_aic)."""
        if logL is None:
            logL = self.logL(dat_set)
        Y = np.hstack(dat_set.out) if dat_set.is_multiple else dat_set.out
        return _bic_aic(self.to_dict(), Y, logL)

    # -- conversion ------------------------------------------------------------
    def to_dict(self):
        """Plain dict (with both 'A' and 'J' keys) as used by linsys.viz
        (MATLAB: linsys2struct)."""
        return {"A": self.A, "J": self.A, "B": self.B, "C": self.C,
                "D": self.D, "Q": self.Q, "R": self.R, "name": self.name}

    linsys2struct = to_dict  # MATLAB name

    @classmethod
    def from_dict(cls, d):
        """Build from a dict with keys A (or J), B, C, D, Q, R (MATLAB:
        struct2linsys). Lists of dicts return lists of models."""
        if isinstance(d, (list, tuple)):
            return [cls.from_dict(x) for x in d]
        A = d["A"] if "A" in d else d["J"]
        return LinSys(A, d["B"], d["C"], d["D"], d["Q"], d["R"],
                      name=d.get("name", ""))

    struct2linsys = from_dict  # MATLAB name

    # -- plotting (delegates to linsys.viz) -------------------------------------
    def viz(self):
        """(MATLAB: linsys.viz -> vizModels)"""
        from .viz import viz_models
        return viz_models([self.to_dict()])

    def viz_fit(self, dat_set):
        """(MATLAB: vizFit -> dset.vizFit -> vizDataFit)"""
        return dat_set.viz_fit(self)

    def viz_res(self, dat_set):
        """(MATLAB: vizRes -> dset.vizRes -> vizDataRes)"""
        return dat_set.viz_res(self)

    @staticmethod
    def viz_many(models):
        """(MATLAB: vizMany -> vizModels)"""
        from .viz import viz_models
        return viz_models(_as_model_dicts(models))

    # -- identification (static constructors) -----------------------------------
    @classmethod
    def id(cls, dat_set, order, opts=None, rng=None):
        """Identify a model from data by (random-restart) EM
        (MATLAB: linsys.id; wraps em.random_start_em / get_flat_model).

        ``order`` 0 fits the static (flat) model; > 0 runs EM. A sequence of
        orders returns a list of models; a list of DataSet returns a list of
        models (one per dataset, fitted independently); a *multiple* DataSet
        is treated as several realizations of the SAME system. ``opts``
        defaults to ``EMOpts(Nreps=0)`` (single EM from the PCA-based
        initialization). Returns FittedLinSys (the MATLAB second output,
        outlog, is stored as ``training_log``)."""
        from .em import EMOpts
        from .em import random_start_em
        from .misc_helpers import get_flat_model
        if opts is None:
            opts = EMOpts(Nreps=0)
        if not np.isscalar(order):
            return [cls.id(dat_set, o, opts, rng) for o in np.ravel(order)]
        if isinstance(dat_set, list):
            return [cls.id(d, order, opts, rng) for d in dat_set]
        if dat_set.is_multiple:
            warnings.warn("Provided a multiple dataset. This will be treated "
                          "as many realizations of the same system for "
                          "identification purposes. If each dataset is a "
                          "realization of an independent system, provide the "
                          "datasets in a Python list instead.")
        if order == 0:
            fm = get_flat_model(dat_set.out, dat_set.in_,
                                opts.include_output_idx)
            this = FittedLinSys(fm.J, fm.B, fm.C, fm.D, fm.Q, fm.R,
                                InitCond(), dat_set, "EM", opts, fm.logL,
                                None, name="Flat")
            if opts.fix_r is not None and not np.isscalar(opts.fix_r):
                this.R = _as2d(opts.fix_r)
            return this
        elif order > 0:
            res = random_start_em(dat_set.out, dat_set.in_, int(order), opts,
                                  rng=rng)
            if dat_set.is_multiple:
                ic = InitCond([x[:, 0] for x in res.X],
                              [p[0] for p in res.P])
            else:
                ic = InitCond(res.X[:, 0], res.P[0])  # MLE initial condition
            return FittedLinSys(res.A, res.B, res.C, res.D, res.Q, res.R,
                                ic, dat_set, "repeatedEM", opts, res.logL,
                                res.out_log, name=f"rEM {int(order)}")
        raise ValueError("Order must be a non-negative integer.")

    @classmethod
    def ss_id(cls, dat_set, order=None, ss_size=10, method="SS"):
        """Identify a model by subspace methods (MATLAB: linsys.SSid).

        method 'SS' -> subspace.subspace_id (biased, Van Overschee & De Moor
        1996 Ch. 4 Algo. 2); 'SSunb' -> subspace.subspace_id_unbiased
        (Algo. 1/3). 'SSEM' and 'subid' are not available (see module
        docstring). Returns a FittedLinSys."""
        from .subspace import subspace_id, subspace_id_unbiased
        if method == "SS":
            r = subspace_id(dat_set.out, dat_set.in_, order, ss_size)
        elif method == "SSunb":
            r = subspace_id_unbiased(dat_set.out, dat_set.in_, order, ss_size)
        else:
            raise NotImplementedError(f"ss_id: method '{method}' is not "
                                      "available in the Python port")
        mdl = LinSys(r.A, r.B, r.C, r.D, r.Q, r.R)
        return FittedLinSys(r.A, r.B, r.C, r.D, r.Q, r.R, InitCond(), dat_set,
                            f"{method}_i{ss_size}", None, mdl.logL(dat_set),
                            None, name=f"{method} {r.A.shape[0]}")

    @staticmethod
    def summary_table(models):
        """Comparison summary of canonized models (MATLAB: summaryTable,
        which returns a MATLAB table; a dict of arrays here).

        Each model is canonized ('canonicalAlt') and scaled by
        1/norm(D[:, 0]). Returns a dict with 'row_names' (model names),
        'tau' (max_order, M) time constants, 'Q' diagonal of Q, 'B' first
        column of B, and 'trR'/'minR'/'maxR' over finite entries of
        diag(R)."""
        mdl = [m.canonize("canonicalAlt")[0].scale(
            1.0 / np.sqrt(np.sum(m.D[:, 0] ** 2))) for m in models]
        M = len(mdl)
        N = max(m.order for m in mdl)
        taus = np.full((N, M), np.nan, dtype=complex)
        bees = np.full((N, M), np.nan)
        dQ = np.full((N, M), np.nan)
        trR = np.full(M, np.nan)
        minR = np.full(M, np.nan)
        maxR = np.full(M, np.nan)
        for i, m in enumerate(mdl):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.sort(-1.0 / np.log(m.eigenvalues.astype(complex)))
            taus[:t.size, i] = t
            bees[:m.order, i] = m.B[:, 0]
            dQ[:m.order, i] = np.diag(m.Q)
            dR = np.diag(m.R)
            dR = dR[~np.isinf(dR)]
            trR[i], minR[i], maxR[i] = dR.sum(), dR.min(), dR.max()
        return {"row_names": [m.name for m in models],
                "tau": np.real_if_close(taus), "Q": dQ, "B": bees,
                "trR": trR, "minR": minR, "maxR": maxR}


# ---------------------------------------------------------------------------
# DataFit (dataFit.m)
# ---------------------------------------------------------------------------
class DataFit:
    """A model, a dataset and the resulting state estimate (MATLAB: dataFit).

    fit_method: 'KS' (smoothed, default), 'KF' (filtered) or 'KP' (one-ahead
    predicted) states. ``logL`` is the exact data log-likelihood.

    Deviation: the given initial condition is stored in
    ``initial_condition`` (the MATLAB class never assigns it - a bug - so
    its deterministic residual always used the default improper prior)."""

    def __init__(self, model, dat_set, fit_method=None, init_c=None):
        if isinstance(dat_set, list) or dat_set.is_multiple:
            raise NotImplementedError("DataFit: multiple datasets are not "
                                      "supported (nor do they work in MATLAB)")
        if init_c is None:
            init_c = InitCond()
        if fit_method is None:
            fit_method = "KS"
        ks = model.ksmooth(dat_set, init_c)
        if fit_method == "KF":
            self.state_estim = ks.filtered
        elif fit_method == "KS":
            self.state_estim = ks.smoothed
        elif fit_method == "KP":
            self.state_estim = ks.one_ahead
        else:
            raise ValueError(f"DataFit: unknown fit_method '{fit_method}'")
        self.model = model
        self.data_set = dat_set
        self.fit_method = fit_method
        self.initial_condition = init_c
        self.logL = ks.logL

    @property
    def output(self):
        """Fitted output C x + D u (MATLAB: output)."""
        N = self.data_set.nsamp
        return (self.model.C @ self.state_estim.state[:, :N]
                + self.model.D @ self.data_set.in_)

    @property
    def residual(self):
        """Fitted output minus data (MATLAB: residual)."""
        return self.output - self.data_set.out

    def n_ahead_output(self, M):
        """Output predicted M steps ahead from the state estimates (MATLAB:
        NaheadOutput). First M samples are NaN. Not defined for 'KS' fits
        (smoothing uses future data).

        y_hat[k] = C (A^M x[k-M] + sum_i A^(M-1-i) B u[k-M+i]) + D u[k];
        for M=1 on 'KF' fits this is the standard one-step-ahead prediction
        (the MATLAB version feeds inputs one sample late; see predict2)."""
        if self.fit_method == "KS":
            raise ValueError("dataFit:oneAheadOutputNotPredictive - "
                             "M-ahead output for smoothed states makes no "
                             "sense (smoothing uses future data!)")
        N = self.data_set.nsamp
        nx = self.model.order
        m_ahead = np.hstack([np.full((nx, M), np.nan),
                             self.state_estim.state[:, :N - M]])
        st = StateEstimate(m_ahead, np.zeros((nx, nx)))
        padded_in = np.hstack([np.full((self.data_set.ninputs, M), np.nan),
                               self.data_set.in_])
        pst = self.model.predict2(st, padded_in, M)
        return self.model.C @ pst.state + self.model.D @ self.data_set.in_

    @property
    def one_ahead_output(self):
        """(MATLAB: oneAheadOutput)"""
        return self.n_ahead_output(1)

    @property
    def one_ahead_residual(self):
        """(MATLAB: oneAheadResidual)"""
        return self.one_ahead_output - self.data_set.out

    def n_ahead_residual(self, M):
        """(MATLAB: NaheadResidual)"""
        return self.n_ahead_output(M) - self.data_set.out

    @property
    def deterministic_residual(self):
        """Residual of the deterministic noiseless simulation from the stored
        initial condition (MATLAB: deterministicResidual)."""
        sim, _ = self.model.simulate(self.data_set.in_,
                                     self.initial_condition, True, True)
        return sim.out - self.data_set.out

    @property
    def goodness_of_fit(self):
        """(MATLAB: goodnessOfFit)"""
        if self.fit_method in ("KF", "KS"):
            return self.logL
        elif self.fit_method == "LS":
            return float(np.sqrt(np.nanmean(np.nansum(self.residual ** 2,
                                                      axis=0))))
        raise NotImplementedError(f"goodness_of_fit: unimplemented for "
                                  f"method '{self.fit_method}'")

    @property
    def obs_noise(self):
        """MLE observation-noise samples (KS fits only) (MATLAB: obsNoise)."""
        if self.fit_method != "KS":
            raise ValueError("Observation noise estimation is only defined "
                             "for KS fitted models")
        return self.residual

    @property
    def state_noise(self):
        """MLE state-noise samples (KS fits only) (MATLAB: stateNoise)."""
        if self.fit_method != "KS":
            raise ValueError("State noise estimation is only defined for KS "
                             "fitted models")
        N = self.data_set.nsamp
        x = self.state_estim.state
        return (x[:, 1:N] - self.model.A @ x[:, :N - 1]
                - self.model.B @ self.data_set.in_[:, :N - 1])


# ---------------------------------------------------------------------------
# FittedLinSys (fittedLinsys.m)
# ---------------------------------------------------------------------------
class FittedLinSys(LinSys):
    """A LinSys plus metadata of the fitting run (MATLAB: fittedLinsys).

    Stores the training DataSet itself (``data_set``; MATLAB stores only its
    hash), the hash, the fitting method/options/log and the goodness of fit
    (the logL for EM fits). ``train_info`` bundles hash/method/options as a
    :class:`TrainInfo`.
    """

    def __init__(self, A, B, C, D, Q, R, i_c=None, data_set=None, method="",
                 opts=None, goodness_of_fit=None, training_log=None, name=""):
        super().__init__(A, B, C, D, Q, R, name=name)
        self.init_cond_prior = InitCond() if i_c is None else i_c
        self.data_set = data_set
        self.data_set_hash = "" if data_set is None else data_set.hash
        self.data_set_non_nan_samples = (
            None if data_set is None else int(np.sum(data_set.non_nan_samp)))
        self.method = method
        self.train_options = opts
        self.goodness_of_fit = goodness_of_fit
        self.training_log = training_log

    @property
    def train_info(self) -> TrainInfo:
        """Fit metadata as a TrainInfo (MATLAB: trainInfo class)."""
        return TrainInfo(self.data_set_hash, self.method, self.train_options)

    def _processed_opts(self):
        from .em import process_em_opts
        if self.method not in ("EM", "repeatedEM"):
            raise ValueError("The fitting method is unknown, cannot count "
                             "free parameters")
        return process_em_opts(self.train_options, self.ninputs, self.order,
                               self.noutputs)

    def r_dof(self):
        """Degrees of freedom of R alone (MATLAB: Rdof)."""
        o = self._processed_opts()
        if o.fix_r is not None:
            return 0
        m = len(o.include_output_idx)
        if o.diag_r:
            return m
        return m * (m + 1) // 2

    def dof(self):
        """Effective degrees of freedom of the fitted model (MATLAB: dof).

        Presumes a diagonal(izable) A: A contributes its eigenvalues only;
        up to nx scale parameters of B/C/Q are redundant."""
        nx, ny = self.order, self.noutputs
        o = self._processed_opts()
        Na = 0 if o.fix_a is not None else nx
        Nb = 0 if o.fix_b is not None else len(o.ind_b) * nx
        Nc = 0 if o.fix_c is not None else len(o.include_output_idx) * nx
        Nd = 0 if o.fix_d is not None else len(o.ind_d) * ny
        Nq = 0 if o.fix_q is not None else nx * (nx + 1) // 2
        Nr = self.r_dof()
        Nx0 = 0 if o.fix_x0 is not None else nx
        Np = 0  # P0 = Q in the current EM: no additional dof
        Nbcq = Nb + Nc + Nq - nx
        df = Na + Nbcq + Nd + Nr + Nx0 + Np
        if self.order == 1 and np.all(self.A == 0):  # flat model
            df = Nd + Nr
        return df

    @property
    def fitted_logL(self):
        """Total training logL (only defined for EM fits) (MATLAB: fittedLogL)."""
        if self.method in ("EM", "repeatedEM"):
            return float(np.sum(self.goodness_of_fit))
        raise ValueError("logL is not the goodness of fit metric unless EM "
                         "was used to fit the model")

    @property
    def BIC(self):
        """-2 logL + log(N) dof (lower is better) (MATLAB: BIC)."""
        if self.method not in ("EM", "repeatedEM"):
            raise ValueError("BIC is not defined unless goodness of fit "
                             "metric is logL")
        return (-2 * self.fitted_logL
                + np.log(self.data_set_non_nan_samples) * self.dof())

    @property
    def AIC(self):
        """-2 logL + 2 dof (lower is better) (MATLAB: AIC)."""
        if self.method not in ("EM", "repeatedEM"):
            raise ValueError("AIC is not defined unless goodness of fit "
                             "metric is logL")
        return -2 * self.fitted_logL + 2 * self.dof()

    @property
    def AICc(self):
        """Sample-size corrected AIC (Burnham & Anderson 2002, eq. 7.91)
        (MATLAB: AICc)."""
        if self.method not in ("EM", "repeatedEM"):
            raise ValueError("AICc is not defined unless goodness of fit "
                             "metric is logL")
        p = self.dof()
        N = self.data_set_non_nan_samples
        v = self.r_dof()
        k = p + v
        return self.AIC + 2 * p * k / (N * self.noutputs - k)

    def likelihood_ratio_test(self, other: "FittedLinSys"):
        """Likelihood-ratio test of two (nested) fits of the SAME dataset via
        Wilks' approximation (MATLAB: likelihoodRatioTest).

        CAUTION (as in MATLAB): the chi^2 approximation is known NOT to be
        valid for LTI-SSM models of different orders.
        Returns (p, chi2_stat, delta_dof)."""
        from scipy.stats import chi2
        if self.data_set_hash != other.data_set_hash:
            raise ValueError("dataFit:LRT - performing likelihood ratio test "
                             "on fits of different datasets")
        for m in (self, other):
            if m.method not in ("EM", "repeatedEM"):
                raise ValueError("LRT is not defined unless goodness of fit "
                                 "metric is logL")
        delta_dof = abs(self.dof() - other.dof())
        chi = 2 * (np.sum(self.goodness_of_fit)
                   - np.sum(other.goodness_of_fit))
        if chi < 2:
            warnings.warn("dataFit:LRT - model with more parameters has "
                          "lower likelihood. Either a bad fit or the models "
                          "were not nested.")
        return float(chi2.sf(chi, delta_dof)), float(chi), int(delta_dof)


def fit_linsys(dat_set, order, opts=None, rng=None):
    """Convenience top-level fit entry point (wraps LinSys.id, the port of
    MATLAB linsys.id). ``dat_set`` may be a DataSet or a (Y, U) tuple."""
    if isinstance(dat_set, tuple) and len(dat_set) == 2:
        Y, U = dat_set
        dat_set = DataSet(U, Y)
    return LinSys.id(dat_set, order, opts, rng)

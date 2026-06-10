# python-linsys

A Python port of [matlab-linsys](https://github.com/pabloi/matlab-linsys): estimation,
identification, and analysis of linear dynamical systems (LDS / linear state-space
models) with Gaussian noise.

```
x[k+1] = A x[k] + B u[k] + w[k],   w ~ N(0, Q)
y[k]   = C x[k] + D u[k] + z[k],   z ~ N(0, R)
x[0] ~ N(x0, P0)
```

All numerics are pure numpy/scipy; matplotlib is an optional dependency for the
visualization module.

## Install

```sh
pip install -e .          # core (numpy, scipy)
pip install -e .[viz]     # + matplotlib
pip install -e .[dev]     # + pytest
```

## Quick start

```python
import numpy as np
from linsys import fwd_sim, kalman, em

# simulate a 2-state, 3-output system driven by a step input
A = np.diag([0.95, 0.7]); B = np.array([[0.05], [0.3]])
C = np.random.randn(3, 2); D = np.zeros((3, 1))
Q = 1e-3 * np.eye(2);      R = 1e-2 * np.eye(3)
U = np.ones((1, 500))
Y, X = fwd_sim(U, A, B, C, D, Q=Q, R=R, rng=np.random.default_rng(0))

# Kalman smoothing with the true model (improper flat prior by default)
res = kalman.smoother_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
res.Xs       # smoothed states (nx, N)
res.logL     # per-sample log-likelihood

# identify a model from data alone with EM
fit = em.em(Y, U, 2)
```

Or with the object-oriented layer (ports of the MATLAB `linsys`/`dset` classes):

```python
from linsys import LinSys, DataSet, fit_linsys

true_model = LinSys(A, B, C, D, Q, R, name="truth")
data = true_model.simulate(U, deterministic_flag=False)   # -> DataSet
fitted = fit_linsys(data, order=2)                        # FittedLinSys via EM
fitted.ksmooth(data)                                      # smoothed StateEstimate
fitted.canonize()                                         # canonical form
```

## Conventions

- 2-D data arrays keep the MATLAB orientation: **columns are time samples** —
  `Y` is `(ny, N)`, `U` is `(nu, N)`, `X` is `(nx, N)`.
- Covariance stacks are time-first: `P` is `(N, nx, nx)` (MATLAB `P(:,:,k)` ↔ `P[k]`).
- An all-NaN column of `Y` is a missing sample (skipped in updates).
- `P0 = inf·I` (the default) is a valid improper flat prior, handled with an
  information-form startup.
- Multiple realizations of one system are passed as Python lists (MATLAB cell arrays).

See [PORTING.md](PORTING.md) for the full set of binding conventions and the
MATLAB → Python name map.

## Modules

| Module | Contents | MATLAB origin |
|---|---|---|
| `linsys.model` | object-oriented layer: `LinSys`, `FittedLinSys`, `DataSet`, `DataFit`, `StateEstimate`, `InitCond`, `TrainInfo`, `fit_linsys` | `@linsys`, `@dset`, … classes |
| `linsys.kalman` | stationary Kalman filter/smoother (covariance, information, square-root, constrained, CS2006 forms); steady-state fast mode, model reduction for ny > nx, outlier rejection, improper priors | `kalman/` |
| `linsys.em` | EM system identification: `em`, `estimate_params` (with fixable parameters, stable-A enforcement, robust Q), `init_em`, `random_start_em`, cross-validation | `EM/` |
| `linsys.subspace` | subspace identification (`subspace_id`, `subspace_id_unbiased`, `subspace_id_v2`) | `subspace/` |
| `linsys.spca` | smooth PCA (`spca`), dynamics fitting (`estimate_dyn*`), cross-validation | `sPCA/` |
| `linsys.hmm` | discrete-state HMM: forward/backward inference, Viterbi, EM fitting (column-stochastic transition matrices) | `discrHMM/` |
| `linsys.stats` | `data_log_likelihood` (exact/approx/fast/max), complete/incomplete-data logL | `misc/` |
| `linsys.model_selection` | BIC/AIC, CV fold splitting, model matching | `misc/` |
| `linsys.utils` | PSD Cholesky factorizations (`cholcov`, `pinvchol`), similarity transforms, canonical forms (`canonize`), simulation (`fwd_sim`), robust covariance | `misc/` |
| `linsys.viz` | matplotlib figures for data fits, likelihood comparisons, residuals, model parameters | `misc/viz/` |

## Tests

Everything is validated on synthetic data against textbook reference
implementations (naive Kalman filter / RTS smoother) and internal consistency
checks:

```sh
python -m pytest
```

## Status / scope

All functional MATLAB code is ported except third-party code (`ext/`) and
folders marked deprecated upstream (`old/`, `legacy/`, `forDeletion/`).
Known deviations from MATLAB and upstream bugs discovered during the port are
documented in module docstrings and tracked as GitHub issues.

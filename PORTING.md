# Porting conventions (matlab-linsys -> python-linsys)

This package is a port of https://github.com/pabloi/matlab-linsys (source located at
`../matlab-linsys` relative to this repo). These conventions are binding for all modules.

## Model

```
x[k+1] = A x[k] + B u[k] + w[k],  w ~ N(0, Q)
y[k]   = C x[k] + D u[k] + z[k],  z ~ N(0, R)
x[0] ~ N(x0, P0)   (note: P0 is cov of x[0] itself, not of x[0|-1])
```

## Data layout

- 2-D data matrices keep the MATLAB orientation: **columns are time samples**.
  `Y` is `(ny, N)`, `U` is `(nu, N)`, `X` is `(nx, N)`. State vectors are 1-D `(nx,)`.
- Stacks of covariance matrices use the **first** axis for time: `P` is `(N, nx, nx)`
  (MATLAB `P(:,:,k)` maps to Python `P[k]`).
- Missing data: an entire sample `Y[:, k]` may be NaN (it is skipped in updates).

## Naming

- snake_case; MATLAB name recorded in each docstring.
- `statKalmanFilter` -> `kalman.filter_stationary`, `statKalmanSmoother` ->
  `kalman.smoother_stationary`, `statInfoFilter2` -> `kalman.info_filter_stationary`, etc.
- `mycholcov` -> `utils.cholcov`, `pinvchol` -> `utils.pinvchol`, `logLnormal` ->
  `utils.logl_normal`.
- Options structs become dataclasses (`KalmanOpts`, `EMOpts`); multi-output functions
  return NamedTuples.

## Numerics

- Infinite prior covariance (`P0 = inf*I`, the default) is supported: the filter starts
  with an information-form pass until uncertainties are finite.
- PSD factorizations use `utils.cholcov` / `utils.pinvchol` / `utils.pinvchol2`,
  which accept semidefinite matrices and (for the `2` variants) Inf/0 diagonals with
  the convention Inf*0 = 0.
- `float64` throughout. No GPU paths (MATLAB gpuArray branches are dropped).

## Scope notes

- `ext/` (third-party), `*/old/`, `*/forDeletion/`, `*/legacy/` folders are NOT ported.
- MATLAB cell-array inputs (multiple realizations of one system) become Python lists.
- Warnings use `warnings.warn`; errors raise `ValueError`/`np.linalg.LinAlgError`.

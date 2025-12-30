"""
Utility functions for linear systems.

This module provides various helper functions for working with
linear state-space models, including simulation, log-likelihood
computation, model comparison, and transformations.

Based on the MATLAB matlab-linsys toolbox.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, List

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg


def simulate(
    A: ArrayLike,
    C: ArrayLike,
    n_steps: int,
    B: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    Q: Optional[ArrayLike] = None,
    R: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
    noise: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a linear state-space system.

    Model:
        x[k+1] = A @ x[k] + B @ u[k] + w[k],  w ~ N(0, Q)
        y[k]   = C @ x[k] + D @ u[k] + v[k],  v ~ N(0, R)

    Parameters
    ----------
    A : array_like, shape (n_states, n_states)
        State transition matrix.
    C : array_like, shape (n_outputs, n_states)
        Observation matrix.
    n_steps : int
        Number of time steps.
    B : array_like, shape (n_states, n_inputs), optional
        Input matrix.
    D : array_like, shape (n_outputs, n_inputs), optional
        Feedthrough matrix.
    Q : array_like, shape (n_states, n_states), optional
        Process noise covariance.
    R : array_like, shape (n_outputs, n_outputs), optional
        Measurement noise covariance.
    u : array_like, shape (n_inputs, n_steps), optional
        Input sequence.
    x0 : array_like, shape (n_states,), optional
        Initial state.
    noise : bool, default=True
        Whether to add noise.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    y : ndarray, shape (n_outputs, n_steps)
        Output sequence.
    x : ndarray, shape (n_states, n_steps)
        State sequence.
    u : ndarray, shape (n_inputs, n_steps)
        Input sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))

    n_states = A.shape[0]
    n_outputs = C.shape[0]

    # Initialize B and D
    if B is not None:
        B = np.atleast_2d(np.asarray(B, dtype=np.float64))
        n_inputs = B.shape[1]
    else:
        n_inputs = 1
        B = np.zeros((n_states, n_inputs))

    if D is not None:
        D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    else:
        D = np.zeros((n_outputs, n_inputs))

    # Initialize covariances
    if Q is None:
        Q = np.eye(n_states)
    else:
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))

    if R is None:
        R = np.eye(n_outputs)
    else:
        R = np.atleast_2d(np.asarray(R, dtype=np.float64))

    # Initialize input
    if u is None:
        u = np.zeros((n_inputs, n_steps))
    else:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))

    # Initialize state
    if x0 is None:
        x0 = np.zeros(n_states)
    else:
        x0 = np.asarray(x0, dtype=np.float64).ravel()

    # Generate noise
    if noise:
        w = _sample_multivariate_normal(Q, n_steps, rng)
        v = _sample_multivariate_normal(R, n_steps, rng)
    else:
        w = np.zeros((n_states, n_steps))
        v = np.zeros((n_outputs, n_steps))

    # Simulate
    x = np.zeros((n_states, n_steps))
    y = np.zeros((n_outputs, n_steps))

    x_curr = x0.copy()
    for k in range(n_steps):
        x[:, k] = x_curr
        y[:, k] = C @ x_curr + D @ u[:, k] + v[:, k]
        x_curr = A @ x_curr + B @ u[:, k] + w[:, k]

    return y, x, u


def _sample_multivariate_normal(
    cov: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from multivariate normal with given covariance."""
    n = cov.shape[0]

    try:
        L = linalg.cholesky(cov, lower=True)
        return L @ rng.standard_normal((n, n_samples))
    except linalg.LinAlgError:
        # Covariance might be singular
        eigvals, eigvecs = linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals))
        return sqrt_cov @ rng.standard_normal((n, n_samples))


def log_likelihood(
    y: ArrayLike,
    A: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    B: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
    P0: Optional[ArrayLike] = None,
    method: str = "exact",
) -> float:
    """
    Compute log-likelihood of observations under a state-space model.

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_steps)
        Observations.
    A, C, Q, R : array_like
        System matrices.
    B, D : array_like, optional
        Input matrices.
    u : array_like, optional
        Inputs.
    x0 : array_like, optional
        Initial state.
    P0 : array_like, optional
        Initial covariance.
    method : str, default="exact"
        Computation method: "exact", "approximate", or "fast".

    Returns
    -------
    log_lik : float
        Log-likelihood value.
    """
    from .kalman import kalman_filter

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n_steps = y.shape[1]

    if u is None:
        u = np.zeros((1, n_steps))

    _, _, _, _, log_lik = kalman_filter(
        y, A, C, Q, R,
        B=B, D=D, u=u, x0=x0, P0=P0,
        steady_state=(method == "fast"),
    )

    return log_lik


def log_likelihood_normal(
    x: ArrayLike,
    mean: ArrayLike,
    cov: ArrayLike,
) -> float:
    """
    Compute log-likelihood under multivariate normal distribution.

    Parameters
    ----------
    x : array_like, shape (n,)
        Data point.
    mean : array_like, shape (n,)
        Mean vector.
    cov : array_like, shape (n, n)
        Covariance matrix.

    Returns
    -------
    log_lik : float
        Log-likelihood.
    """
    x = np.asarray(x).ravel()
    mean = np.asarray(mean).ravel()
    cov = np.atleast_2d(np.asarray(cov))

    n = len(x)
    diff = x - mean

    try:
        L = linalg.cholesky(cov, lower=True)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        z = linalg.solve_triangular(L, diff, lower=True)
        mahal_sq = np.dot(z, z)
    except linalg.LinAlgError:
        eigvals = linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        log_det = np.sum(np.log(eigvals))
        mahal_sq = diff @ linalg.pinv(cov) @ diff

    return -0.5 * (n * np.log(2 * np.pi) + log_det + mahal_sq)


def bic_aic(
    log_lik: float,
    n_params: int,
    n_samples: int,
) -> Tuple[float, float]:
    """
    Compute BIC and AIC model selection criteria.

    Parameters
    ----------
    log_lik : float
        Log-likelihood of the model.
    n_params : int
        Number of free parameters.
    n_samples : int
        Number of samples.

    Returns
    -------
    bic : float
        Bayesian Information Criterion (lower is better).
    aic : float
        Akaike Information Criterion (lower is better).
    """
    bic = -2 * log_lik + n_params * np.log(n_samples)
    aic = -2 * log_lik + 2 * n_params

    return bic, aic


def count_parameters(
    n_states: int,
    n_inputs: int,
    n_outputs: int,
    diagonal_A: bool = False,
    diagonal_Q: bool = False,
    diagonal_R: bool = False,
    fix_B: bool = False,
    fix_D: bool = False,
) -> int:
    """
    Count number of free parameters in a state-space model.

    Parameters
    ----------
    n_states : int
        Number of states.
    n_inputs : int
        Number of inputs.
    n_outputs : int
        Number of outputs.
    diagonal_A : bool
        Whether A is constrained to be diagonal.
    diagonal_Q : bool
        Whether Q is constrained to be diagonal.
    diagonal_R : bool
        Whether R is constrained to be diagonal.
    fix_B : bool
        Whether B is fixed.
    fix_D : bool
        Whether D is fixed.

    Returns
    -------
    n_params : int
        Number of free parameters.
    """
    # A matrix
    if diagonal_A:
        n_A = n_states
    else:
        n_A = n_states * n_states

    # B matrix
    if fix_B:
        n_B = 0
    else:
        n_B = n_states * n_inputs

    # C matrix
    n_C = n_outputs * n_states

    # D matrix
    if fix_D:
        n_D = 0
    else:
        n_D = n_outputs * n_inputs

    # Q matrix (symmetric)
    if diagonal_Q:
        n_Q = n_states
    else:
        n_Q = n_states * (n_states + 1) // 2

    # R matrix (symmetric)
    if diagonal_R:
        n_R = n_outputs
    else:
        n_R = n_outputs * (n_outputs + 1) // 2

    # Initial state and covariance
    n_x0 = n_states
    n_P0 = n_states * (n_states + 1) // 2

    return n_A + n_B + n_C + n_D + n_Q + n_R + n_x0 + n_P0


def forward_simulate(
    A: ArrayLike,
    C: ArrayLike,
    x0: ArrayLike,
    n_steps: int,
    B: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic forward simulation (no noise).

    Parameters
    ----------
    A : array_like, shape (n_states, n_states)
        State transition matrix.
    C : array_like, shape (n_outputs, n_states)
        Observation matrix.
    x0 : array_like, shape (n_states,)
        Initial state.
    n_steps : int
        Number of time steps.
    B : array_like, shape (n_states, n_inputs), optional
        Input matrix.
    D : array_like, shape (n_outputs, n_inputs), optional
        Feedthrough matrix.
    u : array_like, shape (n_inputs, n_steps), optional
        Input sequence.

    Returns
    -------
    y : ndarray, shape (n_outputs, n_steps)
        Output sequence.
    x : ndarray, shape (n_states, n_steps)
        State sequence.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    x0 = np.asarray(x0, dtype=np.float64).ravel()

    n_states = A.shape[0]
    n_outputs = C.shape[0]

    if B is not None:
        B = np.atleast_2d(np.asarray(B, dtype=np.float64))
        n_inputs = B.shape[1]
    else:
        n_inputs = 1
        B = np.zeros((n_states, n_inputs))

    if D is not None:
        D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    else:
        D = np.zeros((n_outputs, n_inputs))

    if u is None:
        u = np.zeros((n_inputs, n_steps))
    else:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))

    x = np.zeros((n_states, n_steps))
    y = np.zeros((n_outputs, n_steps))

    x_curr = x0.copy()
    for k in range(n_steps):
        x[:, k] = x_curr
        y[:, k] = C @ x_curr + D @ u[:, k]
        x_curr = A @ x_curr + B @ u[:, k]

    return y, x


def compute_output(
    C: ArrayLike,
    x: ArrayLike,
    D: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Compute outputs from states.

    Parameters
    ----------
    C : array_like, shape (n_outputs, n_states)
        Observation matrix.
    x : array_like, shape (n_states, n_steps)
        State sequence.
    D : array_like, shape (n_outputs, n_inputs), optional
        Feedthrough matrix.
    u : array_like, shape (n_inputs, n_steps), optional
        Input sequence.

    Returns
    -------
    y : ndarray, shape (n_outputs, n_steps)
        Output sequence.
    """
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))

    y = C @ x

    if D is not None and u is not None:
        D = np.atleast_2d(np.asarray(D, dtype=np.float64))
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        y = y + D @ u

    return y


def substitute_nans(
    y: ArrayLike,
    method: str = "interpolate",
) -> np.ndarray:
    """
    Replace NaN values in observations.

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_steps)
        Observations with potential NaN values.
    method : str, default="interpolate"
        Method: "interpolate", "zero", "mean", or "last".

    Returns
    -------
    y_filled : ndarray
        Observations with NaN values replaced.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64)).copy()

    for i in range(y.shape[0]):
        nan_mask = np.isnan(y[i, :])
        if not np.any(nan_mask):
            continue

        if method == "zero":
            y[i, nan_mask] = 0

        elif method == "mean":
            valid_mean = np.nanmean(y[i, :])
            y[i, nan_mask] = valid_mean

        elif method == "last":
            # Forward fill
            for j in range(1, y.shape[1]):
                if np.isnan(y[i, j]):
                    y[i, j] = y[i, j - 1]

        elif method == "interpolate":
            # Linear interpolation
            valid_idx = np.where(~nan_mask)[0]
            if len(valid_idx) > 0:
                nan_idx = np.where(nan_mask)[0]
                y[i, nan_idx] = np.interp(nan_idx, valid_idx, y[i, valid_idx])

    return y


def diagonalize_A(
    A: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalize the state transition matrix.

    Finds T such that T^{-1} @ A @ T is diagonal (or block-diagonal for
    complex eigenvalues).

    Parameters
    ----------
    A : array_like, shape (n, n)
        State transition matrix.

    Returns
    -------
    A_diag : ndarray
        Diagonalized A matrix.
    T : ndarray
        Transformation matrix (eigenvectors).
    eigenvalues : ndarray
        Eigenvalues of A.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))

    eigenvalues, T = linalg.eig(A)

    # Check if T is invertible
    try:
        T_inv = linalg.inv(T)
        A_diag = T_inv @ A @ T
    except linalg.LinAlgError:
        # Eigenvector matrix is singular
        A_diag = np.diag(eigenvalues)
        T = np.eye(A.shape[0])

    return A_diag, T, eigenvalues


def canonize_system(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    Q: Optional[ArrayLike] = None,
    R: Optional[ArrayLike] = None,
    form: str = "modal",
) -> Tuple[np.ndarray, ...]:
    """
    Transform system to canonical form.

    Parameters
    ----------
    A, B, C, D : array_like
        System matrices.
    Q, R : array_like, optional
        Noise covariances.
    form : str, default="modal"
        Canonical form: "modal" diagonalizes A.

    Returns
    -------
    A, B, C, D, Q, R : ndarray
        Transformed system matrices.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))

    n_states = A.shape[0]

    if Q is None:
        Q = np.eye(n_states)
    else:
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))

    if R is None:
        R = np.eye(C.shape[0])
    else:
        R = np.atleast_2d(np.asarray(R, dtype=np.float64))

    if form == "modal":
        _, T, _ = diagonalize_A(A)
        try:
            T_inv = linalg.inv(T)
        except linalg.LinAlgError:
            # Cannot diagonalize, return original
            return A, B, C, D, Q, R

        A_new = T_inv @ A @ T
        B_new = T_inv @ B
        C_new = C @ T
        D_new = D
        Q_new = T_inv @ Q @ T_inv.T
        R_new = R

        return A_new, B_new, C_new, D_new, Q_new, R_new

    else:
        raise ValueError(f"Unknown canonical form: {form}")


def rotate_factors(
    C: ArrayLike,
    rotation: str = "varimax",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation to factor loading matrix C.

    Parameters
    ----------
    C : array_like, shape (n_outputs, n_factors)
        Factor loading matrix.
    rotation : str, default="varimax"
        Rotation method.

    Returns
    -------
    C_rotated : ndarray
        Rotated loading matrix.
    R : ndarray
        Rotation matrix.
    """
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))

    if rotation == "varimax":
        C_rotated, R = _varimax_rotation(C)
    else:
        raise ValueError(f"Unknown rotation: {rotation}")

    return C_rotated, R


def _varimax_rotation(
    loadings: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Varimax rotation for factor loadings."""
    n, k = loadings.shape

    rotation = np.eye(k)
    loadings_rotated = loadings.copy()

    for _ in range(max_iter):
        for i in range(k):
            for j in range(i + 1, k):
                # Compute 2x2 rotation angle
                x = loadings_rotated[:, i]
                y = loadings_rotated[:, j]

                u = x ** 2 - y ** 2
                v = 2 * x * y

                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)

                num = D - 2 * A * B / n
                denom = C - (A ** 2 - B ** 2) / n

                if abs(denom) < 1e-10:
                    continue

                phi = 0.25 * np.arctan2(num, denom)

                # Apply rotation
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                loadings_rotated[:, i] = x * cos_phi + y * sin_phi
                loadings_rotated[:, j] = -x * sin_phi + y * cos_phi

                # Update rotation matrix
                rot_ij = np.eye(k)
                rot_ij[i, i] = cos_phi
                rot_ij[j, j] = cos_phi
                rot_ij[i, j] = sin_phi
                rot_ij[j, i] = -sin_phi
                rotation = rotation @ rot_ij

        # Check convergence
        diff = np.max(np.abs(loadings_rotated - loadings @ rotation))
        if diff < tol:
            break

    return loadings_rotated, rotation


def hankel_matrix(
    data: ArrayLike,
    n_rows: int,
    n_cols: Optional[int] = None,
) -> np.ndarray:
    """
    Construct a block Hankel matrix from data.

    Parameters
    ----------
    data : array_like, shape (n_channels, n_samples)
        Input data matrix.
    n_rows : int
        Number of block rows.
    n_cols : int, optional
        Number of columns.

    Returns
    -------
    H : ndarray
        Block Hankel matrix.
    """
    from .subspace import hankel_matrix as _hankel_matrix
    return _hankel_matrix(data, n_rows, n_cols)


def model_snr(
    y: ArrayLike,
    A: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    B: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> float:
    """
    Compute signal-to-noise ratio of a state-space model.

    SNR = var(signal) / var(noise) = var(C @ x) / var(y - C @ x)

    Parameters
    ----------
    y : array_like
        Observations.
    A, C, Q, R : array_like
        System matrices.
    B, D : array_like, optional
        Input matrices.
    u : array_like, optional
        Inputs.

    Returns
    -------
    snr : float
        Signal-to-noise ratio.
    """
    from .kalman import kalman_smoother

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n_steps = y.shape[1]

    if u is None:
        u = np.zeros((1, n_steps))

    x_smooth, _, _, _, _, _ = kalman_smoother(
        y, A, C, Q, R, B=B, D=D, u=u,
    )

    # Predicted output from smoothed states
    y_pred = compute_output(C, x_smooth, D, u)

    # Signal and noise variance
    signal_var = np.var(y_pred)
    noise_var = np.var(y - y_pred)

    if noise_var < 1e-10:
        return np.inf

    return signal_var / noise_var


def compare_models(
    y: ArrayLike,
    models: List,
    u: Optional[ArrayLike] = None,
) -> dict:
    """
    Compare multiple models on the same data.

    Parameters
    ----------
    y : array_like
        Observations.
    models : list of LinearSystem
        Models to compare.
    u : array_like, optional
        Inputs.

    Returns
    -------
    comparison : dict
        Dictionary with log-likelihood, BIC, AIC for each model.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n_outputs, n_steps = y.shape
    n_samples = n_outputs * n_steps

    results = {
        "log_likelihood": [],
        "bic": [],
        "aic": [],
        "n_params": [],
    }

    for model in models:
        ll = model.log_likelihood(y, u)
        n_params = count_parameters(
            model.n_states, model.n_inputs, model.n_outputs
        )
        bic_val, aic_val = bic_aic(ll, n_params, n_samples)

        results["log_likelihood"].append(ll)
        results["bic"].append(bic_val)
        results["aic"].append(aic_val)
        results["n_params"].append(n_params)

    return results

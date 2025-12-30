"""
Subspace identification methods for linear systems.

This module implements subspace-based system identification methods,
which estimate state-space models from input-output data using
linear algebra techniques (SVD, projections).

Based on the MATLAB matlab-linsys toolbox and:
- Van Overschee & De Moor (1996) "Subspace Identification for Linear Systems"
- Shadmehr & Mussa-Ivaldi (2012)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg


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
        Number of block rows (each block has n_channels rows).
    n_cols : int, optional
        Number of columns. Defaults to n_samples - n_rows + 1.

    Returns
    -------
    H : ndarray, shape (n_channels * n_rows, n_cols)
        Block Hankel matrix.
    """
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n_channels, n_samples = data.shape

    if n_cols is None:
        n_cols = n_samples - n_rows + 1

    if n_cols <= 0 or n_rows > n_samples:
        raise ValueError(
            f"Cannot create Hankel matrix with {n_rows} rows from "
            f"{n_samples} samples"
        )

    H = np.zeros((n_channels * n_rows, n_cols))

    for i in range(n_rows):
        H[i * n_channels:(i + 1) * n_channels, :] = data[:, i:i + n_cols]

    return H


def oblique_projection(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute oblique projection of A onto B along C.

    Computes A / C * B, the projection of A onto row space of B
    along the row space of C.

    Parameters
    ----------
    A : ndarray
        Matrix to project.
    B : ndarray
        Target row space.
    C : ndarray
        Direction of projection.

    Returns
    -------
    proj : ndarray
        Projected matrix.
    pinv_C : ndarray
        Pseudo-inverse used in projection.
    """
    # Stack B and C
    BC = np.vstack([B, C])

    # Compute pseudo-inverse
    pinv_BC = linalg.pinv(BC)

    # Extract part corresponding to B
    n_B = B.shape[0]
    pinv_B = pinv_BC[:, :n_B]

    # Project
    proj = A @ pinv_B @ B

    return proj, pinv_BC


def perpendicular_projection(
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Project A onto the space perpendicular to B.

    Parameters
    ----------
    A : ndarray
        Matrix to project.
    B : ndarray
        Space to project away from.

    Returns
    -------
    proj : ndarray
        Projected matrix (A with component in row space of B removed).
    """
    pinv_B = linalg.pinv(B)
    return A - A @ pinv_B @ B


def subspace_id(
    y: ArrayLike,
    n_states: int,
    u: Optional[ArrayLike] = None,
    horizon: int = 10,
) -> Tuple[np.ndarray, ...]:
    """
    Subspace identification for linear systems.

    Implements the N4SID-style algorithm following Van Overschee & De Moor.

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_samples)
        Output data.
    n_states : int
        Number of states to identify.
    u : array_like, shape (n_inputs, n_samples), optional
        Input data. Defaults to zeros.
    horizon : int, default=10
        Block size for Hankel matrices.

    Returns
    -------
    A : ndarray, shape (n_states, n_states)
        State transition matrix.
    B : ndarray, shape (n_states, n_inputs)
        Input matrix.
    C : ndarray, shape (n_outputs, n_states)
        Output matrix.
    D : ndarray, shape (n_outputs, n_inputs)
        Feedthrough matrix.
    X : ndarray, shape (n_states, n_samples)
        Estimated states.
    Q : ndarray, shape (n_states, n_states)
        Process noise covariance.
    R : ndarray, shape (n_outputs, n_outputs)
        Measurement noise covariance.
    S : ndarray, shape (n_states, n_outputs)
        Cross-covariance between process and measurement noise.

    Examples
    --------
    >>> # Generate data from a known system
    >>> sys = LinearSystem.random(n_states=2, n_outputs=3)
    >>> y, x, u = sys.simulate(1000)
    >>> # Identify the system
    >>> A, B, C, D, X, Q, R, S = subspace_id(y, n_states=2, u=u)
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n_outputs, n_samples = y.shape

    if u is None:
        u = np.zeros((1, n_samples))
    else:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))

    n_inputs = u.shape[0]
    i = horizon  # Block size

    # Number of columns in Hankel matrices
    j = n_samples - 2 * i

    if j <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for horizon {horizon}. "
            f"Need at least {2 * horizon + 1} samples."
        )

    # Construct block Hankel matrices
    # Past: rows 1 to i
    Y_past = hankel_matrix(y, i, j)
    U_past = hankel_matrix(u, i, j)

    # Future: rows i+1 to 2i
    Y_future = hankel_matrix(y[:, i:], i, j)
    U_future = hankel_matrix(u[:, i:], i, j)

    # Combined past
    W_past = np.vstack([U_past, Y_past])

    # Oblique projection of future outputs onto past, along future inputs
    # O_i = Y_f / [U_f; W_p] * W_p
    combined = np.vstack([U_future, W_past])
    O_proj, pinv_combined = oblique_projection(Y_future, W_past, U_future)

    # Remove effect of future inputs
    O_perp = perpendicular_projection(O_proj, U_future)

    # SVD to extract state sequence
    U_svd, S_svd, Vh_svd = linalg.svd(O_perp, full_matrices=False)

    # Determine rank (number of states)
    if n_states > len(S_svd):
        raise ValueError(
            f"Requested {n_states} states but only {len(S_svd)} "
            "singular values available"
        )

    # Truncate to n_states
    U1 = U_svd[:, :n_states]
    S1 = np.diag(S_svd[:n_states])
    V1 = Vh_svd[:n_states, :]

    # State sequences
    S1_sqrt = np.diag(np.sqrt(S_svd[:n_states]))

    # Extended observability matrix
    # Gamma = U1 @ S1_sqrt

    # State sequence (at time i+1 to i+j)
    X_mid = S1_sqrt @ V1

    # Shift to get state at i+2 to i+j+1
    # We use the structure of the Hankel matrix
    V1_shifted = Vh_svd[:n_states, 1:]  # Remove first column
    X_next = S1_sqrt @ V1_shifted

    # Extract corresponding outputs and inputs
    Y_mid = y[:, i:i + j - 1]  # y[i+1] to y[i+j-1]
    U_mid = u[:, i:i + j - 1]  # u[i+1] to u[i+j-1]

    X_curr = X_mid[:, :-1]  # States at i+1 to i+j-1

    # Solve for system matrices using least squares
    # [x_{k+1}]   [A  B] [x_k]
    # [y_k    ] = [C  D] [u_k]

    top = X_next
    bottom = Y_mid
    left = np.vstack([X_curr, U_mid])

    # Solve the least squares problem
    rhs = np.vstack([top, bottom])

    try:
        sol = linalg.lstsq(left.T, rhs.T)[0].T
    except linalg.LinAlgError:
        sol = rhs @ linalg.pinv(left)

    A = sol[:n_states, :n_states]
    B = sol[:n_states, n_states:]
    C = sol[n_states:, :n_states]
    D = sol[n_states:, n_states:]

    # Estimate noise covariances from residuals
    # State residuals
    x_pred = A @ X_curr + B @ U_mid
    state_residuals = X_next - x_pred

    # Output residuals
    y_pred = C @ X_curr + D @ U_mid
    output_residuals = Y_mid - y_pred

    n_residuals = state_residuals.shape[1]

    Q = state_residuals @ state_residuals.T / n_residuals
    R = output_residuals @ output_residuals.T / n_residuals
    S = state_residuals @ output_residuals.T / n_residuals

    # Ensure Q and R are positive semi-definite
    Q = (Q + Q.T) / 2
    R = (R + R.T) / 2

    eigvals_Q, eigvecs_Q = linalg.eigh(Q)
    eigvals_Q = np.maximum(eigvals_Q, 1e-10)
    Q = eigvecs_Q @ np.diag(eigvals_Q) @ eigvecs_Q.T

    eigvals_R, eigvecs_R = linalg.eigh(R)
    eigvals_R = np.maximum(eigvals_R, 1e-10)
    R = eigvecs_R @ np.diag(eigvals_R) @ eigvecs_R.T

    # Estimate full state sequence
    # X = C^+ @ (Y - D @ U)
    try:
        C_pinv = linalg.pinv(C)
        X = C_pinv @ (y - D @ u)
    except linalg.LinAlgError:
        X = np.zeros((n_states, n_samples))
        X[:, i:i + j - 1] = X_curr

    return A, B, C, D, X, Q, R, S


def subspace_id_unbiased(
    y: ArrayLike,
    n_states: int,
    u: Optional[ArrayLike] = None,
    horizon: int = 10,
) -> Tuple[np.ndarray, ...]:
    """
    Unbiased subspace identification.

    A variant that attempts to reduce bias in the estimates.

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_samples)
        Output data.
    n_states : int
        Number of states to identify.
    u : array_like, shape (n_inputs, n_samples), optional
        Input data.
    horizon : int, default=10
        Block size for Hankel matrices.

    Returns
    -------
    A, B, C, D, X, Q, R, S : ndarray
        System matrices, states, and noise covariances.
    """
    # Start with basic subspace ID
    A, B, C, D, X, Q, R, S = subspace_id(y, n_states, u, horizon)

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n_samples = y.shape[1]

    if u is None:
        u = np.zeros((1, n_samples))
    else:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))

    # Refine estimates using full data
    # Re-estimate states using Kalman filter with initial estimates
    from .kalman import kalman_smoother

    x_smooth, P_smooth, Pt, _, _, _ = kalman_smoother(
        y, A, C, Q, R, B=B, D=D, u=u,
    )

    # Re-estimate A, B, C, D from smoothed states
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_outputs = C.shape[0]

    # Build regression matrices
    # For state equation: x_{k+1} = A @ x_k + B @ u_k + w
    X_curr = x_smooth[:, :-1]
    X_next = x_smooth[:, 1:]
    U_curr = u[:, :-1]

    state_regressors = np.vstack([X_curr, U_curr])
    try:
        AB = linalg.lstsq(state_regressors.T, X_next.T)[0].T
    except linalg.LinAlgError:
        AB = X_next @ linalg.pinv(state_regressors)

    A = AB[:, :n_states]
    B = AB[:, n_states:]

    # For observation equation: y_k = C @ x_k + D @ u_k + v
    obs_regressors = np.vstack([x_smooth, u])
    try:
        CD = linalg.lstsq(obs_regressors.T, y.T)[0].T
    except linalg.LinAlgError:
        CD = y @ linalg.pinv(obs_regressors)

    C = CD[:, :n_states]
    D = CD[:, n_states:]

    # Re-estimate noise covariances
    x_pred = A @ X_curr + B @ U_curr
    state_residuals = X_next - x_pred

    y_pred = C @ x_smooth + D @ u
    output_residuals = y - y_pred

    n_residuals = state_residuals.shape[1]

    Q = state_residuals @ state_residuals.T / n_residuals
    R = output_residuals @ output_residuals.T / n_samples
    S = state_residuals @ output_residuals[:, :-1].T / n_residuals

    # Ensure positive semi-definite
    Q = (Q + Q.T) / 2
    R = (R + R.T) / 2

    eigvals_Q, eigvecs_Q = linalg.eigh(Q)
    eigvals_Q = np.maximum(eigvals_Q, 1e-10)
    Q = eigvecs_Q @ np.diag(eigvals_Q) @ eigvecs_Q.T

    eigvals_R, eigvecs_R = linalg.eigh(R)
    eigvals_R = np.maximum(eigvals_R, 1e-10)
    R = eigvecs_R @ np.diag(eigvals_R) @ eigvecs_R.T

    return A, B, C, D, x_smooth, Q, R, S


def estimate_transition_matrix(
    X: ArrayLike,
    U: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate transition matrix from state sequence.

    Given states X, estimates A (and optionally B) such that:
        x_{k+1} ≈ A @ x_k + B @ u_k

    Parameters
    ----------
    X : array_like, shape (n_states, n_samples)
        State sequence.
    U : array_like, shape (n_inputs, n_samples), optional
        Input sequence.

    Returns
    -------
    A : ndarray, shape (n_states, n_states)
        Estimated transition matrix.
    B : ndarray, shape (n_states, n_inputs)
        Estimated input matrix (zeros if U is None).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n_states, n_samples = X.shape

    X_curr = X[:, :-1]
    X_next = X[:, 1:]

    if U is not None:
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        U_curr = U[:, :-1]
        n_inputs = U.shape[0]

        regressors = np.vstack([X_curr, U_curr])
        try:
            AB = linalg.lstsq(regressors.T, X_next.T)[0].T
        except linalg.LinAlgError:
            AB = X_next @ linalg.pinv(regressors)

        A = AB[:, :n_states]
        B = AB[:, n_states:]
    else:
        try:
            A = linalg.lstsq(X_curr.T, X_next.T)[0].T
        except linalg.LinAlgError:
            A = X_next @ linalg.pinv(X_curr)

        B = np.zeros((n_states, 1))

    return A, B


def fit_matrix_powers(
    data: ArrayLike,
    n_powers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit data to matrix powers: data[k] ≈ C @ A^k @ x0.

    Useful for estimating A from output correlations or impulse responses.

    Parameters
    ----------
    data : array_like, shape (n_outputs, n_powers)
        Data assumed to follow matrix power structure.
    n_powers : int
        Number of matrix powers to fit.

    Returns
    -------
    A : ndarray
        Estimated base matrix.
    C : ndarray
        Estimated output matrix.
    """
    data = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n_outputs, n_samples = data.shape

    if n_powers > n_samples:
        n_powers = n_samples

    # Use Hankel matrix approach
    # H = [data[0], data[1], ..., data[n-1]]
    #     [data[1], data[2], ..., data[n]]
    #     ...

    n_rows = (n_samples + 1) // 2
    n_cols = n_samples - n_rows + 1

    if n_cols < 2:
        raise ValueError("Not enough data for matrix power fitting")

    H = hankel_matrix(data, n_rows, n_cols)

    # SVD to get low-rank approximation
    U, S, Vh = linalg.svd(H, full_matrices=False)

    # Estimate rank from singular values
    # Use first n_powers components
    rank = min(n_powers, len(S), n_rows, n_cols)

    U1 = U[:, :rank]
    S1 = S[:rank]
    V1 = Vh[:rank, :]

    # Extract observability matrix and state sequence
    Gamma = U1 @ np.diag(np.sqrt(S1))

    # C is the first block of Gamma
    C = Gamma[:n_outputs, :]

    # A from shift property of Gamma
    Gamma_up = Gamma[:-n_outputs, :]
    Gamma_down = Gamma[n_outputs:, :]

    try:
        A = linalg.lstsq(Gamma_up, Gamma_down)[0]
    except linalg.LinAlgError:
        A = linalg.pinv(Gamma_up) @ Gamma_down

    return A, C

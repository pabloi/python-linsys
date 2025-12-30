"""
Kalman filtering and smoothing algorithms.

This module implements:
- Standard Kalman filter (forward pass)
- Rauch-Tung-Striebel (RTS) smoother (backward pass)
- Steady-state (stationary) variants for efficiency
- Support for missing data (NaN values)

Based on the MATLAB matlab-linsys toolbox.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg


def kalman_predict(
    x: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    B: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman filter prediction step.

    Computes the predicted state and covariance:
        x_pred = A @ x + B @ u
        P_pred = A @ P @ A.T + Q

    Parameters
    ----------
    x : ndarray, shape (n_states,)
        Current state estimate.
    P : ndarray, shape (n_states, n_states)
        Current state covariance.
    A : ndarray, shape (n_states, n_states)
        State transition matrix.
    Q : ndarray, shape (n_states, n_states)
        Process noise covariance.
    B : ndarray, shape (n_states, n_inputs), optional
        Input matrix.
    u : ndarray, shape (n_inputs,), optional
        Input vector.

    Returns
    -------
    x_pred : ndarray, shape (n_states,)
        Predicted state.
    P_pred : ndarray, shape (n_states, n_states)
        Predicted covariance.
    """
    x_pred = A @ x
    if B is not None and u is not None:
        x_pred = x_pred + B @ u

    P_pred = A @ P @ A.T + Q

    return x_pred, P_pred


def kalman_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
    D: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Kalman filter update step.

    Computes the updated (filtered) state and covariance given an observation.

    Parameters
    ----------
    x_pred : ndarray, shape (n_states,)
        Predicted state.
    P_pred : ndarray, shape (n_states, n_states)
        Predicted covariance.
    y : ndarray, shape (n_outputs,)
        Observation vector (may contain NaN for missing data).
    C : ndarray, shape (n_outputs, n_states)
        Observation matrix.
    R : ndarray, shape (n_outputs, n_outputs)
        Measurement noise covariance.
    D : ndarray, shape (n_outputs, n_inputs), optional
        Feedthrough matrix.
    u : ndarray, shape (n_inputs,), optional
        Input vector.

    Returns
    -------
    x_filt : ndarray, shape (n_states,)
        Filtered state.
    P_filt : ndarray, shape (n_states, n_states)
        Filtered covariance.
    K : ndarray, shape (n_states, n_outputs)
        Kalman gain.
    innovation : ndarray, shape (n_outputs,)
        Innovation (prediction error).
    log_lik : float
        Log-likelihood contribution of this observation.
    """
    n_outputs = C.shape[0]

    # Predicted observation
    y_pred = C @ x_pred
    if D is not None and u is not None:
        y_pred = y_pred + D @ u

    # Handle missing data
    valid_mask = ~np.isnan(y)
    n_valid = np.sum(valid_mask)

    if n_valid == 0:
        # All missing - no update
        return x_pred.copy(), P_pred.copy(), np.zeros((len(x_pred), n_outputs)), np.full(n_outputs, np.nan), 0.0

    if n_valid < n_outputs:
        # Partial observation - reduce dimensions
        y_valid = y[valid_mask]
        C_valid = C[valid_mask, :]
        R_valid = R[np.ix_(valid_mask, valid_mask)]
        y_pred_valid = y_pred[valid_mask]
    else:
        y_valid = y
        C_valid = C
        R_valid = R
        y_pred_valid = y_pred

    # Innovation
    innovation_valid = y_valid - y_pred_valid

    # Innovation covariance
    S = C_valid @ P_pred @ C_valid.T + R_valid

    # Kalman gain via solving S @ K.T = C @ P
    try:
        # Use Cholesky for numerical stability
        L = linalg.cholesky(S, lower=True)
        K_valid = linalg.cho_solve((L, True), C_valid @ P_pred).T
    except linalg.LinAlgError:
        # Fall back to pseudo-inverse
        K_valid = P_pred @ C_valid.T @ linalg.pinv(S)

    # Update state
    x_filt = x_pred + K_valid @ innovation_valid

    # Update covariance (Joseph form for numerical stability)
    I_KC = np.eye(len(x_pred)) - K_valid @ C_valid
    P_filt = I_KC @ P_pred @ I_KC.T + K_valid @ R_valid @ K_valid.T

    # Ensure symmetry
    P_filt = (P_filt + P_filt.T) / 2

    # Log-likelihood
    try:
        log_lik = _log_likelihood_normal(innovation_valid, S)
    except (linalg.LinAlgError, ValueError):
        log_lik = -np.inf

    # Reconstruct full-sized outputs
    innovation = np.full(n_outputs, np.nan)
    innovation[valid_mask] = innovation_valid

    K = np.zeros((len(x_pred), n_outputs))
    K[:, valid_mask] = K_valid

    return x_filt, P_filt, K, innovation, log_lik


def _log_likelihood_normal(
    x: np.ndarray,
    cov: np.ndarray,
) -> float:
    """
    Compute log-likelihood of x under multivariate normal with given covariance.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Data vector.
    cov : ndarray, shape (n, n)
        Covariance matrix.

    Returns
    -------
    log_lik : float
        Log-likelihood.
    """
    n = len(x)
    try:
        L = linalg.cholesky(cov, lower=True)
        # log|cov| = 2 * sum(log(diag(L)))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        # x.T @ cov^{-1} @ x = ||L^{-1} @ x||^2
        z = linalg.solve_triangular(L, x, lower=True)
        mahal_sq = np.dot(z, z)
    except linalg.LinAlgError:
        # Singular covariance - use pseudo-inverse
        eigvals, eigvecs = linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        log_det = np.sum(np.log(eigvals))
        z = eigvecs.T @ x
        mahal_sq = np.sum(z ** 2 / eigvals)

    log_lik = -0.5 * (n * np.log(2 * np.pi) + log_det + mahal_sq)
    return log_lik


def kalman_filter(
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
    steady_state: bool = True,
    steady_state_tol: float = 1e-6,
    steady_state_window: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Kalman filter for linear state-space models.

    Implements the forward pass of the Kalman filter, optionally using
    steady-state gains for efficiency after convergence.

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_steps)
        Observations. NaN values are treated as missing data.
    A : array_like, shape (n_states, n_states)
        State transition matrix.
    C : array_like, shape (n_outputs, n_states)
        Observation matrix.
    Q : array_like, shape (n_states, n_states)
        Process noise covariance.
    R : array_like, shape (n_outputs, n_outputs)
        Measurement noise covariance.
    B : array_like, shape (n_states, n_inputs), optional
        Input matrix.
    D : array_like, shape (n_outputs, n_inputs), optional
        Feedthrough matrix.
    u : array_like, shape (n_inputs, n_steps), optional
        Inputs.
    x0 : array_like, shape (n_states,), optional
        Initial state. Defaults to zeros.
    P0 : array_like, shape (n_states, n_states), optional
        Initial covariance. Defaults to Q.
    steady_state : bool, default=True
        Whether to use steady-state Kalman gain after convergence.
    steady_state_tol : float, default=1e-6
        Tolerance for steady-state convergence.
    steady_state_window : int, default=20
        Number of steps to check for steady-state.

    Returns
    -------
    x_filt : ndarray, shape (n_states, n_steps)
        Filtered state estimates.
    P_filt : ndarray, shape (n_states, n_states, n_steps)
        Filtered state covariances.
    x_pred : ndarray, shape (n_states, n_steps)
        One-step-ahead predicted states.
    P_pred : ndarray, shape (n_states, n_states, n_steps)
        One-step-ahead predicted covariances.
    log_lik : float
        Total log-likelihood.
    """
    # Convert inputs
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))

    n_outputs, n_steps = y.shape
    n_states = A.shape[0]

    if B is not None:
        B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    if D is not None:
        D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    if u is not None:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))

    # Initialize state
    if x0 is None:
        x0 = np.zeros(n_states)
    else:
        x0 = np.asarray(x0, dtype=np.float64).ravel()

    if P0 is None:
        P0 = Q.copy()
    else:
        P0 = np.atleast_2d(np.asarray(P0, dtype=np.float64))

    # Allocate output arrays
    x_filt = np.zeros((n_states, n_steps))
    P_filt = np.zeros((n_states, n_states, n_steps))
    x_pred = np.zeros((n_states, n_steps))
    P_pred = np.zeros((n_states, n_states, n_steps))

    # Initialize
    x_curr = x0.copy()
    P_curr = P0.copy()
    log_lik = 0.0

    # Steady-state detection
    use_steady_state = False
    K_ss = None
    P_ss = None

    for k in range(n_steps):
        # Get input for this step
        u_k = u[:, k] if u is not None else None

        # Prediction
        x_p, P_p = kalman_predict(x_curr, P_curr, A, Q, B, u_k)
        x_pred[:, k] = x_p
        P_pred[:, :, k] = P_p

        # Update
        if use_steady_state and K_ss is not None:
            # Use steady-state gain
            y_p = C @ x_p
            if D is not None and u_k is not None:
                y_p = y_p + D @ u_k

            valid_mask = ~np.isnan(y[:, k])
            if np.all(valid_mask):
                innovation = y[:, k] - y_p
                x_f = x_p + K_ss @ innovation
                P_f = P_ss.copy()

                # Compute log-likelihood
                S = C @ P_p @ C.T + R
                try:
                    log_lik += _log_likelihood_normal(innovation, S)
                except (linalg.LinAlgError, ValueError):
                    pass
            else:
                # Missing data - full update
                x_f, P_f, _, _, ll = kalman_update(x_p, P_p, y[:, k], C, R, D, u_k)
                log_lik += ll
        else:
            x_f, P_f, K, innovation, ll = kalman_update(x_p, P_p, y[:, k], C, R, D, u_k)
            log_lik += ll

            # Check for steady-state convergence
            if steady_state and not use_steady_state and k >= steady_state_window:
                P_diff = np.max(np.abs(P_f - P_filt[:, :, k - 1]))
                if P_diff < steady_state_tol:
                    use_steady_state = True
                    K_ss = K
                    P_ss = P_f.copy()

        x_filt[:, k] = x_f
        P_filt[:, :, k] = P_f
        x_curr = x_f
        P_curr = P_f

    return x_filt, P_filt, x_pred, P_pred, log_lik


def kalman_smoother(
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
    **filter_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Kalman smoother (Rauch-Tung-Striebel algorithm).

    Combines forward filtering with backward smoothing to obtain
    optimal state estimates given all observations.

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_steps)
        Observations. NaN values are treated as missing data.
    A : array_like, shape (n_states, n_states)
        State transition matrix.
    C : array_like, shape (n_outputs, n_states)
        Observation matrix.
    Q : array_like, shape (n_states, n_states)
        Process noise covariance.
    R : array_like, shape (n_outputs, n_outputs)
        Measurement noise covariance.
    B : array_like, shape (n_states, n_inputs), optional
        Input matrix.
    D : array_like, shape (n_outputs, n_inputs), optional
        Feedthrough matrix.
    u : array_like, shape (n_inputs, n_steps), optional
        Inputs.
    x0 : array_like, shape (n_states,), optional
        Initial state.
    P0 : array_like, shape (n_states, n_states), optional
        Initial covariance.
    **filter_kwargs
        Additional arguments passed to kalman_filter.

    Returns
    -------
    x_smooth : ndarray, shape (n_states, n_steps)
        Smoothed state estimates.
    P_smooth : ndarray, shape (n_states, n_states, n_steps)
        Smoothed state covariances.
    Pt : ndarray, shape (n_states, n_states, n_steps-1)
        Cross-covariances E[x_k @ x_{k+1}.T | Y_{1:T}].
    x_filt : ndarray, shape (n_states, n_steps)
        Filtered state estimates (from forward pass).
    P_filt : ndarray, shape (n_states, n_states, n_steps)
        Filtered state covariances (from forward pass).
    log_lik : float
        Total log-likelihood.
    """
    # Forward pass
    x_filt, P_filt, x_pred, P_pred, log_lik = kalman_filter(
        y, A, C, Q, R, B=B, D=D, u=u, x0=x0, P0=P0, **filter_kwargs
    )

    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    n_states, n_steps = x_filt.shape

    # Allocate smoothed estimates
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    Pt = np.zeros((n_states, n_states, n_steps - 1))

    # Initialize at last time step
    x_smooth[:, -1] = x_filt[:, -1]
    P_smooth[:, :, -1] = P_filt[:, :, -1]

    # Backward pass
    for k in range(n_steps - 2, -1, -1):
        # Smoother gain
        # J_k = P_filt[k] @ A.T @ inv(P_pred[k+1])
        try:
            J = linalg.solve(P_pred[:, :, k + 1].T, (P_filt[:, :, k] @ A.T).T).T
        except linalg.LinAlgError:
            J = P_filt[:, :, k] @ A.T @ linalg.pinv(P_pred[:, :, k + 1])

        # Smoothed state
        x_smooth[:, k] = x_filt[:, k] + J @ (x_smooth[:, k + 1] - x_pred[:, k + 1])

        # Smoothed covariance
        P_smooth[:, :, k] = P_filt[:, :, k] + J @ (P_smooth[:, :, k + 1] - P_pred[:, :, k + 1]) @ J.T

        # Ensure symmetry
        P_smooth[:, :, k] = (P_smooth[:, :, k] + P_smooth[:, :, k].T) / 2

        # Cross-covariance (for EM algorithm)
        # Pt[k] = E[x_{k+1} @ x_k.T | Y] = (I - K_{k+1} C) A P_filt[k] + smoother term
        Pt[:, :, k] = P_smooth[:, :, k + 1] @ J.T

    return x_smooth, P_smooth, Pt, x_filt, P_filt, log_lik


def rts_smooth_step(
    x_filt: np.ndarray,
    P_filt: np.ndarray,
    x_smooth_next: np.ndarray,
    P_smooth_next: np.ndarray,
    x_pred_next: np.ndarray,
    P_pred_next: np.ndarray,
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single step of RTS smoother.

    Parameters
    ----------
    x_filt : ndarray, shape (n_states,)
        Filtered state at time k.
    P_filt : ndarray, shape (n_states, n_states)
        Filtered covariance at time k.
    x_smooth_next : ndarray, shape (n_states,)
        Smoothed state at time k+1.
    P_smooth_next : ndarray, shape (n_states, n_states)
        Smoothed covariance at time k+1.
    x_pred_next : ndarray, shape (n_states,)
        Predicted state at time k+1.
    P_pred_next : ndarray, shape (n_states, n_states)
        Predicted covariance at time k+1.
    A : ndarray, shape (n_states, n_states)
        State transition matrix.

    Returns
    -------
    x_smooth : ndarray, shape (n_states,)
        Smoothed state at time k.
    P_smooth : ndarray, shape (n_states, n_states)
        Smoothed covariance at time k.
    J : ndarray, shape (n_states, n_states)
        Smoother gain.
    """
    # Smoother gain
    try:
        J = linalg.solve(P_pred_next.T, (P_filt @ A.T).T).T
    except linalg.LinAlgError:
        J = P_filt @ A.T @ linalg.pinv(P_pred_next)

    # Smoothed estimates
    x_smooth = x_filt + J @ (x_smooth_next - x_pred_next)
    P_smooth = P_filt + J @ (P_smooth_next - P_pred_next) @ J.T

    # Ensure symmetry
    P_smooth = (P_smooth + P_smooth.T) / 2

    return x_smooth, P_smooth, J


def info_filter(
    y: ArrayLike,
    A: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    B: Optional[ArrayLike] = None,
    D: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
    xi0: Optional[ArrayLike] = None,
    Omega0: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Information filter (inverse covariance formulation of Kalman filter).

    This formulation can be more numerically stable when the prior
    covariance is very large (improper prior).

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_steps)
        Observations.
    A : array_like, shape (n_states, n_states)
        State transition matrix.
    C : array_like, shape (n_outputs, n_states)
        Observation matrix.
    Q : array_like, shape (n_states, n_states)
        Process noise covariance.
    R : array_like, shape (n_outputs, n_outputs)
        Measurement noise covariance.
    B : array_like, shape (n_states, n_inputs), optional
        Input matrix.
    D : array_like, shape (n_outputs, n_inputs), optional
        Feedthrough matrix.
    u : array_like, shape (n_inputs, n_steps), optional
        Inputs.
    xi0 : array_like, shape (n_states,), optional
        Initial information state (Omega0 @ x0).
    Omega0 : array_like, shape (n_states, n_states), optional
        Initial information matrix (inverse of P0).

    Returns
    -------
    x_filt : ndarray, shape (n_states, n_steps)
        Filtered state estimates.
    P_filt : ndarray, shape (n_states, n_states, n_steps)
        Filtered state covariances.
    log_lik : float
        Total log-likelihood.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))

    n_outputs, n_steps = y.shape
    n_states = A.shape[0]

    if B is not None:
        B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    if D is not None:
        D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    if u is not None:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))

    # Initialize information state
    if Omega0 is None:
        Omega0 = np.zeros((n_states, n_states))  # Improper prior
    else:
        Omega0 = np.atleast_2d(np.asarray(Omega0, dtype=np.float64))

    if xi0 is None:
        xi0 = np.zeros(n_states)
    else:
        xi0 = np.asarray(xi0, dtype=np.float64).ravel()

    # Precompute inverses
    try:
        Q_inv = linalg.inv(Q)
    except linalg.LinAlgError:
        Q_inv = linalg.pinv(Q)

    try:
        R_inv = linalg.inv(R)
    except linalg.LinAlgError:
        R_inv = linalg.pinv(R)

    # Information from observations
    info_C = C.T @ R_inv @ C
    info_y_base = C.T @ R_inv

    # Allocate outputs
    x_filt = np.zeros((n_states, n_steps))
    P_filt = np.zeros((n_states, n_states, n_steps))

    Omega = Omega0.copy()
    xi = xi0.copy()
    log_lik = 0.0

    for k in range(n_steps):
        u_k = u[:, k] if u is not None else None

        # Prediction in information form
        try:
            A_inv = linalg.inv(A)
        except linalg.LinAlgError:
            A_inv = linalg.pinv(A)

        M = A_inv.T @ Omega @ A_inv
        L = np.eye(n_states) + M @ Q

        try:
            L_inv = linalg.inv(L)
        except linalg.LinAlgError:
            L_inv = linalg.pinv(L)

        Omega_pred = L_inv @ M
        xi_pred = L_inv @ A_inv.T @ xi
        if B is not None and u_k is not None:
            xi_pred = xi_pred + Omega_pred @ B @ u_k

        # Update
        y_k = y[:, k]
        valid_mask = ~np.isnan(y_k)
        n_valid = np.sum(valid_mask)

        if n_valid > 0:
            y_valid = y_k[valid_mask]
            C_valid = C[valid_mask, :]
            R_inv_valid = R_inv[np.ix_(valid_mask, valid_mask)]

            y_eff = y_valid.copy()
            if D is not None and u_k is not None:
                y_eff = y_eff - D[valid_mask, :] @ u_k

            Omega = Omega_pred + C_valid.T @ R_inv_valid @ C_valid
            xi = xi_pred + C_valid.T @ R_inv_valid @ y_eff
        else:
            Omega = Omega_pred
            xi = xi_pred

        # Convert to state space
        try:
            P = linalg.inv(Omega)
            x = P @ xi
        except linalg.LinAlgError:
            P = linalg.pinv(Omega)
            x = P @ xi

        x_filt[:, k] = x
        P_filt[:, :, k] = P

        # Compute log-likelihood contribution
        if n_valid > 0:
            try:
                P_pred = linalg.inv(Omega_pred)
            except linalg.LinAlgError:
                P_pred = linalg.pinv(Omega_pred)

            x_pred = P_pred @ xi_pred
            y_pred_valid = C_valid @ x_pred
            if D is not None and u_k is not None:
                y_pred_valid = y_pred_valid + D[valid_mask, :] @ u_k

            innovation = y_valid - y_pred_valid
            S = C_valid @ P_pred @ C_valid.T + R[np.ix_(valid_mask, valid_mask)]
            try:
                log_lik += _log_likelihood_normal(innovation, S)
            except (linalg.LinAlgError, ValueError):
                pass

    return x_filt, P_filt, log_lik


def steady_state_kalman_gain(
    A: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute steady-state Kalman gain and covariances.

    Parameters
    ----------
    A : array_like, shape (n_states, n_states)
        State transition matrix.
    C : array_like, shape (n_outputs, n_states)
        Observation matrix.
    Q : array_like, shape (n_states, n_states)
        Process noise covariance.
    R : array_like, shape (n_outputs, n_outputs)
        Measurement noise covariance.
    max_iter : int, default=1000
        Maximum iterations for convergence.
    tol : float, default=1e-10
        Convergence tolerance.

    Returns
    -------
    K : ndarray, shape (n_states, n_outputs)
        Steady-state Kalman gain.
    P_pred : ndarray, shape (n_states, n_states)
        Steady-state predicted covariance.
    P_filt : ndarray, shape (n_states, n_states)
        Steady-state filtered covariance.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))

    n_states = A.shape[0]

    # Try solving DARE first
    try:
        P_pred = linalg.solve_discrete_are(A.T, C.T, Q, R)
    except linalg.LinAlgError:
        # Fall back to iteration
        P_pred = Q.copy()
        for _ in range(max_iter):
            S = C @ P_pred @ C.T + R
            try:
                K = linalg.solve(S.T, (P_pred @ C.T).T).T
            except linalg.LinAlgError:
                K = P_pred @ C.T @ linalg.pinv(S)

            P_filt = (np.eye(n_states) - K @ C) @ P_pred
            P_pred_new = A @ P_filt @ A.T + Q

            if np.max(np.abs(P_pred_new - P_pred)) < tol:
                P_pred = P_pred_new
                break
            P_pred = P_pred_new

    # Compute Kalman gain
    S = C @ P_pred @ C.T + R
    try:
        K = linalg.solve(S.T, (P_pred @ C.T).T).T
    except linalg.LinAlgError:
        K = P_pred @ C.T @ linalg.pinv(S)

    P_filt = (np.eye(n_states) - K @ C) @ P_pred

    return K, P_pred, P_filt

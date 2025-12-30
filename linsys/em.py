"""
Expectation-Maximization (EM) algorithm for linear system identification.

This module implements the EM algorithm for identifying LTI state-space
models from input-output data. The EM algorithm alternates between:
- E-step: Estimate latent states using Kalman smoothing
- M-step: Update model parameters to maximize expected log-likelihood

Based on the MATLAB matlab-linsys toolbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from .kalman import kalman_smoother


@dataclass
class EMResult:
    """
    Result of EM identification.

    Attributes
    ----------
    A : ndarray
        State transition matrix.
    B : ndarray
        Input matrix.
    C : ndarray
        Observation matrix.
    D : ndarray
        Feedthrough matrix.
    Q : ndarray
        Process noise covariance.
    R : ndarray
        Measurement noise covariance.
    x0 : ndarray
        Initial state estimate.
    P0 : ndarray
        Initial state covariance.
    x_smooth : ndarray
        Smoothed states from final iteration.
    P_smooth : ndarray
        Smoothed covariances from final iteration.
    log_likelihood : list
        Log-likelihood at each iteration.
    converged : bool
        Whether the algorithm converged.
    n_iterations : int
        Number of iterations performed.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray
    x_smooth: np.ndarray
    P_smooth: np.ndarray
    log_likelihood: List[float]
    converged: bool
    n_iterations: int


@dataclass
class EMOptions:
    """
    Options for EM algorithm.

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on log-likelihood improvement per dimension.
    min_improvement : float
        Minimum relative improvement to continue.
    fix_A : bool
        If True, don't update A.
    fix_B : bool
        If True, don't update B.
    fix_C : bool
        If True, don't update C.
    fix_D : bool
        If True, don't update D.
    fix_Q : bool
        If True, don't update Q.
    fix_R : bool
        If True, don't update R.
    fix_x0 : bool
        If True, don't update x0.
    fix_P0 : bool
        If True, don't update P0.
    diagonal_A : bool
        If True, constrain A to be diagonal.
    diagonal_Q : bool
        If True, constrain Q to be diagonal.
    diagonal_R : bool
        If True, constrain R to be diagonal.
    spherical_R : bool
        If True, constrain R to be scalar * I.
    stable : bool
        If True, enforce stability (|eig(A)| < 1).
    max_eig : float
        Maximum eigenvalue magnitude for stability.
    robust_Q : bool
        If True, use robust estimation for Q.
    verbose : bool
        If True, print progress.
    """

    max_iter: int = 500
    tol: float = 1e-6
    min_improvement: float = 1e-10
    fix_A: bool = False
    fix_B: bool = False
    fix_C: bool = False
    fix_D: bool = False
    fix_Q: bool = False
    fix_R: bool = False
    fix_x0: bool = False
    fix_P0: bool = False
    diagonal_A: bool = False
    diagonal_Q: bool = False
    diagonal_R: bool = False
    spherical_R: bool = False
    stable: bool = True
    max_eig: float = 0.999
    robust_Q: bool = False
    verbose: bool = False


def em_identify(
    y: ArrayLike,
    n_states: int,
    u: Optional[ArrayLike] = None,
    A0: Optional[ArrayLike] = None,
    B0: Optional[ArrayLike] = None,
    C0: Optional[ArrayLike] = None,
    D0: Optional[ArrayLike] = None,
    Q0: Optional[ArrayLike] = None,
    R0: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
    P0: Optional[ArrayLike] = None,
    opts: Optional[EMOptions] = None,
) -> "LinearSystem":
    """
    Identify a linear system using the EM algorithm.

    Parameters
    ----------
    y : array_like, shape (n_outputs, n_steps) or list of arrays
        Observations. Can be a list for multiple realizations.
    n_states : int
        Number of states to estimate.
    u : array_like, shape (n_inputs, n_steps), optional
        Inputs. Can be a list for multiple realizations.
    A0, B0, C0, D0, Q0, R0 : array_like, optional
        Initial parameter guesses.
    x0 : array_like, optional
        Initial state guess.
    P0 : array_like, optional
        Initial covariance guess.
    opts : EMOptions, optional
        Algorithm options.

    Returns
    -------
    sys : LinearSystem
        Identified linear system.
    """
    result = em_algorithm(
        y, n_states, u=u,
        A0=A0, B0=B0, C0=C0, D0=D0, Q0=Q0, R0=R0,
        x0_init=x0, P0_init=P0, opts=opts,
    )

    from .linear_system import LinearSystem
    return LinearSystem(
        A=result.A, B=result.B, C=result.C, D=result.D,
        Q=result.Q, R=result.R, x0=result.x0, P0=result.P0,
    )


def em_algorithm(
    y: ArrayLike,
    n_states: int,
    u: Optional[ArrayLike] = None,
    A0: Optional[ArrayLike] = None,
    B0: Optional[ArrayLike] = None,
    C0: Optional[ArrayLike] = None,
    D0: Optional[ArrayLike] = None,
    Q0: Optional[ArrayLike] = None,
    R0: Optional[ArrayLike] = None,
    x0_init: Optional[ArrayLike] = None,
    P0_init: Optional[ArrayLike] = None,
    opts: Optional[EMOptions] = None,
) -> EMResult:
    """
    Run the EM algorithm for system identification.

    Returns full results including convergence information.
    """
    if opts is None:
        opts = EMOptions()

    # Handle multiple realizations
    if isinstance(y, list):
        y_list = [np.atleast_2d(np.asarray(yi, dtype=np.float64)) for yi in y]
        n_outputs = y_list[0].shape[0]
        n_steps_list = [yi.shape[1] for yi in y_list]
    else:
        y = np.atleast_2d(np.asarray(y, dtype=np.float64))
        y_list = [y]
        n_outputs = y.shape[0]
        n_steps_list = [y.shape[1]]

    if u is not None:
        if isinstance(u, list):
            u_list = [np.atleast_2d(np.asarray(ui, dtype=np.float64)) for ui in u]
            n_inputs = u_list[0].shape[1] if u_list[0].ndim == 1 else u_list[0].shape[0]
        else:
            u = np.atleast_2d(np.asarray(u, dtype=np.float64))
            u_list = [u]
            n_inputs = u.shape[0]
    else:
        u_list = [np.zeros((1, n)) for n in n_steps_list]
        n_inputs = 1

    # Initialize parameters
    A, B, C, D, Q, R, x0, P0 = _initialize_parameters(
        n_states, n_inputs, n_outputs,
        A0, B0, C0, D0, Q0, R0, x0_init, P0_init,
        y_list, u_list,
    )

    log_likelihoods = []
    prev_ll = -np.inf

    for iteration in range(opts.max_iter):
        # E-step: Estimate states given current parameters
        x_smooth_list = []
        P_smooth_list = []
        Pt_list = []
        total_ll = 0.0

        for yi, ui in zip(y_list, u_list):
            xs, Ps, Pt, _, _, ll = kalman_smoother(
                yi, A, C, Q, R, B=B, D=D, u=ui, x0=x0, P0=P0,
            )
            x_smooth_list.append(xs)
            P_smooth_list.append(Ps)
            Pt_list.append(Pt)
            total_ll += ll

        log_likelihoods.append(total_ll)

        # Check convergence
        n_total = sum(n_steps_list) * n_outputs
        ll_improvement = (total_ll - prev_ll) / n_total

        if opts.verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: log-likelihood = {total_ll:.4f}, "
                  f"improvement = {ll_improvement:.2e}")

        if iteration > 0:
            if ll_improvement < opts.tol:
                if opts.verbose:
                    print(f"Converged at iteration {iteration}")
                return EMResult(
                    A=A, B=B, C=C, D=D, Q=Q, R=R, x0=x0, P0=P0,
                    x_smooth=x_smooth_list[0] if len(x_smooth_list) == 1 else x_smooth_list,
                    P_smooth=P_smooth_list[0] if len(P_smooth_list) == 1 else P_smooth_list,
                    log_likelihood=log_likelihoods,
                    converged=True,
                    n_iterations=iteration + 1,
                )

            if ll_improvement < -opts.min_improvement:
                # Log-likelihood decreased significantly - something wrong
                if opts.verbose:
                    print(f"Warning: Log-likelihood decreased at iteration {iteration}")

        prev_ll = total_ll

        # M-step: Update parameters
        A, B, C, D, Q, R, x0, P0 = _m_step(
            y_list, u_list, x_smooth_list, P_smooth_list, Pt_list,
            A, B, C, D, Q, R, x0, P0, opts,
        )

    if opts.verbose:
        print(f"Maximum iterations ({opts.max_iter}) reached")

    return EMResult(
        A=A, B=B, C=C, D=D, Q=Q, R=R, x0=x0, P0=P0,
        x_smooth=x_smooth_list[0] if len(x_smooth_list) == 1 else x_smooth_list,
        P_smooth=P_smooth_list[0] if len(P_smooth_list) == 1 else P_smooth_list,
        log_likelihood=log_likelihoods,
        converged=False,
        n_iterations=opts.max_iter,
    )


def _initialize_parameters(
    n_states: int,
    n_inputs: int,
    n_outputs: int,
    A0: Optional[ArrayLike],
    B0: Optional[ArrayLike],
    C0: Optional[ArrayLike],
    D0: Optional[ArrayLike],
    Q0: Optional[ArrayLike],
    R0: Optional[ArrayLike],
    x0_init: Optional[ArrayLike],
    P0_init: Optional[ArrayLike],
    y_list: List[np.ndarray],
    u_list: List[np.ndarray],
) -> Tuple[np.ndarray, ...]:
    """Initialize EM parameters with sensible defaults."""

    # State transition matrix
    if A0 is not None:
        A = np.atleast_2d(np.asarray(A0, dtype=np.float64))
    else:
        # Random stable A
        A = np.random.randn(n_states, n_states) * 0.5
        eigenvalues = linalg.eigvals(A)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0.95:
            A = A * (0.95 / max_eig)

    # Input matrix
    if B0 is not None:
        B = np.atleast_2d(np.asarray(B0, dtype=np.float64))
    else:
        B = np.zeros((n_states, n_inputs))

    # Observation matrix
    if C0 is not None:
        C = np.atleast_2d(np.asarray(C0, dtype=np.float64))
    else:
        # Initialize from PCA of observations
        y_all = np.concatenate(y_list, axis=1)
        y_centered = y_all - np.mean(y_all, axis=1, keepdims=True)

        if n_states <= min(y_all.shape):
            U, S, Vh = linalg.svd(y_centered, full_matrices=False)
            C = U[:, :n_states] @ np.diag(np.sqrt(S[:n_states]))
        else:
            C = np.random.randn(n_outputs, n_states)

    # Feedthrough matrix
    if D0 is not None:
        D = np.atleast_2d(np.asarray(D0, dtype=np.float64))
    else:
        D = np.zeros((n_outputs, n_inputs))

    # Process noise covariance
    if Q0 is not None:
        Q = np.atleast_2d(np.asarray(Q0, dtype=np.float64))
    else:
        Q = np.eye(n_states) * 0.1

    # Measurement noise covariance
    if R0 is not None:
        R = np.atleast_2d(np.asarray(R0, dtype=np.float64))
    else:
        # Estimate from observation variance
        y_all = np.concatenate(y_list, axis=1)
        R = np.diag(np.var(y_all, axis=1)) * 0.5

    # Initial state
    if x0_init is not None:
        x0 = np.asarray(x0_init, dtype=np.float64).ravel()
    else:
        x0 = np.zeros(n_states)

    # Initial covariance
    if P0_init is not None:
        P0 = np.atleast_2d(np.asarray(P0_init, dtype=np.float64))
    else:
        P0 = np.eye(n_states) * 10

    return A, B, C, D, Q, R, x0, P0


def _m_step(
    y_list: List[np.ndarray],
    u_list: List[np.ndarray],
    x_smooth_list: List[np.ndarray],
    P_smooth_list: List[np.ndarray],
    Pt_list: List[np.ndarray],
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    opts: EMOptions,
) -> Tuple[np.ndarray, ...]:
    """
    M-step: Update parameters to maximize expected log-likelihood.

    Based on the sufficient statistics from the E-step.
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    n_outputs = C.shape[0]
    n_realizations = len(y_list)

    # Compute sufficient statistics across all realizations
    # E[x_k x_k^T], E[x_k x_{k-1}^T], E[x_k u_k^T], etc.

    # Aggregate statistics
    total_samples = 0
    sum_xx = np.zeros((n_states, n_states))
    sum_xx_prev = np.zeros((n_states, n_states))  # E[x_k x_k^T] for k < N
    sum_xx_next = np.zeros((n_states, n_states))  # E[x_k x_k^T] for k > 0
    sum_xx_cross = np.zeros((n_states, n_states))  # E[x_{k+1} x_k^T]
    sum_xu = np.zeros((n_states, n_inputs))
    sum_xu_prev = np.zeros((n_states, n_inputs))  # For k < N
    sum_uu = np.zeros((n_inputs, n_inputs))
    sum_uu_prev = np.zeros((n_inputs, n_inputs))

    sum_yx = np.zeros((n_outputs, n_states))
    sum_yu = np.zeros((n_outputs, n_inputs))
    sum_yy = np.zeros((n_outputs, n_outputs))

    sum_x0 = np.zeros(n_states)
    sum_P0 = np.zeros((n_states, n_states))

    for y, u, xs, Ps, Pt in zip(y_list, u_list, x_smooth_list, P_smooth_list, Pt_list):
        n_steps = y.shape[1]
        total_samples += n_steps

        for k in range(n_steps):
            # Expected outer products
            xx_k = Ps[:, :, k] + np.outer(xs[:, k], xs[:, k])
            sum_xx += xx_k

            if k < n_steps - 1:
                sum_xx_prev += xx_k
                sum_xu_prev += np.outer(xs[:, k], u[:, k])
                sum_uu_prev += np.outer(u[:, k], u[:, k])

            if k > 0:
                sum_xx_next += xx_k

            sum_xu += np.outer(xs[:, k], u[:, k])
            sum_uu += np.outer(u[:, k], u[:, k])

            # Handle missing data in y
            valid_mask = ~np.isnan(y[:, k])
            if np.any(valid_mask):
                y_valid = y[valid_mask, k]
                C_valid = C[valid_mask, :]
                D_valid = D[valid_mask, :]

                sum_yx[valid_mask, :] += np.outer(y_valid, xs[:, k])
                sum_yu[valid_mask, :] += np.outer(y_valid, u[:, k])
                sum_yy[np.ix_(valid_mask, valid_mask)] += np.outer(y_valid, y_valid)

        # Cross-covariance terms
        for k in range(n_steps - 1):
            # Pt[:, :, k] = E[x_{k+1} x_k^T | Y]
            sum_xx_cross += Pt[:, :, k] + np.outer(xs[:, k + 1], xs[:, k])

        # Initial state statistics
        sum_x0 += xs[:, 0]
        sum_P0 += Ps[:, :, 0] + np.outer(xs[:, 0], xs[:, 0])

    # Update parameters
    n_prev = total_samples - n_realizations  # Number of k < N transitions

    # Update A and B (state equation)
    if not opts.fix_A or not opts.fix_B:
        # Solve: [A B] @ [[sum_xx_prev, sum_xu_prev], [sum_xu_prev.T, sum_uu_prev]] = [sum_xx_cross, sum_xu_next?]
        # This is: [A B] @ M = N where M is state+input covariance

        # Build regression problem for state equation
        lhs = sum_xx_cross  # E[x_{k+1} x_k^T] summed
        rhs_xx = sum_xx_prev
        rhs_xu = sum_xu_prev
        rhs_uu = sum_uu_prev

        if not opts.fix_A and not opts.fix_B:
            # Solve for both A and B
            M = np.block([
                [rhs_xx, rhs_xu],
                [rhs_xu.T, rhs_uu]
            ])

            # Compute sum of x_{k+1} u_k^T for B estimation
            sum_xu_next = np.zeros((n_states, n_inputs))
            for y, u, xs in zip(y_list, u_list, x_smooth_list):
                n_steps = y.shape[1]
                for k in range(n_steps - 1):
                    sum_xu_next += np.outer(xs[:, k + 1], u[:, k])

            N = np.hstack([lhs, sum_xu_next])

            try:
                AB = linalg.solve(M.T, N.T).T
            except linalg.LinAlgError:
                AB = N @ linalg.pinv(M)

            A_new = AB[:, :n_states]
            B_new = AB[:, n_states:]

            if not opts.fix_A:
                A = A_new
            if not opts.fix_B:
                B = B_new

        elif not opts.fix_A:
            # Only update A
            try:
                A = linalg.solve(rhs_xx.T, lhs.T).T
            except linalg.LinAlgError:
                A = lhs @ linalg.pinv(rhs_xx)

        # Apply diagonal constraint
        if opts.diagonal_A and not opts.fix_A:
            A = np.diag(np.diag(A))

        # Enforce stability
        if opts.stable and not opts.fix_A:
            eigenvalues, eigenvectors = linalg.eig(A)
            max_eig = np.max(np.abs(eigenvalues))
            if max_eig > opts.max_eig:
                # Scale eigenvalues
                scale = opts.max_eig / max_eig
                eigenvalues_scaled = eigenvalues * scale
                A = (eigenvectors @ np.diag(eigenvalues_scaled) @ linalg.inv(eigenvectors)).real

    # Update C and D (observation equation)
    if not opts.fix_C or not opts.fix_D:
        if not opts.fix_C and not opts.fix_D:
            M = np.block([
                [sum_xx, sum_xu],
                [sum_xu.T, sum_uu]
            ])
            N = np.hstack([sum_yx, sum_yu])

            try:
                CD = linalg.solve(M.T, N.T).T
            except linalg.LinAlgError:
                CD = N @ linalg.pinv(M)

            C_new = CD[:, :n_states]
            D_new = CD[:, n_states:]

            if not opts.fix_C:
                C = C_new
            if not opts.fix_D:
                D = D_new

        elif not opts.fix_C:
            try:
                C = linalg.solve(sum_xx.T, sum_yx.T).T
            except linalg.LinAlgError:
                C = sum_yx @ linalg.pinv(sum_xx)

    # Update Q (process noise covariance)
    if not opts.fix_Q:
        # Q = E[(x_{k+1} - A x_k - B u_k)(x_{k+1} - A x_k - B u_k)^T]
        Q_sum = np.zeros((n_states, n_states))

        for y, u, xs, Ps, Pt in zip(y_list, u_list, x_smooth_list, P_smooth_list, Pt_list):
            n_steps = y.shape[1]
            for k in range(n_steps - 1):
                xx_k = Ps[:, :, k] + np.outer(xs[:, k], xs[:, k])
                xx_kp1 = Ps[:, :, k + 1] + np.outer(xs[:, k + 1], xs[:, k + 1])
                xx_cross = Pt[:, :, k] + np.outer(xs[:, k + 1], xs[:, k])

                # E[(x_{k+1} - A x_k - B u_k)(...)^T]
                pred = A @ xs[:, k] + B @ u[:, k]
                res = xs[:, k + 1] - pred

                # Full covariance formula
                Q_k = (xx_kp1
                       - A @ xx_cross.T
                       - xx_cross @ A.T
                       + A @ xx_k @ A.T
                       - np.outer(B @ u[:, k], xs[:, k + 1])
                       - np.outer(xs[:, k + 1], B @ u[:, k])
                       + A @ np.outer(xs[:, k], u[:, k]) @ B.T
                       + B @ np.outer(u[:, k], xs[:, k]) @ A.T
                       + B @ np.outer(u[:, k], u[:, k]) @ B.T)
                Q_sum += Q_k

        Q = Q_sum / n_prev

        # Ensure positive semi-definite
        Q = (Q + Q.T) / 2
        eigvals, eigvecs = linalg.eigh(Q)
        eigvals = np.maximum(eigvals, 1e-10)
        Q = eigvecs @ np.diag(eigvals) @ eigvecs.T

        if opts.diagonal_Q:
            Q = np.diag(np.diag(Q))

    # Update R (measurement noise covariance)
    if not opts.fix_R:
        R_sum = np.zeros((n_outputs, n_outputs))
        n_obs = 0

        for y, u, xs, Ps in zip(y_list, u_list, x_smooth_list, P_smooth_list):
            n_steps = y.shape[1]
            for k in range(n_steps):
                valid_mask = ~np.isnan(y[:, k])
                if not np.any(valid_mask):
                    continue

                n_obs += np.sum(valid_mask)
                y_k = y[valid_mask, k]
                C_valid = C[valid_mask, :]
                D_valid = D[valid_mask, :]

                pred = C_valid @ xs[:, k] + D_valid @ u[:, k]
                res = y_k - pred

                # E[(y_k - C x_k - D u_k)(...)^T]
                xx_k = Ps[:, :, k] + np.outer(xs[:, k], xs[:, k])
                R_k = (np.outer(res, res)
                       + C_valid @ Ps[:, :, k] @ C_valid.T)
                R_sum[np.ix_(valid_mask, valid_mask)] += R_k

        R = R_sum / total_samples

        # Ensure positive semi-definite
        R = (R + R.T) / 2
        eigvals, eigvecs = linalg.eigh(R)
        eigvals = np.maximum(eigvals, 1e-10)
        R = eigvecs @ np.diag(eigvals) @ eigvecs.T

        if opts.diagonal_R:
            R = np.diag(np.diag(R))
        elif opts.spherical_R:
            R = np.eye(n_outputs) * np.mean(np.diag(R))

    # Update initial state
    if not opts.fix_x0:
        x0 = sum_x0 / n_realizations

    if not opts.fix_P0:
        P0 = sum_P0 / n_realizations - np.outer(x0, x0)
        P0 = (P0 + P0.T) / 2
        eigvals, eigvecs = linalg.eigh(P0)
        eigvals = np.maximum(eigvals, 1e-10)
        P0 = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return A, B, C, D, Q, R, x0, P0


def em_step(
    y: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    x0: ArrayLike,
    P0: ArrayLike,
    u: Optional[ArrayLike] = None,
    opts: Optional[EMOptions] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Perform a single EM iteration.

    Parameters
    ----------
    y : array_like
        Observations.
    A, B, C, D, Q, R : array_like
        Current parameter estimates.
    x0, P0 : array_like
        Current initial state estimates.
    u : array_like, optional
        Inputs.
    opts : EMOptions, optional
        Algorithm options.

    Returns
    -------
    A, B, C, D, Q, R, x0, P0 : ndarray
        Updated parameter estimates.
    log_lik : float
        Log-likelihood after E-step.
    """
    if opts is None:
        opts = EMOptions()

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n_steps = y.shape[1]

    if u is None:
        u = np.zeros((1, n_steps))
    else:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))

    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    C = np.atleast_2d(np.asarray(C, dtype=np.float64))
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    P0 = np.atleast_2d(np.asarray(P0, dtype=np.float64))

    # E-step
    xs, Ps, Pt, _, _, log_lik = kalman_smoother(
        y, A, C, Q, R, B=B, D=D, u=u, x0=x0, P0=P0,
    )

    # M-step
    A, B, C, D, Q, R, x0, P0 = _m_step(
        [y], [u], [xs], [Ps], [Pt],
        A, B, C, D, Q, R, x0, P0, opts,
    )

    return A, B, C, D, Q, R, x0, P0, log_lik


def random_start_em(
    y: ArrayLike,
    n_states: int,
    n_restarts: int = 10,
    u: Optional[ArrayLike] = None,
    opts: Optional[EMOptions] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> "LinearSystem":
    """
    Run EM from multiple random initializations and return the best result.

    Parameters
    ----------
    y : array_like
        Observations.
    n_states : int
        Number of states.
    n_restarts : int, default=10
        Number of random restarts.
    u : array_like, optional
        Inputs.
    opts : EMOptions, optional
        Algorithm options.
    rng : numpy.random.Generator, optional
        Random number generator.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    sys : LinearSystem
        Best identified system.
    """
    if rng is None:
        rng = np.random.default_rng()

    if opts is None:
        opts = EMOptions(verbose=False)
    else:
        opts = EMOptions(**{k: v for k, v in opts.__dict__.items()})
        opts.verbose = False

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n_outputs, n_steps = y.shape

    best_result = None
    best_ll = -np.inf

    for i in range(n_restarts):
        # Random initialization
        A0 = rng.standard_normal((n_states, n_states)) * 0.5
        eigenvalues = linalg.eigvals(A0)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0.9:
            A0 = A0 * (0.9 / max_eig)

        C0 = rng.standard_normal((n_outputs, n_states))

        try:
            result = em_algorithm(
                y, n_states, u=u,
                A0=A0, C0=C0,
                opts=opts,
            )

            final_ll = result.log_likelihood[-1]
            if final_ll > best_ll:
                best_ll = final_ll
                best_result = result

            if verbose:
                print(f"Restart {i + 1}/{n_restarts}: log-likelihood = {final_ll:.4f}")

        except (linalg.LinAlgError, ValueError) as e:
            if verbose:
                print(f"Restart {i + 1}/{n_restarts}: failed ({e})")

    if best_result is None:
        raise RuntimeError("All EM restarts failed")

    from .linear_system import LinearSystem
    return LinearSystem(
        A=best_result.A, B=best_result.B,
        C=best_result.C, D=best_result.D,
        Q=best_result.Q, R=best_result.R,
        x0=best_result.x0, P0=best_result.P0,
    )

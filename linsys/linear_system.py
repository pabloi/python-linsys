"""
Linear System class for state-space representation of LTI systems.

A linear time-invariant (LTI) state-space model is described by:
    x[k+1] = A @ x[k] + B @ u[k] + w[k]    (state equation)
    y[k]   = C @ x[k] + D @ u[k] + v[k]    (observation equation)

where:
    x[k] is the state vector (n_states,)
    u[k] is the input vector (n_inputs,)
    y[k] is the output vector (n_outputs,)
    w[k] ~ N(0, Q) is the process noise
    v[k] ~ N(0, R) is the measurement noise
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg


@dataclass
class LinearSystem:
    """
    Linear Time-Invariant State-Space Model.

    Parameters
    ----------
    A : array_like, shape (n_states, n_states)
        State transition matrix.
    B : array_like, shape (n_states, n_inputs), optional
        Input-to-state matrix. Defaults to zeros if not provided.
    C : array_like, shape (n_outputs, n_states)
        State-to-output matrix.
    D : array_like, shape (n_outputs, n_inputs), optional
        Input-to-output feedthrough matrix. Defaults to zeros if not provided.
    Q : array_like, shape (n_states, n_states), optional
        Process noise covariance. Defaults to identity if not provided.
    R : array_like, shape (n_outputs, n_outputs), optional
        Measurement noise covariance. Defaults to identity if not provided.
    x0 : array_like, shape (n_states,), optional
        Initial state estimate.
    P0 : array_like, shape (n_states, n_states), optional
        Initial state covariance.
    name : str, optional
        Model identifier.

    Attributes
    ----------
    n_states : int
        Number of states (order of the system).
    n_inputs : int
        Number of inputs.
    n_outputs : int
        Number of outputs.

    Examples
    --------
    >>> # Simple 2-state system
    >>> A = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> C = np.array([[1.0, 0.0]])
    >>> sys = LinearSystem(A=A, C=C)
    >>> sys.n_states
    2
    """

    A: np.ndarray
    C: np.ndarray
    B: Optional[np.ndarray] = None
    D: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    x0: Optional[np.ndarray] = None
    P0: Optional[np.ndarray] = None
    name: str = ""

    def __post_init__(self):
        """Validate and initialize matrices."""
        # Convert to numpy arrays
        self.A = np.atleast_2d(np.asarray(self.A, dtype=np.float64))
        self.C = np.atleast_2d(np.asarray(self.C, dtype=np.float64))

        n_states = self.A.shape[0]
        n_outputs = self.C.shape[0]

        # Validate A matrix
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got shape {self.A.shape}")

        # Validate C matrix
        if self.C.shape[1] != n_states:
            raise ValueError(
                f"C columns ({self.C.shape[1]}) must match A rows ({n_states})"
            )

        # Initialize B matrix
        if self.B is None:
            self.B = np.zeros((n_states, 1), dtype=np.float64)
        else:
            self.B = np.atleast_2d(np.asarray(self.B, dtype=np.float64))
            if self.B.shape[0] != n_states:
                raise ValueError(
                    f"B rows ({self.B.shape[0]}) must match A rows ({n_states})"
                )

        n_inputs = self.B.shape[1]

        # Initialize D matrix
        if self.D is None:
            self.D = np.zeros((n_outputs, n_inputs), dtype=np.float64)
        else:
            self.D = np.atleast_2d(np.asarray(self.D, dtype=np.float64))
            if self.D.shape != (n_outputs, n_inputs):
                raise ValueError(
                    f"D shape {self.D.shape} must be ({n_outputs}, {n_inputs})"
                )

        # Initialize Q matrix (process noise covariance)
        if self.Q is None:
            self.Q = np.eye(n_states, dtype=np.float64)
        else:
            self.Q = np.atleast_2d(np.asarray(self.Q, dtype=np.float64))
            if self.Q.shape != (n_states, n_states):
                raise ValueError(
                    f"Q shape {self.Q.shape} must be ({n_states}, {n_states})"
                )

        # Initialize R matrix (measurement noise covariance)
        if self.R is None:
            self.R = np.eye(n_outputs, dtype=np.float64)
        else:
            self.R = np.atleast_2d(np.asarray(self.R, dtype=np.float64))
            if self.R.shape != (n_outputs, n_outputs):
                raise ValueError(
                    f"R shape {self.R.shape} must be ({n_outputs}, {n_outputs})"
                )

        # Initialize x0 (initial state)
        if self.x0 is None:
            self.x0 = np.zeros(n_states, dtype=np.float64)
        else:
            self.x0 = np.asarray(self.x0, dtype=np.float64).ravel()
            if self.x0.shape[0] != n_states:
                raise ValueError(
                    f"x0 length ({self.x0.shape[0]}) must match n_states ({n_states})"
                )

        # Initialize P0 (initial covariance)
        if self.P0 is None:
            self.P0 = np.eye(n_states, dtype=np.float64) * 1e6  # Large initial uncertainty
        else:
            self.P0 = np.atleast_2d(np.asarray(self.P0, dtype=np.float64))
            if self.P0.shape != (n_states, n_states):
                raise ValueError(
                    f"P0 shape {self.P0.shape} must be ({n_states}, {n_states})"
                )

    @property
    def n_states(self) -> int:
        """Number of states (order of the system)."""
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        """Number of inputs."""
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        """Number of outputs."""
        return self.C.shape[0]

    @property
    def order(self) -> int:
        """Alias for n_states (system order)."""
        return self.n_states

    @property
    def hash(self) -> str:
        """MD5 hash of model matrices for comparison."""
        data = np.concatenate([
            self.A.ravel(), self.B.ravel(), self.C.ravel(), self.D.ravel(),
            self.Q.ravel(), self.R.ravel()
        ])
        return hashlib.md5(data.tobytes()).hexdigest()

    def is_stable(self) -> bool:
        """Check if the system is asymptotically stable."""
        eigenvalues = linalg.eigvals(self.A)
        return np.all(np.abs(eigenvalues) < 1.0)

    def is_observable(self, tol: float = 1e-10) -> bool:
        """Check if the system is observable."""
        obs_matrix = self.observability_matrix()
        rank = np.linalg.matrix_rank(obs_matrix, tol=tol)
        return rank == self.n_states

    def is_controllable(self, tol: float = 1e-10) -> bool:
        """Check if the system is controllable."""
        ctrl_matrix = self.controllability_matrix()
        rank = np.linalg.matrix_rank(ctrl_matrix, tol=tol)
        return rank == self.n_states

    def observability_matrix(self) -> np.ndarray:
        """
        Compute the observability matrix.

        Returns
        -------
        O : ndarray, shape (n_outputs * n_states, n_states)
            Observability matrix [C; CA; CA^2; ...; CA^(n-1)]
        """
        n = self.n_states
        O = np.zeros((self.n_outputs * n, n))
        CA = self.C.copy()
        for i in range(n):
            O[i * self.n_outputs:(i + 1) * self.n_outputs, :] = CA
            CA = CA @ self.A
        return O

    def controllability_matrix(self) -> np.ndarray:
        """
        Compute the controllability matrix.

        Returns
        -------
        C : ndarray, shape (n_states, n_inputs * n_states)
            Controllability matrix [B, AB, A^2B, ..., A^(n-1)B]
        """
        n = self.n_states
        C = np.zeros((n, self.n_inputs * n))
        AB = self.B.copy()
        for i in range(n):
            C[:, i * self.n_inputs:(i + 1) * self.n_inputs] = AB
            AB = self.A @ AB
        return C

    def eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues of the state transition matrix A."""
        return linalg.eigvals(self.A)

    def steady_state_kalman_gain(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the steady-state Kalman gain and predicted covariance.

        Solves the discrete algebraic Riccati equation (DARE) to find the
        steady-state predicted covariance P and the corresponding Kalman gain K.

        Returns
        -------
        K : ndarray, shape (n_states, n_outputs)
            Steady-state Kalman gain.
        P : ndarray, shape (n_states, n_states)
            Steady-state predicted covariance.
        """
        # Solve DARE: P = A @ P @ A.T + Q - A @ P @ C.T @ inv(C @ P @ C.T + R) @ C @ P @ A.T
        try:
            P = linalg.solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
        except linalg.LinAlgError:
            # Fall back to iterative solution
            P = self._solve_dare_iterative()

        # Kalman gain: K = P @ C.T @ inv(C @ P @ C.T + R)
        S = self.C @ P @ self.C.T + self.R
        K = linalg.solve(S.T, (P @ self.C.T).T).T

        return K, P

    def _solve_dare_iterative(
        self, max_iter: int = 1000, tol: float = 1e-10
    ) -> np.ndarray:
        """Solve DARE iteratively when closed-form solution fails."""
        P = self.Q.copy()
        for _ in range(max_iter):
            S = self.C @ P @ self.C.T + self.R
            K = linalg.solve(S.T, (P @ self.C.T).T).T
            P_new = self.A @ (P - K @ self.C @ P) @ self.A.T + self.Q
            if np.max(np.abs(P_new - P)) < tol:
                return P_new
            P = P_new
        return P

    def simulate(
        self,
        n_steps: int,
        u: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
        noise: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the system forward in time.

        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate.
        u : array_like, shape (n_inputs, n_steps), optional
            Input sequence. Defaults to zeros.
        x0 : array_like, shape (n_states,), optional
            Initial state. Defaults to self.x0.
        noise : bool, default=True
            Whether to add process and measurement noise.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        y : ndarray, shape (n_outputs, n_steps)
            Output sequence.
        x : ndarray, shape (n_states, n_steps)
            State sequence.
        u : ndarray, shape (n_inputs, n_steps)
            Input sequence (returned for convenience).
        """
        if rng is None:
            rng = np.random.default_rng()

        # Initialize input
        if u is None:
            u = np.zeros((self.n_inputs, n_steps))
        else:
            u = np.atleast_2d(np.asarray(u))
            if u.shape[1] != n_steps:
                raise ValueError(f"u must have {n_steps} columns")

        # Initialize state
        if x0 is None:
            x0 = self.x0
        x = np.zeros((self.n_states, n_steps))
        y = np.zeros((self.n_outputs, n_steps))

        # Generate noise if needed
        if noise:
            # Ensure Q and R are positive semi-definite for Cholesky
            try:
                L_Q = linalg.cholesky(self.Q, lower=True)
                w = L_Q @ rng.standard_normal((self.n_states, n_steps))
            except linalg.LinAlgError:
                # Q might be singular, use eigendecomposition
                eigvals, eigvecs = linalg.eigh(self.Q)
                eigvals = np.maximum(eigvals, 0)
                w = eigvecs @ np.diag(np.sqrt(eigvals)) @ rng.standard_normal((self.n_states, n_steps))

            try:
                L_R = linalg.cholesky(self.R, lower=True)
                v = L_R @ rng.standard_normal((self.n_outputs, n_steps))
            except linalg.LinAlgError:
                eigvals, eigvecs = linalg.eigh(self.R)
                eigvals = np.maximum(eigvals, 0)
                v = eigvecs @ np.diag(np.sqrt(eigvals)) @ rng.standard_normal((self.n_outputs, n_steps))
        else:
            w = np.zeros((self.n_states, n_steps))
            v = np.zeros((self.n_outputs, n_steps))

        # Simulate
        x_curr = x0.copy()
        for k in range(n_steps):
            x[:, k] = x_curr
            y[:, k] = self.C @ x_curr + self.D @ u[:, k] + v[:, k]
            x_curr = self.A @ x_curr + self.B @ u[:, k] + w[:, k]

        return y, x, u

    def predict(
        self,
        y: ArrayLike,
        u: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
        P0: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict outputs using Kalman filtering.

        Parameters
        ----------
        y : array_like, shape (n_outputs, n_steps)
            Observed output sequence.
        u : array_like, shape (n_inputs, n_steps), optional
            Input sequence.
        x0 : array_like, shape (n_states,), optional
            Initial state estimate.
        P0 : array_like, shape (n_states, n_states), optional
            Initial state covariance.

        Returns
        -------
        y_pred : ndarray, shape (n_outputs, n_steps)
            Predicted outputs (one-step-ahead predictions).
        x_filt : ndarray, shape (n_states, n_steps)
            Filtered state estimates.
        """
        from .kalman import kalman_filter

        y = np.atleast_2d(np.asarray(y))
        n_steps = y.shape[1]

        if u is None:
            u = np.zeros((self.n_inputs, n_steps))
        else:
            u = np.atleast_2d(np.asarray(u))

        x_filt, P_filt, x_pred, P_pred, log_lik = kalman_filter(
            y, self.A, self.C, self.Q, self.R,
            B=self.B, D=self.D, u=u,
            x0=x0 if x0 is not None else self.x0,
            P0=P0 if P0 is not None else self.P0,
        )

        # One-step-ahead predictions
        y_pred = self.C @ x_pred + self.D @ u

        return y_pred, x_filt

    def filter(
        self,
        y: ArrayLike,
        u: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
        P0: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply Kalman filter to observations.

        Parameters
        ----------
        y : array_like, shape (n_outputs, n_steps)
            Observed output sequence.
        u : array_like, shape (n_inputs, n_steps), optional
            Input sequence.
        x0 : array_like, shape (n_states,), optional
            Initial state estimate.
        P0 : array_like, shape (n_states, n_states), optional
            Initial state covariance.

        Returns
        -------
        x_filt : ndarray, shape (n_states, n_steps)
            Filtered state estimates.
        P_filt : ndarray, shape (n_states, n_states, n_steps)
            Filtered state covariances.
        log_lik : float
            Log-likelihood of the observations.
        """
        from .kalman import kalman_filter

        y = np.atleast_2d(np.asarray(y))
        n_steps = y.shape[1]

        if u is None:
            u = np.zeros((self.n_inputs, n_steps))
        else:
            u = np.atleast_2d(np.asarray(u))

        x_filt, P_filt, x_pred, P_pred, log_lik = kalman_filter(
            y, self.A, self.C, self.Q, self.R,
            B=self.B, D=self.D, u=u,
            x0=x0 if x0 is not None else self.x0,
            P0=P0 if P0 is not None else self.P0,
        )

        return x_filt, P_filt, log_lik

    def smooth(
        self,
        y: ArrayLike,
        u: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
        P0: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Apply Kalman smoother to observations.

        Parameters
        ----------
        y : array_like, shape (n_outputs, n_steps)
            Observed output sequence.
        u : array_like, shape (n_inputs, n_steps), optional
            Input sequence.
        x0 : array_like, shape (n_states,), optional
            Initial state estimate.
        P0 : array_like, shape (n_states, n_states), optional
            Initial state covariance.

        Returns
        -------
        x_smooth : ndarray, shape (n_states, n_steps)
            Smoothed state estimates.
        P_smooth : ndarray, shape (n_states, n_states, n_steps)
            Smoothed state covariances.
        Pt : ndarray, shape (n_states, n_states, n_steps-1)
            Cross-covariances E[x_k @ x_{k+1}.T | Y].
        log_lik : float
            Log-likelihood of the observations.
        """
        from .kalman import kalman_smoother

        y = np.atleast_2d(np.asarray(y))
        n_steps = y.shape[1]

        if u is None:
            u = np.zeros((self.n_inputs, n_steps))
        else:
            u = np.atleast_2d(np.asarray(u))

        x_smooth, P_smooth, Pt, x_filt, P_filt, log_lik = kalman_smoother(
            y, self.A, self.C, self.Q, self.R,
            B=self.B, D=self.D, u=u,
            x0=x0 if x0 is not None else self.x0,
            P0=P0 if P0 is not None else self.P0,
        )

        return x_smooth, P_smooth, Pt, log_lik

    def log_likelihood(
        self,
        y: ArrayLike,
        u: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
        P0: Optional[ArrayLike] = None,
    ) -> float:
        """
        Compute the log-likelihood of observations under this model.

        Parameters
        ----------
        y : array_like, shape (n_outputs, n_steps)
            Observed output sequence.
        u : array_like, shape (n_inputs, n_steps), optional
            Input sequence.
        x0 : array_like, shape (n_states,), optional
            Initial state estimate.
        P0 : array_like, shape (n_states, n_states), optional
            Initial state covariance.

        Returns
        -------
        log_lik : float
            Log-likelihood of the observations.
        """
        _, _, log_lik = self.filter(y, u, x0, P0)
        return log_lik

    def residuals(
        self,
        y: ArrayLike,
        u: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction residuals (innovations).

        Parameters
        ----------
        y : array_like, shape (n_outputs, n_steps)
            Observed output sequence.
        u : array_like, shape (n_inputs, n_steps), optional
            Input sequence.

        Returns
        -------
        residuals : ndarray, shape (n_outputs, n_steps)
            Prediction residuals (y - y_pred).
        y_pred : ndarray, shape (n_outputs, n_steps)
            Predicted outputs.
        """
        y = np.atleast_2d(np.asarray(y))
        y_pred, _ = self.predict(y, u)
        return y - y_pred, y_pred

    def transform(
        self,
        T: ArrayLike,
    ) -> "LinearSystem":
        """
        Apply a similarity transformation to the state space.

        The transformation x_new = T @ x yields:
            A_new = T @ A @ T^{-1}
            B_new = T @ B
            C_new = C @ T^{-1}
            Q_new = T @ Q @ T.T

        Parameters
        ----------
        T : array_like, shape (n_states, n_states)
            Transformation matrix (must be invertible).

        Returns
        -------
        sys : LinearSystem
            Transformed system.
        """
        T = np.atleast_2d(np.asarray(T))
        T_inv = linalg.inv(T)

        return LinearSystem(
            A=T @ self.A @ T_inv,
            B=T @ self.B,
            C=self.C @ T_inv,
            D=self.D,
            Q=T @ self.Q @ T.T,
            R=self.R,
            x0=T @ self.x0,
            P0=T @ self.P0 @ T.T,
            name=self.name,
        )

    def canonize(self, form: str = "modal") -> "LinearSystem":
        """
        Convert to canonical form.

        Parameters
        ----------
        form : str, default="modal"
            Canonical form: "modal" diagonalizes A.

        Returns
        -------
        sys : LinearSystem
            System in canonical form.
        """
        if form == "modal":
            eigenvalues, eigenvectors = linalg.eig(self.A)
            T = eigenvectors
            try:
                return self.transform(linalg.inv(T))
            except linalg.LinAlgError:
                # If eigenvector matrix is singular, return original
                return LinearSystem(
                    A=self.A.copy(), B=self.B.copy(), C=self.C.copy(),
                    D=self.D.copy(), Q=self.Q.copy(), R=self.R.copy(),
                    x0=self.x0.copy(), P0=self.P0.copy(), name=self.name,
                )
        else:
            raise ValueError(f"Unknown canonical form: {form}")

    def reduce(self, n_states: int) -> "LinearSystem":
        """
        Reduce model order using balanced truncation.

        Parameters
        ----------
        n_states : int
            Number of states in reduced model.

        Returns
        -------
        sys : LinearSystem
            Reduced order model.
        """
        if n_states >= self.n_states:
            return LinearSystem(
                A=self.A.copy(), B=self.B.copy(), C=self.C.copy(),
                D=self.D.copy(), Q=self.Q.copy(), R=self.R.copy(),
                x0=self.x0.copy(), P0=self.P0.copy(), name=self.name,
            )

        # Compute controllability Gramian
        Wc = linalg.solve_discrete_lyapunov(self.A, self.B @ self.B.T)

        # Compute observability Gramian
        Wo = linalg.solve_discrete_lyapunov(self.A.T, self.C.T @ self.C)

        # Compute Hankel singular values and balancing transformation
        L_c = linalg.cholesky(Wc + 1e-10 * np.eye(self.n_states), lower=True)
        M = L_c.T @ Wo @ L_c
        U, s, Vh = linalg.svd(M)

        # Balancing transformation
        Sigma_sqrt = np.diag(np.sqrt(np.sqrt(s[:n_states])))
        Sigma_sqrt_inv = np.diag(1.0 / np.sqrt(np.sqrt(s[:n_states])))

        T = Sigma_sqrt_inv @ U[:, :n_states].T @ L_c.T
        T_inv = L_c @ U[:, :n_states] @ Sigma_sqrt

        return LinearSystem(
            A=T @ self.A @ T_inv,
            B=T @ self.B,
            C=self.C @ T_inv,
            D=self.D,
            Q=T @ self.Q @ T.T,
            R=self.R,
            x0=T @ self.x0,
            P0=T @ self.P0 @ T.T,
            name=self.name,
        )

    def copy(self) -> "LinearSystem":
        """Create a deep copy of this system."""
        return LinearSystem(
            A=self.A.copy(),
            B=self.B.copy(),
            C=self.C.copy(),
            D=self.D.copy(),
            Q=self.Q.copy(),
            R=self.R.copy(),
            x0=self.x0.copy(),
            P0=self.P0.copy(),
            name=self.name,
        )

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return (
            f"LinearSystem(n_states={self.n_states}, n_inputs={self.n_inputs}, "
            f"n_outputs={self.n_outputs}{name_str})"
        )

    def __str__(self) -> str:
        lines = [
            f"Linear System: {self.name}" if self.name else "Linear System",
            f"  States: {self.n_states}",
            f"  Inputs: {self.n_inputs}",
            f"  Outputs: {self.n_outputs}",
            f"  Stable: {self.is_stable()}",
        ]
        return "\n".join(lines)

    @classmethod
    def from_matrices(
        cls,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        **kwargs,
    ) -> "LinearSystem":
        """
        Create a LinearSystem from matrices.

        Convenience constructor that accepts all standard state-space matrices.
        """
        return cls(A=A, B=B, C=C, D=D, Q=Q, R=R, **kwargs)

    @classmethod
    def random(
        cls,
        n_states: int,
        n_inputs: int = 1,
        n_outputs: int = 1,
        stable: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> "LinearSystem":
        """
        Generate a random LinearSystem.

        Parameters
        ----------
        n_states : int
            Number of states.
        n_inputs : int, default=1
            Number of inputs.
        n_outputs : int, default=1
            Number of outputs.
        stable : bool, default=True
            If True, ensure the system is stable.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        sys : LinearSystem
            Random linear system.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Generate A matrix
        A = rng.standard_normal((n_states, n_states))
        if stable:
            # Scale to ensure stability
            eigenvalues = linalg.eigvals(A)
            max_eig = np.max(np.abs(eigenvalues))
            if max_eig > 0.95:
                A = A * (0.95 / max_eig)

        B = rng.standard_normal((n_states, n_inputs))
        C = rng.standard_normal((n_outputs, n_states))
        D = rng.standard_normal((n_outputs, n_inputs)) * 0.1

        # Generate positive definite Q and R
        Q_sqrt = rng.standard_normal((n_states, n_states))
        Q = Q_sqrt @ Q_sqrt.T + 0.01 * np.eye(n_states)

        R_sqrt = rng.standard_normal((n_outputs, n_outputs))
        R = R_sqrt @ R_sqrt.T + 0.01 * np.eye(n_outputs)

        return cls(A=A, B=B, C=C, D=D, Q=Q, R=R)

    @classmethod
    def identify(
        cls,
        y: ArrayLike,
        n_states: int,
        u: Optional[ArrayLike] = None,
        method: str = "em",
        **kwargs,
    ) -> "LinearSystem":
        """
        Identify a linear system from input-output data.

        Parameters
        ----------
        y : array_like, shape (n_outputs, n_steps)
            Observed output sequence.
        n_states : int
            Number of states to estimate.
        u : array_like, shape (n_inputs, n_steps), optional
            Input sequence.
        method : str, default="em"
            Identification method: "em" or "subspace".
        **kwargs
            Additional arguments passed to the identification method.

        Returns
        -------
        sys : LinearSystem
            Identified linear system.
        """
        if method == "em":
            from .em import em_identify
            return em_identify(y, n_states, u=u, **kwargs)
        elif method == "subspace":
            from .subspace import subspace_id
            A, B, C, D, X, Q, R, S = subspace_id(y, n_states, u=u, **kwargs)
            return cls(A=A, B=B, C=C, D=D, Q=Q, R=R)
        else:
            raise ValueError(f"Unknown identification method: {method}")

"""Tests for Kalman filtering and smoothing."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linsys import kalman_filter, kalman_smoother, kalman_predict, kalman_update
from linsys.kalman import steady_state_kalman_gain


class TestKalmanPredictUpdate:
    """Test individual predict and update steps."""

    def test_predict_step(self):
        """Test Kalman prediction step."""
        n_states = 2
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        Q = np.eye(n_states) * 0.1

        x = np.array([1.0, 2.0])
        P = np.eye(n_states) * 0.5

        x_pred, P_pred = kalman_predict(x, P, A, Q)

        # Check predicted state
        expected_x = A @ x
        assert_allclose(x_pred, expected_x)

        # Check predicted covariance
        expected_P = A @ P @ A.T + Q
        assert_allclose(P_pred, expected_P)

    def test_predict_with_input(self):
        """Test prediction step with input."""
        A = np.array([[0.9]])
        B = np.array([[1.0]])
        Q = np.array([[0.1]])

        x = np.array([0.0])
        P = np.array([[1.0]])
        u = np.array([1.0])

        x_pred, P_pred = kalman_predict(x, P, A, Q, B=B, u=u)

        expected_x = A @ x + B @ u
        assert_allclose(x_pred, expected_x)

    def test_update_step(self):
        """Test Kalman update step."""
        n_states, n_outputs = 2, 2
        C = np.eye(n_outputs)
        R = np.eye(n_outputs) * 0.1

        x_pred = np.array([1.0, 2.0])
        P_pred = np.eye(n_states) * 0.5
        y = np.array([1.1, 1.9])

        x_filt, P_filt, K, innovation, log_lik = kalman_update(
            x_pred, P_pred, y, C, R
        )

        # Innovation should be y - C @ x_pred
        expected_innov = y - C @ x_pred
        assert_allclose(innovation, expected_innov)

        # Filtered state should be between prediction and observation
        assert np.all((x_filt >= np.minimum(x_pred, y) - 0.1) &
                      (x_filt <= np.maximum(x_pred, y) + 0.1))

    def test_update_with_nan(self):
        """Test update step handles missing data."""
        C = np.eye(2)
        R = np.eye(2) * 0.1

        x_pred = np.array([1.0, 2.0])
        P_pred = np.eye(2) * 0.5
        y = np.array([1.1, np.nan])  # Second observation missing

        x_filt, P_filt, K, innovation, log_lik = kalman_update(
            x_pred, P_pred, y, C, R
        )

        # First dimension should be updated
        assert x_filt[0] != x_pred[0]
        # Second dimension should be unchanged
        assert x_filt[1] == x_pred[1]


class TestKalmanFilter:
    """Test full Kalman filter."""

    def test_filter_dimensions(self):
        """Test that filter returns correct dimensions."""
        n_states, n_outputs, n_steps = 3, 2, 100

        A = np.random.randn(n_states, n_states) * 0.5
        C = np.random.randn(n_outputs, n_states)
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_outputs) * 0.2
        y = np.random.randn(n_outputs, n_steps)

        x_filt, P_filt, x_pred, P_pred, log_lik = kalman_filter(
            y, A, C, Q, R
        )

        assert x_filt.shape == (n_states, n_steps)
        assert P_filt.shape == (n_states, n_states, n_steps)
        assert x_pred.shape == (n_states, n_steps)
        assert P_pred.shape == (n_states, n_states, n_steps)
        assert isinstance(log_lik, float)

    def test_filter_with_known_system(self):
        """Test filter on a known system."""
        rng = np.random.default_rng(42)

        # Simple scalar system
        A = np.array([[0.9]])
        C = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])

        # True state sequence
        n_steps = 100
        x_true = np.zeros((1, n_steps))
        y = np.zeros((1, n_steps))

        x_true[0, 0] = 1.0
        for k in range(n_steps):
            y[0, k] = C[0, 0] * x_true[0, k] + rng.normal(0, np.sqrt(R[0, 0]))
            if k < n_steps - 1:
                x_true[0, k + 1] = A[0, 0] * x_true[0, k] + rng.normal(0, np.sqrt(Q[0, 0]))

        # Filter
        x_filt, P_filt, _, _, log_lik = kalman_filter(y, A, C, Q, R)

        # RMSE should be reasonable
        rmse = np.sqrt(np.mean((x_filt - x_true) ** 2))
        assert rmse < 0.5

    def test_filter_missing_data(self):
        """Test that filter handles missing data."""
        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        C = np.eye(2)
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.2

        n_steps = 50
        y = np.random.randn(2, n_steps)

        # Introduce missing data
        y[0, 10:20] = np.nan
        y[1, 30:35] = np.nan

        # Should not raise
        x_filt, P_filt, _, _, log_lik = kalman_filter(y, A, C, Q, R)

        assert not np.any(np.isnan(x_filt))
        assert np.isfinite(log_lik)

    def test_steady_state_convergence(self):
        """Test that filter converges to steady state."""
        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        C = np.eye(2)
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.2

        n_steps = 200
        y = np.random.randn(2, n_steps)

        x_filt, P_filt, _, _, _ = kalman_filter(
            y, A, C, Q, R, steady_state=True, steady_state_window=20
        )

        # Covariance should converge
        P_diff = np.max(np.abs(P_filt[:, :, -1] - P_filt[:, :, -2]))
        assert P_diff < 1e-5


class TestKalmanSmoother:
    """Test Kalman smoother."""

    def test_smoother_dimensions(self):
        """Test that smoother returns correct dimensions."""
        n_states, n_outputs, n_steps = 3, 2, 100

        A = np.random.randn(n_states, n_states) * 0.5
        C = np.random.randn(n_outputs, n_states)
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_outputs) * 0.2
        y = np.random.randn(n_outputs, n_steps)

        x_smooth, P_smooth, Pt, x_filt, P_filt, log_lik = kalman_smoother(
            y, A, C, Q, R
        )

        assert x_smooth.shape == (n_states, n_steps)
        assert P_smooth.shape == (n_states, n_states, n_steps)
        assert Pt.shape == (n_states, n_states, n_steps - 1)

    def test_smoother_reduces_variance(self):
        """Test that smoothing reduces estimation variance."""
        rng = np.random.default_rng(42)

        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        C = np.eye(2)
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.5

        # Simulate
        n_steps = 100
        x_true = np.zeros((2, n_steps))
        y = np.zeros((2, n_steps))

        x_true[:, 0] = rng.standard_normal(2)
        for k in range(n_steps):
            y[:, k] = C @ x_true[:, k] + rng.standard_normal(2) * np.sqrt(R[0, 0])
            if k < n_steps - 1:
                x_true[:, k + 1] = A @ x_true[:, k] + rng.standard_normal(2) * np.sqrt(Q[0, 0])

        # Smooth
        x_smooth, P_smooth, _, x_filt, P_filt, _ = kalman_smoother(y, A, C, Q, R)

        # Smoother variance should be less than or equal to filter variance
        for k in range(n_steps):
            assert np.trace(P_smooth[:, :, k]) <= np.trace(P_filt[:, :, k]) + 1e-10

    def test_smoother_last_equals_filter(self):
        """Test that smoother equals filter at last time step."""
        A = np.array([[0.9]])
        C = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[0.2]])

        y = np.random.randn(1, 50)

        x_smooth, P_smooth, _, x_filt, P_filt, _ = kalman_smoother(y, A, C, Q, R)

        # At last time step, smoother should equal filter
        assert_allclose(x_smooth[:, -1], x_filt[:, -1])
        assert_allclose(P_smooth[:, :, -1], P_filt[:, :, -1])


class TestSteadyStateKalmanGain:
    """Test steady-state Kalman gain computation."""

    def test_gain_computation(self):
        """Test that steady-state gain is computed."""
        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        C = np.eye(2)
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.2

        K, P_pred, P_filt = steady_state_kalman_gain(A, C, Q, R)

        assert K.shape == (2, 2)
        assert P_pred.shape == (2, 2)
        assert P_filt.shape == (2, 2)

    def test_gain_satisfies_riccati(self):
        """Test that P satisfies the DARE."""
        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        C = np.eye(2)
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.2

        K, P_pred, P_filt = steady_state_kalman_gain(A, C, Q, R)

        # Check: P_pred = A @ P_filt @ A.T + Q
        P_pred_check = A @ P_filt @ A.T + Q
        assert_allclose(P_pred, P_pred_check, rtol=1e-5)

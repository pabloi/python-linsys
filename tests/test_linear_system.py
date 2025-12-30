"""Tests for LinearSystem class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from linsys import LinearSystem


class TestLinearSystemBasic:
    """Test basic LinearSystem functionality."""

    def test_create_minimal(self):
        """Test creating a minimal system with just A and C."""
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        C = np.array([[1.0, 0.0], [0.0, 1.0]])

        sys = LinearSystem(A=A, C=C)

        assert sys.n_states == 2
        assert sys.n_outputs == 2
        assert sys.n_inputs == 1

    def test_create_full(self):
        """Test creating a full system with all matrices."""
        n_states, n_inputs, n_outputs = 3, 2, 4

        A = np.random.randn(n_states, n_states) * 0.5
        B = np.random.randn(n_states, n_inputs)
        C = np.random.randn(n_outputs, n_states)
        D = np.random.randn(n_outputs, n_inputs)
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_outputs) * 0.2

        sys = LinearSystem(A=A, B=B, C=C, D=D, Q=Q, R=R)

        assert sys.n_states == n_states
        assert sys.n_inputs == n_inputs
        assert sys.n_outputs == n_outputs

    def test_dimension_validation(self):
        """Test that invalid dimensions raise errors."""
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        C_wrong = np.array([[1.0, 0.0, 0.0]])  # Wrong number of columns

        with pytest.raises(ValueError):
            LinearSystem(A=A, C=C_wrong)

    def test_stability_check(self):
        """Test stability checking."""
        # Stable system
        A_stable = np.array([[0.5, 0.0], [0.0, 0.3]])
        C = np.eye(2)
        sys_stable = LinearSystem(A=A_stable, C=C)
        assert sys_stable.is_stable()

        # Unstable system
        A_unstable = np.array([[1.5, 0.0], [0.0, 0.3]])
        sys_unstable = LinearSystem(A=A_unstable, C=C)
        assert not sys_unstable.is_stable()

    def test_eigenvalues(self):
        """Test eigenvalue computation."""
        eigenvalues = np.array([0.9, 0.5])
        A = np.diag(eigenvalues)
        C = np.eye(2)

        sys = LinearSystem(A=A, C=C)
        computed_eigs = np.sort(np.abs(sys.eigenvalues()))

        assert_allclose(computed_eigs, np.sort(eigenvalues))


class TestLinearSystemSimulation:
    """Test simulation functionality."""

    def test_simulate_deterministic(self):
        """Test deterministic simulation (no noise)."""
        A = np.array([[0.9]])
        C = np.array([[1.0]])
        x0 = np.array([1.0])

        sys = LinearSystem(A=A, C=C, x0=x0)
        y, x, u = sys.simulate(n_steps=10, noise=False)

        # Check that state decays geometrically
        expected_x = 0.9 ** np.arange(10)
        assert_allclose(x.ravel(), expected_x)
        assert_allclose(y.ravel(), expected_x)

    def test_simulate_with_input(self):
        """Test simulation with input."""
        A = np.array([[0.5]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])

        sys = LinearSystem(A=A, B=B, C=C, D=D)

        # Step input
        n_steps = 20
        u = np.ones((1, n_steps))

        y, x, _ = sys.simulate(n_steps=n_steps, u=u, noise=False)

        # Should converge to steady state x = B/(1-A) = 1/0.5 = 2
        assert x[0, -1] > 1.9

    def test_simulate_with_noise(self):
        """Test that noisy simulation produces different results."""
        sys = LinearSystem.random(n_states=2, n_outputs=2, rng=np.random.default_rng(42))

        y1, _, _ = sys.simulate(n_steps=100, noise=True, rng=np.random.default_rng(1))
        y2, _, _ = sys.simulate(n_steps=100, noise=True, rng=np.random.default_rng(2))

        # Different seeds should give different results
        assert not np.allclose(y1, y2)


class TestLinearSystemFiltering:
    """Test Kalman filtering and smoothing."""

    def test_filter_recovers_state(self):
        """Test that filtering approximately recovers the true state."""
        rng = np.random.default_rng(42)

        # Create a simple stable system with low noise
        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        C = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.eye(2) * 0.01

        sys = LinearSystem(A=A, C=C, Q=Q, R=R)

        # Simulate
        y, x_true, u = sys.simulate(n_steps=100, noise=True, rng=rng)

        # Filter
        x_filt, P_filt, log_lik = sys.filter(y)

        # Filtered states should be close to true states (at least for later times)
        rmse = np.sqrt(np.mean((x_filt[:, 50:] - x_true[:, 50:]) ** 2))
        assert rmse < 0.5

    def test_smoother_better_than_filter(self):
        """Test that smoothing is at least as good as filtering."""
        rng = np.random.default_rng(42)

        sys = LinearSystem.random(n_states=2, n_outputs=2, stable=True, rng=rng)

        # Simulate
        y, x_true, u = sys.simulate(n_steps=100, noise=True, rng=rng)

        # Filter and smooth
        x_filt, P_filt, log_lik_filt = sys.filter(y)
        x_smooth, P_smooth, Pt, log_lik_smooth = sys.smooth(y)

        # RMSE should be lower for smoother
        rmse_filt = np.sqrt(np.mean((x_filt - x_true) ** 2))
        rmse_smooth = np.sqrt(np.mean((x_smooth - x_true) ** 2))

        assert rmse_smooth <= rmse_filt * 1.1  # Allow small tolerance

    def test_log_likelihood_positive(self):
        """Test that log-likelihood is computed correctly."""
        rng = np.random.default_rng(42)

        sys = LinearSystem.random(n_states=2, n_outputs=2, stable=True, rng=rng)
        y, _, _ = sys.simulate(n_steps=100, noise=True, rng=rng)

        log_lik = sys.log_likelihood(y)

        # Log-likelihood should be negative (log of probability < 1)
        assert log_lik < 0


class TestLinearSystemTransformations:
    """Test system transformations."""

    def test_transform_preserves_dynamics(self):
        """Test that similarity transform preserves input-output behavior."""
        rng = np.random.default_rng(42)

        sys = LinearSystem.random(n_states=3, n_outputs=2, stable=True, rng=rng)

        # Create a random invertible transformation
        T = rng.standard_normal((3, 3))
        while np.abs(np.linalg.det(T)) < 0.1:
            T = rng.standard_normal((3, 3))

        sys_transformed = sys.transform(T)

        # Simulate both systems with same input and noise
        u = rng.standard_normal((1, 50))
        y1, _, _ = sys.simulate(n_steps=50, u=u, noise=False)
        y2, _, _ = sys_transformed.simulate(n_steps=50, u=u, noise=False)

        # Outputs should be the same
        assert_allclose(y1, y2, rtol=1e-5)

    def test_copy_independent(self):
        """Test that copy creates an independent object."""
        sys = LinearSystem.random(n_states=2, n_outputs=2)
        sys_copy = sys.copy()

        # Modify original
        sys.A[0, 0] = 999

        # Copy should be unaffected
        assert sys_copy.A[0, 0] != 999


class TestLinearSystemRandom:
    """Test random system generation."""

    def test_random_stable(self):
        """Test that random systems can be stable."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            sys = LinearSystem.random(
                n_states=5, n_outputs=3, n_inputs=2,
                stable=True, rng=rng
            )
            assert sys.is_stable()

    def test_random_dimensions(self):
        """Test that random systems have correct dimensions."""
        sys = LinearSystem.random(n_states=4, n_outputs=3, n_inputs=2)

        assert sys.n_states == 4
        assert sys.n_outputs == 3
        assert sys.n_inputs == 2


class TestLinearSystemIdentify:
    """Test system identification."""

    def test_identify_subspace(self):
        """Test subspace identification."""
        rng = np.random.default_rng(42)

        # Create true system
        true_sys = LinearSystem.random(n_states=2, n_outputs=3, stable=True, rng=rng)

        # Generate data
        y, x, u = true_sys.simulate(n_steps=500, noise=True, rng=rng)

        # Identify
        est_sys = LinearSystem.identify(y, n_states=2, u=u, method="subspace")

        # Should have correct dimensions
        assert est_sys.n_states == 2
        assert est_sys.n_outputs == 3

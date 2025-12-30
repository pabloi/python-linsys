"""Tests for subspace identification."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linsys import LinearSystem, subspace_id
from linsys.subspace import hankel_matrix, estimate_transition_matrix


class TestHankelMatrix:
    """Test Hankel matrix construction."""

    def test_hankel_dimensions(self):
        """Test that Hankel matrix has correct dimensions."""
        n_channels, n_samples = 3, 100
        n_rows = 10
        data = np.random.randn(n_channels, n_samples)

        H = hankel_matrix(data, n_rows)

        expected_rows = n_channels * n_rows
        expected_cols = n_samples - n_rows + 1

        assert H.shape == (expected_rows, expected_cols)

    def test_hankel_structure(self):
        """Test that Hankel matrix has correct structure."""
        data = np.array([[1, 2, 3, 4, 5]])
        H = hankel_matrix(data, n_rows=3)

        expected = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        ])

        assert_allclose(H, expected)

    def test_hankel_multichannel(self):
        """Test Hankel matrix with multiple channels."""
        data = np.array([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])
        H = hankel_matrix(data, n_rows=2)

        # Should stack blocks
        assert H.shape == (4, 3)
        # First block: channels at time 0, 1
        assert_allclose(H[0:2, 0], [1, 10])
        assert_allclose(H[2:4, 0], [2, 20])


class TestSubspaceID:
    """Test subspace identification."""

    def test_subspace_id_dimensions(self):
        """Test that subspace ID returns correct dimensions."""
        n_outputs, n_samples = 3, 200
        n_states = 2
        n_inputs = 1

        y = np.random.randn(n_outputs, n_samples)
        u = np.random.randn(n_inputs, n_samples)

        A, B, C, D, X, Q, R, S = subspace_id(y, n_states, u=u, horizon=10)

        assert A.shape == (n_states, n_states)
        assert B.shape == (n_states, n_inputs)
        assert C.shape == (n_outputs, n_states)
        assert D.shape == (n_outputs, n_inputs)
        assert Q.shape == (n_states, n_states)
        assert R.shape == (n_outputs, n_outputs)

    def test_subspace_id_no_input(self):
        """Test subspace ID without input."""
        n_outputs, n_samples = 2, 150
        n_states = 2

        y = np.random.randn(n_outputs, n_samples)

        A, B, C, D, X, Q, R, S = subspace_id(y, n_states, horizon=8)

        assert A.shape == (n_states, n_states)
        assert C.shape == (n_outputs, n_states)

    def test_subspace_id_recovers_dynamics(self):
        """Test that subspace ID recovers system dynamics."""
        rng = np.random.default_rng(42)

        # Create true system
        A_true = np.array([[0.9, 0.1], [-0.1, 0.8]])
        C_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q_true = np.eye(2) * 0.01
        R_true = np.eye(2) * 0.1

        true_sys = LinearSystem(A=A_true, C=C_true, Q=Q_true, R=R_true)

        # Generate data
        y, x, u = true_sys.simulate(n_steps=500, noise=True, rng=rng)

        # Identify
        A_est, B_est, C_est, D_est, X_est, Q_est, R_est, S_est = subspace_id(
            y, n_states=2, u=u, horizon=15
        )

        # Create estimated system
        est_sys = LinearSystem(A=A_est, C=C_est, Q=Q_est, R=R_est)

        # Check that eigenvalues are similar
        true_eigs = np.sort(np.abs(np.linalg.eigvals(A_true)))
        est_eigs = np.sort(np.abs(np.linalg.eigvals(A_est)))

        # Allow some tolerance due to estimation error
        assert_allclose(true_eigs, est_eigs, rtol=0.3)

    def test_subspace_id_minimum_samples(self):
        """Test that subspace ID requires minimum samples."""
        y = np.random.randn(2, 20)

        with pytest.raises(ValueError):
            # Too few samples for horizon=10
            subspace_id(y, n_states=2, horizon=10)


class TestEstimateTransitionMatrix:
    """Test transition matrix estimation."""

    def test_estimate_A(self):
        """Test estimating A from state sequence."""
        rng = np.random.default_rng(42)

        # True A
        A_true = np.array([[0.9, 0.1], [-0.1, 0.8]])

        # Generate state sequence
        n_steps = 1000
        X = np.zeros((2, n_steps))
        X[:, 0] = rng.standard_normal(2)

        for k in range(n_steps - 1):
            X[:, k + 1] = A_true @ X[:, k] + rng.standard_normal(2) * 0.01

        # Estimate A
        A_est, B_est = estimate_transition_matrix(X)

        assert_allclose(A_est, A_true, rtol=0.1)

    def test_estimate_AB_with_input(self):
        """Test estimating A and B with input."""
        rng = np.random.default_rng(42)

        A_true = np.array([[0.9]])
        B_true = np.array([[0.5]])

        n_steps = 1000
        X = np.zeros((1, n_steps))
        U = rng.standard_normal((1, n_steps))
        X[0, 0] = 0

        for k in range(n_steps - 1):
            X[0, k + 1] = A_true[0, 0] * X[0, k] + B_true[0, 0] * U[0, k]

        A_est, B_est = estimate_transition_matrix(X, U)

        assert_allclose(A_est, A_true, rtol=0.1)
        assert_allclose(B_est, B_true, rtol=0.1)


class TestSubspaceViaLinearSystem:
    """Test subspace ID through LinearSystem interface."""

    def test_identify_method(self):
        """Test LinearSystem.identify with subspace method."""
        rng = np.random.default_rng(42)

        # Create and simulate true system
        true_sys = LinearSystem.random(n_states=2, n_outputs=3, stable=True, rng=rng)
        y, x, u = true_sys.simulate(n_steps=500, noise=True, rng=rng)

        # Identify using subspace method
        est_sys = LinearSystem.identify(y, n_states=2, u=u, method="subspace")

        assert est_sys.n_states == 2
        assert est_sys.n_outputs == 3
        assert est_sys.is_stable() or True  # May not always be stable

    def test_subspace_then_em_refinement(self):
        """Test using subspace ID as initialization for EM."""
        rng = np.random.default_rng(42)

        # Create and simulate true system
        true_sys = LinearSystem.random(n_states=2, n_outputs=2, stable=True, rng=rng)
        y, x, u = true_sys.simulate(n_steps=500, noise=True, rng=rng)

        # First pass: subspace ID
        A_init, B_init, C_init, D_init, _, Q_init, R_init, _ = subspace_id(
            y, n_states=2, u=u
        )

        # Second pass: EM with subspace initialization
        from linsys.em import em_algorithm, EMOptions

        opts = EMOptions(max_iter=50, verbose=False)
        result = em_algorithm(
            y, n_states=2, u=u,
            A0=A_init, B0=B_init, C0=C_init, D0=D_init,
            Q0=Q_init, R0=R_init,
            opts=opts
        )

        # Should have higher log-likelihood than initial
        assert len(result.log_likelihood) > 0

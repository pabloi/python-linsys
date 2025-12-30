"""Tests for EM algorithm."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linsys import LinearSystem, em_identify
from linsys.em import em_algorithm, em_step, EMOptions


class TestEMBasic:
    """Test basic EM functionality."""

    def test_em_runs(self):
        """Test that EM runs without errors."""
        rng = np.random.default_rng(42)

        # Create a simple system
        true_sys = LinearSystem.random(n_states=2, n_outputs=2, stable=True, rng=rng)

        # Generate data
        y, _, _ = true_sys.simulate(n_steps=200, noise=True, rng=rng)

        # Run EM
        opts = EMOptions(max_iter=10, verbose=False)
        result = em_algorithm(y, n_states=2, opts=opts)

        assert result.A.shape == (2, 2)
        assert result.C.shape == (2, 2)
        assert len(result.log_likelihood) > 0

    def test_em_identify_interface(self):
        """Test the high-level em_identify interface."""
        rng = np.random.default_rng(42)

        true_sys = LinearSystem.random(n_states=2, n_outputs=3, stable=True, rng=rng)
        y, _, _ = true_sys.simulate(n_steps=200, noise=True, rng=rng)

        # Should return a LinearSystem
        sys = em_identify(y, n_states=2, opts=EMOptions(max_iter=20))

        assert isinstance(sys, LinearSystem)
        assert sys.n_states == 2
        assert sys.n_outputs == 3

    def test_em_step(self):
        """Test single EM step."""
        rng = np.random.default_rng(42)

        n_states, n_outputs, n_steps = 2, 2, 100

        A = np.random.randn(n_states, n_states) * 0.5
        B = np.zeros((n_states, 1))
        C = np.random.randn(n_outputs, n_states)
        D = np.zeros((n_outputs, 1))
        Q = np.eye(n_states) * 0.1
        R = np.eye(n_outputs) * 0.2
        x0 = np.zeros(n_states)
        P0 = np.eye(n_states)

        y = rng.standard_normal((n_outputs, n_steps))

        # Run one step
        A_new, B_new, C_new, D_new, Q_new, R_new, x0_new, P0_new, log_lik = em_step(
            y, A, B, C, D, Q, R, x0, P0
        )

        # Matrices should have correct shapes
        assert A_new.shape == A.shape
        assert C_new.shape == C.shape
        assert Q_new.shape == Q.shape
        assert R_new.shape == R.shape


class TestEMConvergence:
    """Test EM convergence properties."""

    def test_log_likelihood_increases(self):
        """Test that log-likelihood increases (or stays same) at each iteration."""
        rng = np.random.default_rng(42)

        true_sys = LinearSystem.random(n_states=2, n_outputs=2, stable=True, rng=rng)
        y, _, _ = true_sys.simulate(n_steps=300, noise=True, rng=rng)

        opts = EMOptions(max_iter=50, verbose=False)
        result = em_algorithm(y, n_states=2, opts=opts)

        # Log-likelihood should be monotonically increasing (with small tolerance)
        ll = result.log_likelihood
        for i in range(1, len(ll)):
            assert ll[i] >= ll[i - 1] - 1e-6, f"LL decreased at iteration {i}"

    def test_convergence_detection(self):
        """Test that convergence is detected."""
        rng = np.random.default_rng(42)

        true_sys = LinearSystem.random(n_states=2, n_outputs=2, stable=True, rng=rng)
        y, _, _ = true_sys.simulate(n_steps=300, noise=True, rng=rng)

        opts = EMOptions(max_iter=500, tol=1e-4, verbose=False)
        result = em_algorithm(y, n_states=2, opts=opts)

        # Should converge before max_iter
        assert result.n_iterations < 500 or result.converged


class TestEMOptions:
    """Test EM options."""

    def test_fix_parameters(self):
        """Test that fixing parameters works."""
        rng = np.random.default_rng(42)

        n_states, n_outputs = 2, 2
        y = rng.standard_normal((n_outputs, 100))

        # Fix A to identity
        A_fixed = np.eye(n_states)

        opts = EMOptions(max_iter=10, fix_A=True)
        result = em_algorithm(y, n_states=n_states, A0=A_fixed, opts=opts)

        # A should still be identity
        assert_allclose(result.A, A_fixed)

    def test_diagonal_constraints(self):
        """Test diagonal constraints."""
        rng = np.random.default_rng(42)

        n_states, n_outputs = 2, 2
        y = rng.standard_normal((n_outputs, 100))

        opts = EMOptions(max_iter=20, diagonal_A=True, diagonal_Q=True, diagonal_R=True)
        result = em_algorithm(y, n_states=n_states, opts=opts)

        # A, Q, R should be diagonal
        def is_diagonal(M, tol=1e-10):
            return np.allclose(M, np.diag(np.diag(M)), atol=tol)

        assert is_diagonal(result.A)
        assert is_diagonal(result.Q)
        assert is_diagonal(result.R)

    def test_stability_constraint(self):
        """Test stability constraint enforcement."""
        rng = np.random.default_rng(42)

        n_states, n_outputs = 3, 2
        y = rng.standard_normal((n_outputs, 100))

        opts = EMOptions(max_iter=20, stable=True, max_eig=0.95)
        result = em_algorithm(y, n_states=n_states, opts=opts)

        # Eigenvalues should be within bounds
        eigenvalues = np.linalg.eigvals(result.A)
        assert np.all(np.abs(eigenvalues) <= 0.999)


class TestEMMultipleRealizations:
    """Test EM with multiple realizations."""

    def test_multiple_realizations(self):
        """Test EM with multiple data realizations."""
        rng = np.random.default_rng(42)

        true_sys = LinearSystem.random(n_states=2, n_outputs=2, stable=True, rng=rng)

        # Generate multiple realizations
        y_list = []
        for _ in range(5):
            y, _, _ = true_sys.simulate(n_steps=100, noise=True, rng=rng)
            y_list.append(y)

        # Run EM
        opts = EMOptions(max_iter=30, verbose=False)
        result = em_algorithm(y_list, n_states=2, opts=opts)

        assert result.A.shape == (2, 2)
        assert len(result.log_likelihood) > 0


class TestEMRecovery:
    """Test that EM can recover true parameters."""

    def test_recover_simple_system(self):
        """Test that EM recovers parameters of a simple system."""
        rng = np.random.default_rng(42)

        # Simple system with known parameters
        A_true = np.array([[0.8, 0.0], [0.0, 0.6]])
        C_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q_true = np.eye(2) * 0.05
        R_true = np.eye(2) * 0.1

        true_sys = LinearSystem(A=A_true, C=C_true, Q=Q_true, R=R_true)

        # Generate lots of data
        y, _, _ = true_sys.simulate(n_steps=1000, noise=True, rng=rng)

        # Run EM with good initialization
        opts = EMOptions(max_iter=200, tol=1e-6, verbose=False)
        result = em_algorithm(
            y, n_states=2,
            A0=A_true + rng.standard_normal((2, 2)) * 0.1,
            C0=C_true + rng.standard_normal((2, 2)) * 0.1,
            opts=opts
        )

        # The estimated system should have similar behavior
        # (not necessarily identical parameters due to rotational invariance)
        est_sys = LinearSystem(
            A=result.A, C=result.C, Q=result.Q, R=result.R,
            x0=result.x0, P0=result.P0
        )

        # Check that eigenvalues of A are similar
        true_eigs = np.sort(np.abs(np.linalg.eigvals(A_true)))
        est_eigs = np.sort(np.abs(np.linalg.eigvals(result.A)))

        assert_allclose(true_eigs, est_eigs, rtol=0.3)

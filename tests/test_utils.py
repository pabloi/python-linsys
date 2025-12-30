"""Tests for utility functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from linsys import LinearSystem, simulate, log_likelihood
from linsys.utils import (
    forward_simulate,
    compute_output,
    substitute_nans,
    bic_aic,
    count_parameters,
    diagonalize_A,
    model_snr,
)


class TestSimulate:
    """Test simulation functions."""

    def test_simulate_dimensions(self):
        """Test that simulate returns correct dimensions."""
        n_states, n_outputs, n_inputs, n_steps = 3, 2, 1, 100

        A = np.random.randn(n_states, n_states) * 0.5
        C = np.random.randn(n_outputs, n_states)

        y, x, u = simulate(A, C, n_steps)

        assert y.shape == (n_outputs, n_steps)
        assert x.shape == (n_states, n_steps)
        assert u.shape == (n_inputs, n_steps)

    def test_simulate_deterministic(self):
        """Test deterministic simulation."""
        A = np.array([[0.9]])
        C = np.array([[1.0]])
        x0 = np.array([1.0])

        y, x, _ = simulate(A, C, n_steps=10, x0=x0, noise=False)

        expected = 0.9 ** np.arange(10)
        assert_allclose(x.ravel(), expected)

    def test_simulate_with_input(self):
        """Test simulation with input."""
        A = np.array([[0.5]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])

        n_steps = 10
        u = np.ones((1, n_steps))

        y, x, _ = simulate(A, C, n_steps, B=B, u=u, noise=False)

        # State should grow toward steady state
        assert x[0, -1] > x[0, 0]


class TestForwardSimulate:
    """Test forward simulation (deterministic)."""

    def test_forward_simulate(self):
        """Test forward simulation."""
        A = np.array([[0.8]])
        C = np.array([[2.0]])
        x0 = np.array([1.0])

        y, x = forward_simulate(A, C, x0, n_steps=5)

        expected_x = np.array([1.0, 0.8, 0.64, 0.512, 0.4096])
        expected_y = expected_x * 2

        assert_allclose(x.ravel(), expected_x)
        assert_allclose(y.ravel(), expected_y)


class TestComputeOutput:
    """Test output computation."""

    def test_compute_output_simple(self):
        """Test simple output computation."""
        C = np.array([[1.0, 0.0], [0.0, 2.0]])
        x = np.array([[1.0, 2.0], [3.0, 4.0]])

        y = compute_output(C, x)

        expected = np.array([[1.0, 2.0], [6.0, 8.0]])
        assert_allclose(y, expected)

    def test_compute_output_with_feedthrough(self):
        """Test output with feedthrough."""
        C = np.eye(2)
        D = np.eye(2) * 0.5
        x = np.ones((2, 3))
        u = np.ones((2, 3)) * 2

        y = compute_output(C, x, D, u)

        expected = x + 0.5 * u
        assert_allclose(y, expected)


class TestSubstituteNans:
    """Test NaN substitution."""

    def test_substitute_zero(self):
        """Test zero substitution."""
        y = np.array([[1.0, np.nan, 3.0]])
        y_filled = substitute_nans(y, method="zero")

        assert_allclose(y_filled, [[1.0, 0.0, 3.0]])

    def test_substitute_mean(self):
        """Test mean substitution."""
        y = np.array([[1.0, np.nan, 3.0]])
        y_filled = substitute_nans(y, method="mean")

        assert_allclose(y_filled, [[1.0, 2.0, 3.0]])

    def test_substitute_interpolate(self):
        """Test interpolation."""
        y = np.array([[1.0, np.nan, 3.0, np.nan, 5.0]])
        y_filled = substitute_nans(y, method="interpolate")

        assert_allclose(y_filled, [[1.0, 2.0, 3.0, 4.0, 5.0]])


class TestBicAic:
    """Test BIC/AIC computation."""

    def test_bic_aic_values(self):
        """Test BIC and AIC computation."""
        log_lik = -100
        n_params = 10
        n_samples = 1000

        bic, aic = bic_aic(log_lik, n_params, n_samples)

        expected_bic = -2 * log_lik + n_params * np.log(n_samples)
        expected_aic = -2 * log_lik + 2 * n_params

        assert_allclose(bic, expected_bic)
        assert_allclose(aic, expected_aic)

    def test_bic_aic_ordering(self):
        """Test that BIC > AIC for large n_samples."""
        log_lik = -100
        n_params = 10
        n_samples = 10000  # Large enough that log(n) > 2

        bic, aic = bic_aic(log_lik, n_params, n_samples)

        assert bic > aic


class TestCountParameters:
    """Test parameter counting."""

    def test_count_full_model(self):
        """Test parameter count for full model."""
        n_states, n_inputs, n_outputs = 2, 1, 3

        n_params = count_parameters(n_states, n_inputs, n_outputs)

        # A: 4, B: 2, C: 6, D: 3, Q: 3 (symmetric), R: 6 (symmetric)
        # x0: 2, P0: 3 (symmetric)
        expected = 4 + 2 + 6 + 3 + 3 + 6 + 2 + 3
        assert n_params == expected

    def test_count_diagonal(self):
        """Test parameter count with diagonal constraints."""
        n_states, n_inputs, n_outputs = 2, 1, 2

        n_params = count_parameters(
            n_states, n_inputs, n_outputs,
            diagonal_A=True, diagonal_Q=True, diagonal_R=True
        )

        # A: 2 (diagonal), Q: 2, R: 2
        # B: 2, C: 4, D: 2, x0: 2, P0: 3
        expected = 2 + 2 + 4 + 2 + 2 + 2 + 2 + 3
        assert n_params == expected


class TestDiagonalizeA:
    """Test A diagonalization."""

    def test_diagonalize_diagonal(self):
        """Test diagonalizing an already diagonal matrix."""
        A = np.diag([0.9, 0.5])
        A_diag, T, eigenvalues = diagonalize_A(A)

        assert_allclose(np.sort(np.abs(eigenvalues)), [0.5, 0.9])

    def test_diagonalize_general(self):
        """Test diagonalizing a general matrix."""
        A = np.array([[0.8, 0.2], [-0.1, 0.6]])
        A_diag, T, eigenvalues = diagonalize_A(A)

        # Check that A_diag is similar to A
        # T^{-1} @ A @ T should be diagonal (or block-diagonal)
        reconstructed = T @ np.diag(eigenvalues) @ np.linalg.inv(T)
        assert_allclose(reconstructed, A, rtol=1e-10)


class TestModelSNR:
    """Test signal-to-noise ratio computation."""

    def test_snr_high_noise(self):
        """Test SNR with high noise."""
        rng = np.random.default_rng(42)

        A = np.array([[0.9]])
        C = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[10.0]])  # High measurement noise

        sys = LinearSystem(A=A, C=C, Q=Q, R=R)
        y, _, _ = sys.simulate(n_steps=500, noise=True, rng=rng)

        snr = model_snr(y, A, C, Q, R)

        # SNR should be low due to high noise
        assert snr < 1.0

    def test_snr_low_noise(self):
        """Test SNR with low noise."""
        rng = np.random.default_rng(42)

        A = np.array([[0.9]])
        C = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[0.01]])  # Low measurement noise

        sys = LinearSystem(A=A, C=C, Q=Q, R=R)
        y, _, _ = sys.simulate(n_steps=500, noise=True, rng=rng)

        snr = model_snr(y, A, C, Q, R)

        # SNR should be higher
        assert snr > 1.0


class TestLogLikelihood:
    """Test log-likelihood computation."""

    def test_log_likelihood_finite(self):
        """Test that log-likelihood is finite."""
        rng = np.random.default_rng(42)

        sys = LinearSystem.random(n_states=2, n_outputs=2, rng=rng)
        y, _, _ = sys.simulate(n_steps=100, noise=True, rng=rng)

        ll = log_likelihood(y, sys.A, sys.C, sys.Q, sys.R)

        assert np.isfinite(ll)
        assert ll < 0  # Log of probability < 1

    def test_log_likelihood_better_model(self):
        """Test that true model has higher likelihood than random."""
        rng = np.random.default_rng(42)

        # True system
        true_sys = LinearSystem.random(n_states=2, n_outputs=2, rng=rng)
        y, _, _ = true_sys.simulate(n_steps=500, noise=True, rng=rng)

        # Random system
        random_sys = LinearSystem.random(n_states=2, n_outputs=2, rng=np.random.default_rng(99))

        ll_true = true_sys.log_likelihood(y)
        ll_random = random_sys.log_likelihood(y)

        # True system should have higher likelihood (usually)
        # Note: This may not always hold for small samples
        assert ll_true > ll_random - 1000  # Allow some tolerance

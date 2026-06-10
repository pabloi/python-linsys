"""Tests for linsys.spca (smooth/dynamic PCA and dynamics estimation)."""
import numpy as np
import pytest

from linsys.spca import cv_spca, estimate_dyn, estimate_dyn_v3b, \
    estimate_dyn_v4, spca

TAUS = np.array([20.0, 60.0])
EIGS_TRUE = np.sort(np.exp(-1.0 / TAUS))  # ~ [0.951, 0.983]


def make_low_rank_data(N=200, ny=5, noise=0.01, offset=True, seed=3):
    """Y = C @ Xh (+ d0) + noise, with Xh decaying exponentials, x0 = 1."""
    rng = np.random.default_rng(seed)
    Xh = np.exp(-np.arange(N)[None, :] / TAUS[:, None])
    C = rng.standard_normal((ny, 2))
    d0 = rng.standard_normal((ny, 1)) if offset else np.zeros((ny, 1))
    Z = noise * rng.standard_normal((ny, N))
    Y = C @ Xh + d0 + Z
    return Y, C, Xh, d0, Z


# ------------------------------------------------------------------- spca
def test_spca_recovers_dynamics_and_reconstruction():
    Y, C, Xh, d0, Z = make_low_rank_data()
    m = spca(Y, dyn_order=2, rng=0)
    assert m.r2 > 0.999
    # Eigenvalues (time constants) close to truth
    np.testing.assert_allclose(np.sort(np.diag(m.J)), EIGS_TRUE, atol=5e-3)
    # Reconstruction error near the noise floor
    ones = np.ones((m.D.shape[1], Y.shape[1]))
    recon = np.hstack([m.C, m.D]) @ np.vstack([m.X, ones])
    rms = np.sqrt(np.mean((Y - recon) ** 2))
    noise_rms = np.sqrt(np.mean(Z ** 2))
    assert rms < 1.5 * noise_rms
    # Convention with constant terms: x(0) = 0 and unit-norm columns of C
    np.testing.assert_allclose(m.X[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(m.C ** 2, axis=0), 1.0, rtol=1e-12)
    # States satisfy the fitted dynamics exactly
    np.testing.assert_allclose(m.X[:, 1:], m.J @ m.X[:, :-1] + m.B,
                               atol=1e-10)


def test_spca_null_bd():
    Y, C, Xh, d0, Z = make_low_rank_data(offset=False)
    m = spca(Y, dyn_order=2, null_bd=True, rng=0)
    assert m.r2 > 0.999
    np.testing.assert_allclose(np.sort(np.diag(m.J)), EIGS_TRUE, atol=5e-3)
    assert m.D.shape[1] == 0
    np.testing.assert_allclose(m.B, 0.0)
    # Decaying convention: states follow X[k+1] = J X[k]
    np.testing.assert_allclose(m.X[:, 1:], m.J @ m.X[:, :-1], atol=1e-10)


def test_spca_force_pcs():
    Y, C, Xh, d0, Z = make_low_rank_data()
    m = spca(Y, dyn_order=2, force_pcs=True, rng=0)
    # No refinement iterations, but the PCA-based fit is already good
    assert m.r2 > 0.99
    np.testing.assert_allclose(np.sort(np.diag(m.J)), EIGS_TRUE, atol=2e-2)


# ----------------------------------------------------------- estimate_dyn
def test_estimate_dyn_recovers_clean_dynamics():
    Y, C, Xh, d0, Z = make_low_rank_data(ny=3, noise=0.0)
    res = estimate_dyn(Y, real_poles_only=True, null_k=False, j0=2, rng=0)
    np.testing.assert_allclose(np.sort(np.diag(res.J)), EIGS_TRUE, atol=1e-3)
    # X ~ [V K] @ Xh reconstruction is near-exact on clean data
    recon = np.hstack([res.V, res.K]) @ res.Xh
    assert np.linalg.norm(Y - recon) / np.linalg.norm(Y) < 1e-3
    assert res.resnorm < 1e-3
    # Xh includes the constant row when null_k=False
    assert res.Xh.shape == (3, Y.shape[1])
    np.testing.assert_allclose(res.Xh[-1], 1.0)
    np.testing.assert_allclose(res.Xh[:2, 0], 1.0)  # x(0) = 1 convention


def test_estimate_dyn_null_k():
    Y, C, Xh, d0, Z = make_low_rank_data(ny=3, noise=0.0, offset=False)
    res = estimate_dyn(Y, real_poles_only=True, null_k=True, j0=2, rng=0)
    np.testing.assert_allclose(np.sort(np.diag(res.J)), EIGS_TRUE, atol=1e-3)
    assert res.Xh.shape == (2, Y.shape[1])
    assert res.K.shape == (3, 0)


def test_estimate_dyn_matrix_initial_guess():
    Y, C, Xh, d0, Z = make_low_rank_data(ny=3, noise=0.0)
    J0 = np.diag([0.93, 0.99])
    res = estimate_dyn(Y, real_poles_only=True, null_k=False, j0=J0)
    np.testing.assert_allclose(np.sort(np.diag(res.J)), EIGS_TRUE, atol=1e-3)


def test_estimate_dyn_complex_poles_not_implemented():
    Y = np.zeros((2, 10))
    with pytest.raises(NotImplementedError):
        estimate_dyn(Y, real_poles_only=False, j0=2)


def test_estimate_dyn_v3b_alias():
    assert estimate_dyn_v3b is estimate_dyn


def test_estimate_dyn_v4_delegation():
    Y, C, Xh, d0, Z = make_low_rank_data(ny=3, noise=0.0, offset=False)
    # U = None / zero -> null_k=True path
    res = estimate_dyn_v4(Y, True, None, 2, rng=0)
    np.testing.assert_allclose(np.sort(np.diag(res.J)), EIGS_TRUE, atol=1e-3)
    res0 = estimate_dyn_v4(Y, True, np.zeros((1, Y.shape[1])), 2, rng=0)
    np.testing.assert_allclose(np.diag(res0.J), np.diag(res.J), atol=1e-6)
    # Constant non-zero U -> null_k=False path (constant term estimated)
    Yc, C2, Xh2, d02, Z2 = make_low_rank_data(ny=3, noise=0.0, offset=True)
    resc = estimate_dyn_v4(Yc, True, np.ones((1, Yc.shape[1])), 2, rng=0)
    np.testing.assert_allclose(np.sort(np.diag(resc.J)), EIGS_TRUE,
                               atol=1e-3)
    assert resc.K.shape == (3, 1)
    # Time-varying U: not implemented (as in MATLAB)
    with pytest.raises(NotImplementedError):
        estimate_dyn_v4(Y, True, np.arange(Y.shape[1])[None, :], 2)


# ---------------------------------------------------------------- cv_spca
def test_cv_spca_folds_consistent():
    Y, C, Xh, d0, Z = make_low_rank_data(N=200)  # divisible by n_folds
    models = cv_spca(Y, dyn_order=2, n_folds=2, rng=0)
    assert len(models) == 2
    for m in models:
        # J was re-scaled to per-sample dynamics: eigenvalues match truth
        np.testing.assert_allclose(np.sort(np.diag(m.J)), EIGS_TRUE,
                                   atol=1e-2)
        # X re-computed for every sample
        assert m.X.shape == (2, Y.shape[1])
        assert not np.isnan(m.X).any()
        # States satisfy the per-sample dynamics
        np.testing.assert_allclose(m.X[:, 1:], m.J @ m.X[:, :-1] + m.B,
                                   atol=1e-8)
        # Reconstruction at the per-sample scale matches the data
        ones = np.ones((m.D.shape[1], Y.shape[1]))
        recon = np.hstack([m.C, m.D]) @ np.vstack([m.X, ones])
        assert np.sqrt(np.mean((Y - recon) ** 2)) < 0.05
    # The folds describe the same underlying state trajectories (alignment
    # is approximate by construction: chngInitState modifies B, so the
    # initial conditions agree only up to ~one per-sample step)
    np.testing.assert_allclose(models[0].X, models[1].X, atol=0.05)

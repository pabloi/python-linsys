"""Tests for linsys.subspace (subspace identification)."""
import numpy as np
import pytest

from common import make_system, simulate
from linsys.subspace import (
    my_hankel, observability_matrix, project_mat, project_obliq,
    project_perp, subspace_id, subspace_id_unbiased, subspace_id_v2,
)


def markov_params(A, B, C, D, k=5):
    """I/O invariants: [D, C@B, C@A@B, ..., C@A^(k-1)@B]."""
    out = [D]
    Ak = np.eye(A.shape[0])
    for _ in range(k):
        out.append(C @ Ak @ B)
        Ak = Ak @ A
    return out


def max_markov_rel_err(true_sys, est, k=5):
    mt = markov_params(*true_sys, k=k)
    mh = markov_params(est.A, est.B, est.C, est.D, k=k)
    return max(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)
               for a, b in zip(mt, mh))


@pytest.fixture(scope="module")
def system():
    return make_system(nx=2, ny=3, nu=1, seed=0)


@pytest.fixture(scope="module")
def clean_data(system):
    A, B, C, D, Q, R = system
    # White-noise input (persistent excitation), no noise
    Y, X, U = simulate(A, B, C, D, np.zeros((2, 2)), np.zeros((3, 3)),
                       N=400, seed=1, step_input=False)
    return Y, U


@pytest.fixture(scope="module")
def noisy_data(system):
    A, B, C, D, Q, R = system
    Y, X, U = simulate(A, B, C, D, Q, R, N=600, seed=2, step_input=False)
    return Y, U


# ---------------------------------------------------------------- helpers
def test_my_hankel_blocks():
    A = np.arange(8.0).reshape(2, 4)
    H = my_hankel(A, 2, 3)
    assert H.shape == (4, 3)
    # column l of H is [a_l; a_{l+1}]
    np.testing.assert_allclose(H[:, 1], np.r_[A[:, 1], A[:, 2]])


def test_projections():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((2, 30))
    A = rng.standard_normal((3, 30))
    Ap = project_mat(A, B)
    Aperp = project_perp(A, B)
    np.testing.assert_allclose(Ap + Aperp, A)
    np.testing.assert_allclose(Aperp @ B.T, 0, atol=1e-10)
    # Oblique projection onto C along B: reconstructs A when A in rowspace(C)
    C = rng.standard_normal((4, 30))
    A2 = rng.standard_normal((3, 4)) @ C
    Aob, _ = project_obliq(A2, B, C)
    # A2 fully in rowspace(C): oblique projection leaves the C-part intact
    assert np.linalg.norm(Aob - A2) / np.linalg.norm(A2) < 0.5


def test_observability_matrix(system):
    A, B, C, D, Q, R = system
    L = observability_matrix(A, C, 3)
    np.testing.assert_allclose(L, np.vstack([C, C @ A, C @ A @ A]))


# ----------------------------------------------------- deterministic data
def test_subspace_id_unbiased_exact_on_clean_data(system, clean_data):
    A, B, C, D, Q, R = system
    Y, U = clean_data
    res = subspace_id_unbiased(Y, U, 2)
    ev = np.sort(np.linalg.eigvals(res.A).real)
    np.testing.assert_allclose(ev, [0.7, 0.95], atol=1e-8)
    assert max_markov_rel_err((A, B, C, D), res) < 1e-8
    # No noise: estimated covariances are ~0
    assert np.linalg.norm(res.Q) < 1e-12
    assert np.linalg.norm(res.R) < 1e-12


def test_subspace_id_v2_exact_on_clean_data(system, clean_data):
    A, B, C, D, Q, R = system
    Y, U = clean_data
    res = subspace_id_v2(Y, U, 2)
    ev = np.sort(np.linalg.eigvals(res.A).real)
    np.testing.assert_allclose(ev, [0.7, 0.95], atol=1e-8)
    assert max_markov_rel_err((A, B, C, D), res) < 1e-8


def test_subspace_id_eigenvalues_on_clean_data(system, clean_data):
    # The biased algorithm recovers eigenvalues well even though the
    # state-basis shift makes B/D slightly inconsistent
    A, B, C, D, Q, R = system
    Y, U = clean_data
    res = subspace_id(Y, U, 2)
    ev = np.sort(np.linalg.eigvals(res.A).real)
    np.testing.assert_allclose(ev, [0.7, 0.95], atol=5e-3)
    assert max_markov_rel_err((A, B, C, D), res) < 0.5


# ------------------------------------------------------------- noisy data
@pytest.mark.parametrize("fn", [subspace_id, subspace_id_unbiased])
def test_recovers_noisy_system(system, noisy_data, fn):
    A, B, C, D, Q, R = system
    Y, U = noisy_data
    res = fn(Y, U, 2)
    ev = np.sort(np.linalg.eigvals(res.A).real)
    np.testing.assert_allclose(ev, [0.7, 0.95], atol=0.05)
    assert max_markov_rel_err((A, B, C, D), res) < 0.2
    # State trajectories explain the data
    recon = res.C @ res.X + res.D @ U
    assert np.linalg.norm(Y - recon) / np.linalg.norm(Y) < 0.2


def test_shapes_and_state_estimate(system, noisy_data):
    Y, U = noisy_data
    res = subspace_id_unbiased(Y, U, 2)
    ny, N = Y.shape
    assert res.A.shape == (2, 2)
    assert res.B.shape == (2, 1)
    assert res.C.shape == (ny, 2)
    assert res.D.shape == (ny, 1)
    assert res.X.shape == (2, N)
    assert res.Q.shape == (2, 2)
    assert res.R.shape == (ny, ny)
    assert res.S.shape == (2, ny)


def test_default_order_warns(system, clean_data):
    Y, U = clean_data
    with pytest.warns(UserWarning, match="Automatic state number"):
        res = subspace_id(Y, U)
    assert res.A.shape == (2, 2)


def test_too_few_samples_raises():
    Y = np.zeros((2, 15))
    U = np.ones((1, 15))
    with pytest.raises(ValueError, match="not enough samples"):
        subspace_id(Y, U, 2, i=10)

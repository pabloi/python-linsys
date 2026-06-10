"""Tests for linsys.model_selection and linsys.misc_helpers."""
import numpy as np
import pytest

from linsys.model_selection import bic_aic, fold_split, best_paired_match
from linsys.misc_helpers import (match_model_inputs, get_flat_model,
                                 rotate_fac, my_psd_sum, linearize,
                                 calc_output, update_state, model_check)
from linsys.utils import cholcov

from common import make_system, simulate


class TestBicAic:
    def test_hand_computation(self):
        model = dict(J=np.diag([0.9, 0.5]), B=np.ones((2, 1)),
                     C=np.ones((3, 2)), D=np.ones((3, 1)),
                     Q=0.1 * np.eye(2), R=0.1 * np.eye(3))
        N = 100
        Y = np.zeros((3, N))
        logL = -123.0
        # nonzero counts: Na=2, Nb=2-2=0, Nc=6, Nd=3, Nq=2, Nr=3 -> k=16
        k = 16
        k_alt = k + 2 * N - 2 - 3 - 2 - 0
        out = bic_aic(model, Y, logL)
        np.testing.assert_allclose(out.BIC, 2 * logL - np.log(N) * k)
        np.testing.assert_allclose(out.AIC, 2 * logL - 2 * k)
        np.testing.assert_allclose(out.BICalt, 2 * logL - np.log(N) * k_alt)

    def test_accepts_a_key(self):
        model = dict(A=np.eye(1), B=np.ones((1, 1)), C=np.ones((2, 1)),
                     D=np.zeros((2, 1)), Q=np.eye(1), R=np.eye(2))
        out = bic_aic(model, np.zeros((2, 50)), 0.0)
        assert np.isfinite(out.BIC) and np.isfinite(out.AIC)


class TestFoldSplit:
    def test_partition(self):
        data = np.arange(12, dtype=float).reshape(2, 6)
        folds = fold_split(data, 3)
        assert len(folds) == 3
        for i, f in enumerate(folds):
            assert f.shape == data.shape
            obs = ~np.isnan(f[0])
            np.testing.assert_array_equal(np.nonzero(obs)[0],
                                          np.arange(i, 6, 3))
            np.testing.assert_array_equal(f[:, obs], data[:, obs])
        # Each time sample appears in exactly one fold:
        count = sum((~np.isnan(f)).astype(int) for f in folds)
        np.testing.assert_array_equal(count, np.ones_like(data))

    def test_matlab_axis(self):
        data = np.arange(8, dtype=float).reshape(4, 2)
        folds = fold_split(data, 2, axis=0)
        assert np.isnan(folds[0][1]).all() and not np.isnan(folds[0][0]).any()


class TestBestPairedMatch:
    def test_exact_match(self):
        v1 = np.array([3.0, 1.0, 2.0])
        v2 = np.array([1.0, 2.0, 3.0])
        ind1, ind2 = best_paired_match(v1, v2)
        np.testing.assert_array_equal(ind2, [0, 1, 2])
        np.testing.assert_array_equal(v1[ind1], v2)

    def test_unequal_lengths(self):
        v1 = np.array([10.0, 1.0, 5.0])
        v2 = np.array([1.1])
        ind1, ind2 = best_paired_match(v1, v2)
        assert ind1[0] == 1  # closest to 1.1
        assert sorted(ind1) == [0, 1, 2]  # leftovers appended
        np.testing.assert_array_equal(ind2, [0])


class TestMatchModelInputs:
    def test_padding(self):
        N = 5
        u1 = np.ones((1, N))
        u2 = np.vstack([np.ones((1, N)), np.arange(N, dtype=float)])
        m1 = dict(B=np.array([[2.0]]), D=np.array([[3.0]]))
        m2 = dict(B=np.array([[1.0, 4.0]]), D=np.array([[5.0, 6.0]]))
        (e1, e2), eu = match_model_inputs([m1, m2], [u1, u2])
        assert eu.shape[1] == N and eu.shape[0] == 2  # two unique rows
        assert e1["B"].shape == (1, 2) and e2["B"].shape == (1, 2)
        # Model outputs preserved: B_new @ eu == B_old @ u_old
        np.testing.assert_allclose(e1["B"] @ eu, m1["B"] @ u1)
        np.testing.assert_allclose(e2["B"] @ eu, m2["B"] @ u2)
        np.testing.assert_allclose(e1["D"] @ eu, m1["D"] @ u1)
        np.testing.assert_allclose(e2["D"] @ eu, m2["D"] @ u2)


class TestGetFlatModel:
    def test_flat_model(self):
        A, B, C, D, Q, R = make_system(nx=2, ny=3, nu=1, seed=0)
        Y, X, U = simulate(A, B, C, D, Q, R, N=150, seed=1)
        fm = get_flat_model(Y, U)
        assert fm.J.shape == (1, 1) and fm.J[0, 0] == 0
        assert fm.C.shape == (3, 1)
        # D is the least-squares static fit
        Dls = np.linalg.lstsq(U.T, Y.T, rcond=None)[0].T
        np.testing.assert_allclose(fm.D, Dls, atol=1e-10)
        assert np.isfinite(fm.logL)

    def test_include_output_idx(self):
        A, B, C, D, Q, R = make_system(nx=2, ny=3, nu=1, seed=0)
        Y, X, U = simulate(A, B, C, D, Q, R, N=100, seed=2)
        fm = get_flat_model(Y, U, include_output_idx=[0, 2])
        assert np.isinf(fm.R[1, 1])
        assert np.isfinite(fm.R[0, 0]) and np.isfinite(fm.R[2, 2])
        assert np.isfinite(fm.logL)


class TestRotateFac:
    def test_product_preserved(self):
        rng = np.random.default_rng(0)
        CD = rng.standard_normal((8, 3))
        XU = rng.standard_normal((3, 50))
        Y = CD @ XU
        for method in ("orthonormal", "varimax", "quartimax", "orthomax",
                       "pablo", "none"):
            CDr, XUr = rotate_fac(CD, XU, method=method)
            np.testing.assert_allclose(CDr @ XUr, Y, atol=1e-8, err_msg=method)

    def test_orthonormal_columns(self):
        rng = np.random.default_rng(1)
        CD = rng.standard_normal((8, 3))
        XU = rng.standard_normal((3, 50))
        CDr, _ = rotate_fac(CD, XU, method="orthonormal")
        np.testing.assert_allclose(CDr.T @ CDr, np.eye(3), atol=1e-10)


class TestMyPsdSum:
    def test_sum_and_psd(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((4, 4))
        b = rng.standard_normal((4, 4))
        A = a @ a.T
        B = b @ b.T
        cC, C = my_psd_sum(A, B)
        np.testing.assert_allclose(C, A + B, atol=1e-9)
        np.testing.assert_allclose(cC.T @ cC, A + B, atol=1e-9)
        assert (np.linalg.eigvalsh(C) > -1e-12).all()

    def test_semidefinite_input(self):
        v = np.array([[1.0], [2.0], [0.0]])
        A = v @ v.T  # rank 1, PSD
        B = np.eye(3)
        cC, C = my_psd_sum(A, B)
        np.testing.assert_allclose(C, A + B, atol=1e-9)

    def test_chol_input(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal((3, 3))
        A = a @ a.T + 3 * np.eye(3)
        cA, _ = cholcov(A)
        B = np.eye(3)
        _, C = my_psd_sum(cA, B)
        np.testing.assert_allclose(C, A + B, atol=1e-9)


class TestSmallHelpers:
    def test_linearize_linear_function(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        C = linearize(lambda x: M @ x, np.array([0.3, -0.2]))
        np.testing.assert_allclose(C, M, atol=1e-8)

    def test_calc_output_update_state_noiseless(self):
        A = np.diag([0.9, 0.5])
        B = np.ones((2, 1))
        C = np.ones((3, 2))
        D = np.zeros((3, 1))
        x = np.array([1.0, 2.0])
        u = np.array([1.0])
        y = calc_output(x, u, C, D, R=np.zeros((3, 3)), rng=0)
        np.testing.assert_allclose(y, C @ x)
        xn = update_state(x, u, A, B, Q=np.zeros((2, 2)), rng=0)
        np.testing.assert_allclose(xn, A @ x + B @ u)

    def test_calc_output_noise_statistics(self):
        C = np.zeros((2, 1))
        D = np.zeros((2, 1))
        R = np.diag([4.0, 9.0])
        rng = np.random.default_rng(4)
        ys = np.array([calc_output(np.zeros(1), np.zeros(1), C, D, R=R,
                                   rng=rng) for _ in range(4000)])
        np.testing.assert_allclose(ys.var(axis=0), [4.0, 9.0], rtol=0.15)

    def test_model_check(self):
        A, B, C, D, Q, R = make_system(nx=2, ny=3, nu=1, seed=0)
        checks = model_check(A, C, Q, R)
        assert checks["observable"]
        assert checks["q_psd"] and checks["r_psd"] and checks["r_invertible"]
        with pytest.raises(ValueError):
            model_check(A, C, Q, np.zeros((3, 3)))  # R not invertible

import numpy as np
import pytest

from linsys.utils import (cholcov, cholcov2, pinvchol, pinvchol2, logl_normal,
                          transform, diagonalize_a, canonize, substitute_nans,
                          rob_cov, fwd_sim)

RNG = np.random.default_rng(0)


def rand_psd(n, rank=None, rng=RNG):
    rank = n if rank is None else rank
    M = rng.standard_normal((n, rank))
    return M @ M.T


class TestCholcov:
    def test_pd(self):
        A = rand_psd(5)
        U, r = cholcov(A)
        assert r == 5
        np.testing.assert_allclose(U.T @ U, A, atol=1e-10)

    def test_psd_rank_deficient(self):
        A = rand_psd(6, rank=3)
        U, r = cholcov(A)
        assert r == 3
        assert U.shape == (3, 6)
        np.testing.assert_allclose(U.T @ U, A, atol=1e-6)

    def test_zero(self):
        U, r = cholcov(np.zeros((4, 4)))
        assert r == 0
        assert U.shape == (0, 4)

    def test_cholcov2_psd(self):
        A = rand_psd(5, rank=2)
        cA = cholcov2(A)
        assert cA.shape == (5, 5)
        np.testing.assert_allclose(cA.T @ cA, A, atol=1e-8)

    def test_cholcov2_inf_diag(self):
        A = np.diag([1.0, np.inf, 2.0])
        cA = cholcov2(A)
        assert np.isinf(cA[1, 1])
        np.testing.assert_allclose(cA[0, 0] ** 2, 1.0)


class TestPinvchol:
    def test_pd(self):
        A = rand_psd(5)
        cInvA, cA, invA = pinvchol(A)
        np.testing.assert_allclose(invA, np.linalg.inv(A), atol=1e-8)
        np.testing.assert_allclose(cInvA @ cInvA.T, np.linalg.inv(A), atol=1e-8)

    def test_psd(self):
        A = rand_psd(6, rank=4)
        cInvA, cA, invA = pinvchol(A)
        np.testing.assert_allclose(invA, np.linalg.pinv(A), atol=1e-6)

    def test_pinvchol2_inf(self):
        A = np.diag([2.0, np.inf, 4.0])
        cInvA, cA, invA = pinvchol2(A)
        np.testing.assert_allclose(invA, np.diag([0.5, 0.0, 0.25]), atol=1e-12)

    def test_pinvchol2_general(self):
        A = rand_psd(5)
        _, _, invA = pinvchol2(A)
        np.testing.assert_allclose(invA, np.linalg.inv(A), atol=1e-8)


class TestLoglNormal:
    def test_matches_direct_formula(self):
        S = rand_psd(3)
        y = RNG.standard_normal((3, 10))
        logL, z2 = logl_normal(y, S)
        iS = np.linalg.inv(S)
        expected = (-0.5 * np.einsum('in,ij,jn->n', y, iS, y)
                    - 0.5 * np.log(np.linalg.det(S))
                    - 1.5 * np.log(2 * np.pi))
        np.testing.assert_allclose(logL, expected, atol=1e-10)


class TestTransform:
    def test_roundtrip(self):
        nx, ny, nu = 3, 4, 2
        A = 0.5 * np.eye(nx) + 0.1 * RNG.standard_normal((nx, nx))
        B = RNG.standard_normal((nx, nu))
        C = RNG.standard_normal((ny, nx))
        Q = rand_psd(nx)
        V = RNG.standard_normal((nx, nx)) + 3 * np.eye(nx)
        A2, B2, C2, Q2, _, _ = transform(V, A, B, C, Q)
        # output behavior is invariant: C2 (V A V^-1)^k V B == C A^k B
        for k in range(4):
            lhs = C2 @ np.linalg.matrix_power(A2, k) @ B2
            rhs = C @ np.linalg.matrix_power(A, k) @ B
            np.testing.assert_allclose(lhs, rhs, atol=1e-8)


class TestDiagonalizeA:
    def test_real_eigs(self):
        A = np.diag([0.9, 0.5]) + 0.01 * RNG.standard_normal((2, 2))
        V, J = diagonalize_a(A)
        np.testing.assert_allclose(V @ J @ np.linalg.inv(V), A, atol=1e-10)
        assert np.all(np.diff(np.diag(J)) >= 0)

    def test_complex_eigs_real_jordan(self):
        th = 0.3
        A = 0.9 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        V, J = diagonalize_a(A)
        assert np.isrealobj(J) and np.isrealobj(V)
        np.testing.assert_allclose(V @ J @ np.linalg.inv(V), A, atol=1e-10)


class TestCanonize:
    @pytest.mark.parametrize("method", ["canonical", "canonicalAlt",
                                        "orthonormal", "varimax"])
    def test_io_behavior_preserved(self, method):
        nx, ny, nu = 3, 5, 2
        A = np.diag([0.95, 0.8, 0.5])
        B = RNG.standard_normal((nx, nu))
        C = RNG.standard_normal((ny, nx))
        Q = rand_psd(nx)
        A2, B2, C2, _, V, Q2, _ = canonize(A, B, C, Q=Q, method=method)
        for k in range(4):
            lhs = C2 @ np.linalg.matrix_power(A2, k) @ B2
            rhs = C @ np.linalg.matrix_power(A, k) @ B
            np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    def test_canonical_is_idempotent_signature(self):
        # canonizing two equivalent models gives the same parameters
        nx, ny, nu = 2, 4, 1
        A = np.diag([0.9, 0.6])
        B = np.ones((nx, nu))
        C = RNG.standard_normal((ny, nx))
        Q = rand_psd(nx)
        T = RNG.standard_normal((nx, nx)) + 2 * np.eye(nx)
        At, Bt, Ct, Qt, _, _ = transform(T, A, B, C, Q)
        A1, B1, C1, _, _, Q1, _ = canonize(A, B, C, Q=Q, method="canonical")
        A2, B2, C2, _, _, Q2, _ = canonize(At, Bt, Ct, Q=Qt, method="canonical")
        np.testing.assert_allclose(A1, A2, atol=1e-8)
        np.testing.assert_allclose(B1, B2, atol=1e-8)
        np.testing.assert_allclose(C1, C2, atol=1e-8)
        np.testing.assert_allclose(Q1, Q2, atol=1e-8)


class TestSubstituteNans:
    def test_interpolates(self):
        y = np.array([[0.0, 1.0, np.nan, 3.0, np.nan]]).T
        out = substitute_nans(y)
        np.testing.assert_allclose(out.ravel(), [0, 1, 2, 3, 4])


class TestRobCov:
    def test_close_to_cov_for_clean_gaussian(self):
        rng = np.random.default_rng(42)
        S = np.array([[2.0, 0.5], [0.5, 1.0]])
        L = np.linalg.cholesky(S)
        w = L @ rng.standard_normal((2, 20000))
        Q = rob_cov(w)
        np.testing.assert_allclose(Q, S, rtol=0.1)


class TestFwdSim:
    def test_deterministic(self):
        A = np.array([[0.9]])
        B = np.array([[0.1]])
        C = np.array([[2.0]])
        D = np.array([[0.0]])
        u = np.ones((1, 50))
        y, x = fwd_sim(u, A, B, C, D)
        # steady state: x_inf = B/(1-A) = 1, y_inf = 2
        np.testing.assert_allclose(x[0, -1], 1.0, atol=1e-2)
        np.testing.assert_allclose(y[0, -1], 2.0, atol=2e-2)

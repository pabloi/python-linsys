"""Tests for the Kalman filter/smoother variants: square-root, constrained,
CS2006 and legacy (v1) information forms."""
import numpy as np
import pytest

from linsys.kalman import (
    filter_stationary, smoother_stationary,
    sqrt_filter_stationary, sqrt_smoother_stationary,
    filter_stationary_constrained, smoother_stationary_constrained,
    filter_stationary_w_constraint,
    smoother_stationary_cs2006,
    info_filter_stationary_v1, info_smoother_stationary_v1, info_update2,
    KalmanOpts,
)
from common import make_system, simulate, naive_kalman_filter, naive_rts_smoother


@pytest.fixture(scope="module")
def sys_and_data():
    A, B, C, D, Q, R = make_system(nx=2, ny=3, nu=1, seed=0)
    Y, X, U = simulate(A, B, C, D, Q, R, N=300, seed=1)
    return A, B, C, D, Q, R, Y, X, U


class TestSqrtFilter:
    def test_matches_filter_stationary(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        r_cov = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        r_sqrt = sqrt_filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(r_sqrt.X, r_cov.X, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.P, r_cov.P, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.Xp, r_cov.Xp, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.Pp, r_cov.Pp, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.logL, r_cov.logL, rtol=1e-9)

    def test_matches_naive_textbook(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        res = sqrt_filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        Xn, Pn, Xpn, Ppn, logLn = naive_kalman_filter(Y, A, C, Q, R, x0, P0,
                                                      B, D, U)
        np.testing.assert_allclose(res.X, Xn, atol=1e-8)
        np.testing.assert_allclose(res.P, Pn, atol=1e-8)
        np.testing.assert_allclose(res.logL, logLn, rtol=1e-8)

    def test_non_diagonal_prior_and_noises(self, sys_and_data):
        # Exercises the factor-convention deviation from MATLAB (which is
        # only correct for diagonal P0/Q/R).
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        rng = np.random.default_rng(3)
        L = rng.standard_normal((2, 2))
        P0 = np.eye(2) + 0.5 * (L @ L.T)
        x0 = rng.standard_normal(2)
        Qn = Q + 5e-4 * np.array([[0.0, 1.0], [1.0, 0.0]])
        M = rng.standard_normal((3, 3))
        Rn = R + 1e-3 * (M @ M.T)
        r_cov = filter_stationary(Y, A, C, Qn, Rn, x0, P0, B, D, U)
        r_sqrt = sqrt_filter_stationary(Y, A, C, Qn, Rn, x0, P0, B, D, U)
        np.testing.assert_allclose(r_sqrt.X, r_cov.X, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.P, r_cov.P, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.logL, r_cov.logL, rtol=1e-9)

    def test_nan_samples(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yn = Y.copy()
        Yn[:, 40:45] = np.nan
        x0, P0 = np.zeros(2), np.eye(2)
        r_cov = filter_stationary(Yn, A, C, Q, R, x0, P0, B, D, U)
        r_sqrt = sqrt_filter_stationary(Yn, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(r_sqrt.X, r_cov.X, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.P, r_cov.P, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.logL, r_cov.logL, rtol=1e-9)

    def test_improper_prior_matches_filter(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        r_cov = filter_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
        r_sqrt = sqrt_filter_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
        np.testing.assert_allclose(r_sqrt.X, r_cov.X, atol=1e-7)
        np.testing.assert_allclose(r_sqrt.P, r_cov.P, atol=1e-7)
        np.testing.assert_allclose(r_sqrt.logL, r_cov.logL, rtol=1e-7)

    def test_fast_mode_matches_exact(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        r_exact = sqrt_filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        r_fast = sqrt_filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U,
                                        KalmanOpts(fast_flag=1))
        np.testing.assert_allclose(r_fast.X, r_exact.X, atol=1e-5)
        np.testing.assert_allclose(r_fast.logL, r_exact.logL, rtol=1e-5)

    def test_outlier_rejection(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yo = Y.copy()
        Yo[:, 100] += 100.0
        x0, P0 = np.zeros(2), np.eye(2)
        res = sqrt_filter_stationary(Yo, A, C, Q, R, x0, P0, B, D, U,
                                     KalmanOpts(outlier_flag=True,
                                                no_reduce_flag=True))
        assert res.rejected[100]
        ref = filter_stationary(Yo, A, C, Q, R, x0, P0, B, D, U,
                                KalmanOpts(outlier_flag=True,
                                           no_reduce_flag=True))
        np.testing.assert_allclose(res.X, ref.X, atol=1e-8)


class TestSqrtSmoother:
    def test_matches_smoother_stationary(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        r_cov = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        r_sqrt = sqrt_smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(r_sqrt.Xs, r_cov.Xs, atol=1e-8)
        np.testing.assert_allclose(r_sqrt.Ps, r_cov.Ps, atol=1e-8)
        np.testing.assert_allclose(r_sqrt.Pt, r_cov.Pt, atol=1e-8)
        np.testing.assert_allclose(r_sqrt.Xf, r_cov.Xf, atol=1e-9)
        np.testing.assert_allclose(r_sqrt.logL, r_cov.logL, rtol=1e-9)

    def test_matches_naive_rts(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        res = sqrt_smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        Xs, Ps, Pt, _, _, logL = naive_rts_smoother(Y, A, C, Q, R, x0, P0,
                                                    B, D, U)
        np.testing.assert_allclose(res.Xs, Xs, atol=1e-7)
        np.testing.assert_allclose(res.Ps, Ps, atol=1e-7)
        np.testing.assert_allclose(res.Pt, Pt, atol=1e-7)
        np.testing.assert_allclose(res.logL, logL, rtol=1e-8)

    def test_improper_prior(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        r_cov = smoother_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
        r_sqrt = sqrt_smoother_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
        np.testing.assert_allclose(r_sqrt.Xs, r_cov.Xs, atol=1e-6)
        np.testing.assert_allclose(r_sqrt.Ps, r_cov.Ps, atol=1e-6)


class TestConstrained:
    def test_no_constraint_matches_unconstrained(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        ref = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        res = filter_stationary_constrained(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(res.X, ref.X, atol=1e-7)
        np.testing.assert_allclose(res.P, ref.P, atol=1e-7)
        np.testing.assert_allclose(res.Xp, ref.Xp, atol=1e-7)

    def test_trivial_constraint_matches_unconstrained(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)

        def empty_constraint(x):  # no actual constraint
            return np.zeros((0, 2)), np.zeros(0)

        ref = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        res = filter_stationary_constrained(Y, A, C, Q, R, x0, P0, B, D, U,
                                            constr_fun=empty_constraint)
        np.testing.assert_allclose(res.X, ref.X, atol=1e-6)
        np.testing.assert_allclose(res.P, ref.P, atol=1e-6)

    def test_active_constraint_satisfied(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        H = np.array([[1.0, 1.0]])
        e = np.array([1.0])

        res = filter_stationary_constrained(Y, A, C, Q, R, x0, P0, B, D, U,
                                            constr_fun=lambda x: (H, e))
        # All filtered states satisfy H x = e exactly:
        np.testing.assert_allclose(H @ res.X, np.ones((1, Y.shape[1])),
                                   atol=1e-9)
        # And the projected covariance has (numerically) no uncertainty
        # along the constrained direction:
        var_constr = np.einsum('ij,njk,lk->nil', H, res.P, H)
        assert np.max(np.abs(var_constr)) < 1e-9

    def test_constrained_smoother(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        # With no constraint, the constrained smoother is the RTS smoother:
        ref = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        res = smoother_stationary_constrained(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(res.Xs, ref.Xs, atol=1e-6)
        np.testing.assert_allclose(res.Ps, ref.Ps, atol=1e-6)
        np.testing.assert_allclose(res.Pt, ref.Pt, atol=1e-6)

        # With an active constraint, the (filter-enforced) constraint holds:
        H = np.array([[1.0, 1.0]])
        e = np.array([1.0])
        res_c = smoother_stationary_constrained(Y, A, C, Q, R, x0, P0, B, D, U,
                                                constr_fun=lambda x: (H, e))
        np.testing.assert_allclose(H @ res_c.Xf, np.ones((1, Y.shape[1])),
                                   atol=1e-9)
        assert np.all(np.isfinite(res_c.Xs))

    def test_soft_constraint_filter(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        H = np.array([[1.0, 1.0]])
        e = np.array([1.0])
        S = np.array([[1e-10]])
        res = filter_stationary_w_constraint(
            Y, A, C, Q, R, x0, P0, B, D, U,
            constr_fun=lambda x: (H, e, S))
        # Soft constraint with tiny variance is (nearly) enforced:
        assert np.max(np.abs(H @ res.X - 1.0)) < 1e-3
        # And with no constraint it matches the unconstrained filter:
        ref = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        res0 = filter_stationary_w_constraint(Y, A, C, Q, R, x0, P0, B, D, U)
        # (predict-first convention: x0/P0 act one step earlier)
        x0b = A @ x0 + B @ U[:, 0]
        P0b = A @ P0 @ A.T + Q
        ref_b = filter_stationary(Y, A, C, Q, R, x0b, P0b, B, D, U)
        np.testing.assert_allclose(res0.X, ref_b.X, atol=1e-7)


class TestCS2006:
    def test_matches_naive_rts(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        res = smoother_stationary_cs2006(Y, A, C, Q, R, x0, P0, B, D, U)
        Xs, Ps, Pt, _, _, logL = naive_rts_smoother(Y, A, C, Q, R, x0, P0,
                                                    B, D, U)
        np.testing.assert_allclose(res.Xs, Xs, atol=1e-7)
        np.testing.assert_allclose(res.Ps, Ps, atol=1e-7)
        np.testing.assert_allclose(res.Pt, Pt, atol=1e-7)
        np.testing.assert_allclose(res.logL, logL, rtol=1e-8)

    def test_matches_smoother_stationary(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        ref = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        res = smoother_stationary_cs2006(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(res.Xs, ref.Xs, atol=1e-7)
        np.testing.assert_allclose(res.Ps, ref.Ps, atol=1e-7)
        np.testing.assert_allclose(res.Pt, ref.Pt, atol=1e-7)
        np.testing.assert_allclose(res.logL, ref.logL, rtol=1e-8)

    def test_no_reduce_matches_reduced(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        r1 = smoother_stationary_cs2006(Y, A, C, Q, R, x0, P0, B, D, U)
        r2 = smoother_stationary_cs2006(Y, A, C, Q, R, x0, P0, B, D, U,
                                        KalmanOpts(no_reduce_flag=True))
        np.testing.assert_allclose(r1.Xs, r2.Xs, atol=1e-7)
        np.testing.assert_allclose(r1.logL, r2.logL, rtol=1e-7)

    def test_nan_samples(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yn = Y.copy()
        Yn[:, 150:160] = np.nan
        x0, P0 = np.zeros(2), np.eye(2)
        ref = smoother_stationary(Yn, A, C, Q, R, x0, P0, B, D, U)
        res = smoother_stationary_cs2006(Yn, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(res.Xs, ref.Xs, atol=1e-7)
        np.testing.assert_allclose(res.Ps, ref.Ps, atol=1e-7)

    def test_improper_prior_tracks(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        with pytest.warns(UserWarning):
            res = smoother_stationary_cs2006(Y, A, C, Q, R, B=B, D=D, U=U)
        err = res.Xs[:, 10:] - X[:, 10:-1]
        assert np.sqrt(np.mean(err ** 2)) < 0.2


class TestInfoV1:
    def test_filter_matches_covariance_form(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        rk = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        ri = info_filter_stationary_v1(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(ri.X, rk.X, atol=1e-6)
        np.testing.assert_allclose(ri.P, rk.P, atol=1e-6)
        np.testing.assert_allclose(ri.Xp, rk.Xp, atol=1e-6)
        np.testing.assert_allclose(ri.Pp, rk.Pp, atol=1e-6)
        np.testing.assert_allclose(ri.logL, rk.logL, rtol=1e-6)

    def test_filter_info_outputs_consistent(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        ri = info_filter_stationary_v1(Y, A, C, Q, R, x0, P0, B, D, U)
        # I[k] is the posterior information: inv(P[k])
        for k in (0, 10, 100):
            np.testing.assert_allclose(ri.I[k] @ ri.P[k], np.eye(2),
                                       atol=1e-6)
            np.testing.assert_allclose(ri.Ip[k] @ ri.Pp[k], np.eye(2),
                                       atol=1e-6)

    def test_filter_nan_samples(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yn = Y.copy()
        Yn[:, 40:45] = np.nan
        x0, P0 = np.zeros(2), np.eye(2)
        rk = filter_stationary(Yn, A, C, Q, R, x0, P0, B, D, U)
        ri = info_filter_stationary_v1(Yn, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(ri.X, rk.X, atol=1e-6)
        np.testing.assert_allclose(ri.logL, rk.logL, rtol=1e-6)

    def test_smoother_matches_rts(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        rs = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        ri = info_smoother_stationary_v1(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(ri.Xs, rs.Xs, atol=1e-5)
        np.testing.assert_allclose(ri.Ps, rs.Ps, atol=1e-5)
        np.testing.assert_allclose(ri.Xf, rs.Xf, atol=1e-5)
        np.testing.assert_allclose(ri.Pf, rs.Pf, atol=1e-5)

    def test_info_update2(self):
        rng = np.random.default_rng(0)
        CtRinvC = np.eye(2) * 2.0
        CtRinvY = rng.standard_normal(2)
        old_i = rng.standard_normal(2)
        old_I = np.eye(2)
        new_i, new_I = info_update2(CtRinvC, CtRinvY, old_i, old_I)
        np.testing.assert_allclose(new_i, old_i + CtRinvY)
        np.testing.assert_allclose(new_I, old_I + CtRinvC)

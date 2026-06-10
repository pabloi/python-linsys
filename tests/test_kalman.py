import numpy as np
import pytest

from linsys.kalman import (filter_stationary, smoother_stationary,
                           info_filter_stationary, info_smoother_stationary,
                           KalmanOpts)
from common import make_system, simulate, naive_kalman_filter, naive_rts_smoother


@pytest.fixture(scope="module")
def sys_and_data():
    A, B, C, D, Q, R = make_system(nx=2, ny=3, nu=1, seed=0)
    Y, X, U = simulate(A, B, C, D, Q, R, N=300, seed=1)
    return A, B, C, D, Q, R, Y, X, U


class TestFilterVsNaive:
    def test_finite_prior_matches_textbook(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0 = np.zeros(2)
        P0 = np.eye(2)
        res = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        Xn, Pn, Xpn, Ppn, logLn = naive_kalman_filter(Y, A, C, Q, R, x0, P0,
                                                      B, D, U)
        np.testing.assert_allclose(res.X, Xn, atol=1e-8)
        np.testing.assert_allclose(res.P, Pn, atol=1e-8)
        np.testing.assert_allclose(res.Xp, Xpn, atol=1e-8)
        np.testing.assert_allclose(res.logL, logLn, rtol=1e-8)

    def test_no_reduce_same_as_reduced(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        r1 = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        r2 = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U,
                               KalmanOpts(no_reduce_flag=True))
        np.testing.assert_allclose(r1.X, r2.X, atol=1e-7)
        np.testing.assert_allclose(r1.logL, r2.logL, rtol=1e-6)

    def test_improper_prior_runs_and_tracks(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        # default x0=0, P0=inf (improper prior)
        res = filter_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
        # after burn-in, filtered states track true states
        err = res.X[:, 50:] - X[:, 50:-1]
        assert np.sqrt(np.mean(err ** 2)) < 0.2
        assert np.isfinite(res.logL)

    def test_improper_prior_converges_to_finite_prior(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        res_inf = filter_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
        res_fin = filter_stationary(Y, A, C, Q, R, np.zeros(2),
                                    1e6 * np.eye(2), B, D, U)
        np.testing.assert_allclose(res_inf.X[:, 50:], res_fin.X[:, 50:],
                                   atol=1e-4)

    def test_fast_mode_matches_exact(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        r_exact = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        r_fast = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U,
                                   KalmanOpts(fast_flag=1))
        np.testing.assert_allclose(r_fast.X, r_exact.X, atol=1e-5)
        np.testing.assert_allclose(r_fast.logL, r_exact.logL, rtol=1e-5)

    def test_nan_samples_skipped(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yn = Y.copy()
        Yn[:, 40:45] = np.nan
        x0, P0 = np.zeros(2), np.eye(2)
        res = filter_stationary(Yn, A, C, Q, R, x0, P0, B, D, U)
        ref = naive_kalman_filter(Yn, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(res.X, ref[0], atol=1e-8)
        assert np.isfinite(res.logL)

    def test_outlier_rejection(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yo = Y.copy()
        Yo[:, 100] += 100.0  # gross outlier
        x0, P0 = np.zeros(2), np.eye(2)
        res = filter_stationary(Yo, A, C, Q, R, x0, P0, B, D, U,
                                KalmanOpts(outlier_flag=True,
                                           no_reduce_flag=True))
        assert res.rejected[100]


class TestSmoother:
    def test_matches_naive_rts(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        res = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        Xs, Ps, Pt, _, _, logL = naive_rts_smoother(Y, A, C, Q, R, x0, P0, B,
                                                    D, U)
        np.testing.assert_allclose(res.Xs, Xs, atol=1e-7)
        np.testing.assert_allclose(res.Ps, Ps, atol=1e-7)
        np.testing.assert_allclose(res.Pt, Pt, atol=1e-7)
        np.testing.assert_allclose(res.logL, logL, rtol=1e-8)

    def test_last_sample_equals_filter(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        res = smoother_stationary(Y, A, C, Q, R, np.zeros(2), np.eye(2), B, D, U)
        np.testing.assert_allclose(res.Xs[:, -1], res.Xf[:, -1])

    def test_smoother_beats_filter(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        res = smoother_stationary(Y, A, C, Q, R, np.zeros(2), np.eye(2), B, D, U)
        ef = np.mean((res.Xf - X[:, :-1]) ** 2)
        es = np.mean((res.Xs - X[:, :-1]) ** 2)
        assert es <= ef * 1.05

    def test_improper_prior(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        res = smoother_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
        err = res.Xs[:, 10:] - X[:, 10:-1]
        assert np.sqrt(np.mean(err ** 2)) < 0.2

    def test_fast_mode_close_to_exact(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        r_exact = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        r_fast = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U,
                                     KalmanOpts(fast_flag=1))
        np.testing.assert_allclose(r_fast.Xs, r_exact.Xs, atol=1e-4)

    def test_nan_handling(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yn = Y.copy()
        Yn[:, 150:160] = np.nan
        res = smoother_stationary(Yn, A, C, Q, R, np.zeros(2), np.eye(2),
                                  B, D, U)
        assert np.all(np.isfinite(res.Xs))


class TestInfoFilter:
    def test_matches_covariance_filter(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        rk = filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        ri = info_filter_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(ri.X, rk.X, atol=1e-6)
        np.testing.assert_allclose(ri.P, rk.P, atol=1e-6)

    def test_info_smoother_matches_rts(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0, P0 = np.zeros(2), np.eye(2)
        rs = smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        ri = info_smoother_stationary(Y, A, C, Q, R, x0, P0, B, D, U)
        np.testing.assert_allclose(ri.Xs, rs.Xs, atol=1e-5)
        np.testing.assert_allclose(ri.Ps, rs.Ps, atol=1e-5)

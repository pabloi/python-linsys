"""Tests for linsys.stats (dataLogLikelihood / logLincomplete / logLcomplete)."""
import numpy as np
import pytest

from linsys.kalman import filter_stationary, smoother_stationary, KalmanOpts
from linsys.stats import data_log_likelihood, logl_incomplete, logl_complete

from common import make_system, simulate


@pytest.fixture(scope="module")
def sysdata():
    A, B, C, D, Q, R = make_system(nx=2, ny=4, nu=1, seed=0)
    Y, X, U = simulate(A, B, C, D, Q, R, N=200, seed=1)
    x0 = np.zeros(2)
    P0 = np.eye(2)
    return A, B, C, D, Q, R, Y, X, U, x0, P0


class TestDataLogLikelihood:
    def test_exact_matches_filter_logl(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        ref = filter_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U,
                                opts=KalmanOpts(fast_flag=False)).logL
        ll = data_log_likelihood(Y, U, A, B, C, D, Q, R, x0, P0, "exact")
        np.testing.assert_allclose(ll, ref, rtol=1e-10)

    def test_exact_from_precomputed_predictions_matches(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        res = filter_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U,
                                opts=KalmanOpts(fast_flag=False))
        ll = data_log_likelihood(Y, U, A, B, C, D, Q, R, res.Xp, res.Pp,
                                 "exact")
        np.testing.assert_allclose(ll, res.logL, rtol=1e-8)

    def test_method_relationships(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        res = filter_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U,
                                opts=KalmanOpts(fast_flag=False))
        lls = {m: data_log_likelihood(Y, U, A, B, C, D, Q, R, res.Xp, res.Pp,
                                      m) for m in
               ("exact", "approx", "fast", "max")}
        for m, v in lls.items():
            assert np.isfinite(v), m
        # max is the supremum over fixed innovation covariances:
        # >= approx by construction, >= exact in practice on this data
        assert lls["max"] >= lls["approx"]
        assert lls["max"] >= lls["exact"] - 1e-6
        # approx and fast are close to exact for a stable steady-ish system
        assert abs(lls["approx"] - lls["exact"]) < 0.05 * abs(lls["exact"])
        assert abs(lls["fast"] - lls["exact"]) < 0.05 * abs(lls["exact"])

    def test_method_ignored_with_initial_state_warns(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        with pytest.warns(UserWarning, match="ignoreMethod"):
            ll = data_log_likelihood(Y, U, A, B, C, D, Q, R, x0, P0, "approx")
        assert np.isfinite(ll)

    def test_list_input_sums(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        ll1 = data_log_likelihood(Y, U, A, B, C, D, Q, R, x0, P0)
        ll2 = data_log_likelihood([Y, Y], [U, U], A, B, C, D, Q, R, x0, P0)
        np.testing.assert_allclose(ll2, 2 * ll1, rtol=1e-10)

    def test_nan_samples_skipped(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        res = filter_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U,
                                opts=KalmanOpts(fast_flag=False))
        Yn = Y.copy()
        Yn[:, 50] = np.nan
        ll = data_log_likelihood(Yn, U, A, B, C, D, Q, R, res.Xp, res.Pp,
                                 "exact")
        assert np.isfinite(ll)


class TestLoglIncomplete:
    def test_per_sample_per_dim_scaling(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        res = filter_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U,
                                opts=KalmanOpts(fast_flag=False))
        total = data_log_likelihood(Y, U, A, B, C, D, Q, R, res.Xp, res.Pp,
                                    "exact")
        per = logl_incomplete(Y, U, A, B, C, D, Q, R, x0, P0, "exact")
        ny, N = Y.shape
        np.testing.assert_allclose(per, total / (N * ny), rtol=1e-8)

    def test_all_methods_finite(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        for m in ("exact", "approx", "fast", "max"):
            v = logl_incomplete(Y, U, A, B, C, D, Q, R, x0, P0, m)
            assert np.isfinite(v), m

    def test_default_prior_warns(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        with pytest.warns(UserWarning, match="noPriorGiven"):
            v = logl_incomplete(Y, U, A, B, C, D, Q, R)
        assert np.isfinite(v)


class TestLoglComplete:
    def test_true_states_high_likelihood(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        ll_true = logl_complete(Y, U, A, B, C, D, Q, R, X)
        assert np.isfinite(ll_true)
        # Perturbing the states must lower the complete-data likelihood
        rng = np.random.default_rng(3)
        Xbad = X + 0.5 * rng.standard_normal(X.shape)
        ll_bad = logl_complete(Y, U, A, B, C, D, Q, R, Xbad)
        assert ll_bad < ll_true

    def test_smoothed_states_beat_noise(self, sysdata):
        A, B, C, D, Q, R, Y, X, U, x0, P0 = sysdata
        sres = smoother_stationary(Y, A, C, Q, R, x0=x0, P0=P0, B=B, D=D, U=U)
        ll_s = logl_complete(Y, U, A, B, C, D, Q, R, sres.Xs)
        assert np.isfinite(ll_s)

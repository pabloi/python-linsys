"""Tests for linsys.em (EM system identification)."""
import numpy as np
import pytest

from linsys.em import (em, em_q0, cv_em, estimate_params, init_em, EMOpts,
                       random_start_em)
from linsys.kalman import smoother_stationary
from linsys.utils import fwd_sim
from common import make_system, simulate


@pytest.fixture(scope="module")
def sys_and_data():
    A, B, C, D, Q, R = make_system(nx=2, ny=3, nu=1, seed=0)
    Y, X, U = simulate(A, B, C, D, Q, R, N=400, seed=1)  # step input
    return A, B, C, D, Q, R, Y, X, U


def _true_logl(A, B, C, D, Q, R, Y, U):
    return smoother_stationary(Y, A, C, Q, R, np.zeros(A.shape[0]), Q,
                               B, D, U).logL


class TestParameterRecovery:
    def test_recovers_io_behavior_and_logl(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        N = Y.shape[1]
        opts = EMOpts(Niter=250)
        res = em(Y, U, 2, opts, rng=0)

        # logL of recovered model close to (or above) the true-model logL:
        l_true = _true_logl(A, B, C, D, Q, R, Y, U)
        assert (res.logL - l_true) / N > -0.05

        # Smoothed-state output fit: residuals at the observation-noise level
        yfit = res.C @ res.X + res.D @ U
        rmse = np.sqrt(np.mean((Y - yfit) ** 2))
        assert rmse < 3 * np.sqrt(np.mean(np.diag(R)))

        # EM models are only identified up to similarity transforms: compare
        # invariants. The dominant mode (0.95) is well excited by the step
        # input and must be recovered (the fast 0.7 mode is only weakly
        # identifiable from step data, so it is not checked):
        ev = np.sort(np.abs(np.linalg.eigvals(res.A)))
        assert abs(ev[-1] - 0.95) < 0.03

        # Deterministic I/O response (Markov-parameter-like invariant):
        y_true, _ = fwd_sim(U, A, B, C, D)
        y_hat, _ = fwd_sim(U, res.A, res.B, res.C, res.D)
        # compare after a short transient, relative to the output scale
        scale = np.sqrt(np.mean(y_true ** 2))
        err = np.sqrt(np.mean((y_true[:, 5:] - y_hat[:, 5:]) ** 2))
        assert err / scale < 0.15


class TestLogLMonotone:
    def test_logl_nondecreasing_exact_em(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Ys, Us = Y[:, :250], U[:, :250]
        opts = EMOpts(Niter=40, fast_flag=0)  # exact E-step: EM guarantee
        res = em(Ys, Us, 2, opts, rng=0)
        hist = res.logl[~np.isnan(res.logl)]
        assert hist.size > 10
        diffs = np.diff(hist)
        # allow only tiny numerical dips
        assert diffs.min() > -1e-6 * max(1.0, abs(hist[-1]))
        assert res.logL >= hist[0]


class TestFixedParams:
    def test_fix_a_is_honored(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        opts = EMOpts(Niter=40, fix_a=A)
        res = em(Y, U, 2, opts, rng=0)
        np.testing.assert_array_equal(res.A, A)

    def test_fix_r_and_fix_q_are_honored(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        opts = EMOpts(Niter=40, fix_r=R, fix_q=Q)
        res = em(Y, U, 2, opts, rng=0)
        np.testing.assert_allclose(res.R, R)
        np.testing.assert_allclose(res.Q, Q)

    def test_fix_x0_p0(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        x0 = np.zeros(2)
        P0 = np.eye(2)
        opts = EMOpts(Niter=30, fix_x0=x0, fix_p0=P0)
        res = em(Y, U, 2, opts, rng=0)
        np.testing.assert_array_equal(res.x0, x0)
        np.testing.assert_array_equal(res.P0, P0)


class TestMissingData:
    def test_nan_block_runs_and_fits(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Yn = Y.copy()
        Yn[:, 120:160] = np.nan  # missing block
        opts = EMOpts(Niter=120)  # fast mode with NaNs: approximate E-step
        with pytest.warns(UserWarning, match="fastAndLoose"):
            res = em(Yn, U, 2, opts, rng=0)
        assert np.isfinite(res.logL)
        for m in (res.A, res.B, res.C, res.D, res.Q, res.R):
            assert np.all(np.isfinite(m))
        idx = ~np.isnan(Yn).any(axis=0)
        yfit = res.C @ res.X + res.D @ U
        rmse = np.sqrt(np.mean((Yn[:, idx] - yfit[:, idx]) ** 2))
        assert rmse < 3 * np.sqrt(np.mean(np.diag(R)))


class TestMultipleRealizations:
    def test_two_trials_run_and_sum_logl(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Y2, _, U2 = simulate(A, B, C, D, Q, R, N=300, seed=7)
        Ys = [Y, Y2]
        Us = [U, U2]
        opts = EMOpts(Niter=80)
        res = em(Ys, Us, 2, opts, rng=0)
        assert isinstance(res.X, list) and len(res.X) == 2
        assert res.X[0].shape == (2, 400) and res.X[1].shape == (2, 300)
        assert np.isfinite(res.logL)
        # returned logL is the sum over realizations of the exact smoother
        # logL under the returned model:
        parts = [smoother_stationary(y, res.A, res.C, res.Q, res.R, x0i, p0i,
                                     res.B, res.D, u).logL
                 for y, u, x0i, p0i in zip(Ys, Us, res.x0, res.P0)]
        np.testing.assert_allclose(res.logL, sum(parts), rtol=1e-10)


class TestRandomStart:
    def test_returns_best_of_runs(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        Ys, Us = Y[:, :250], U[:, :250]
        opts = EMOpts(Niter=150, Nreps=2, refine_max_iter=25)
        res = random_start_em(Ys, Us, 2, opts, rng=0)
        assert np.isfinite(res.logL)
        # must be at least as good as a single short default-init run
        # (random_start_em's rep 0 is exactly that):
        base = em(Ys, Us, 2, EMOpts(Niter=100), rng=0)
        assert res.logL >= base.logL - 1.0
        # and it still fits the data well:
        yfit = res.C @ res.X + res.D @ Us
        rmse = np.sqrt(np.mean((Ys - yfit) ** 2))
        assert rmse < 3 * np.sqrt(np.mean(np.diag(R)))


class TestEMQ0:
    def test_deterministic_states(self, sys_and_data):
        A, B, C, D, _, R, _, _, _ = sys_and_data
        # data generated with NO process noise:
        U = np.ones((1, 300))
        Y, _ = fwd_sim(U, A, B, C, D, Q=None, R=R, rng=3)
        res = em_q0(Y, U, 2)
        np.testing.assert_array_equal(res.Q, np.zeros((2, 2)))
        assert np.isfinite(res.logL)
        yfit = res.C @ res.X + res.D @ U
        rmse = np.sqrt(np.mean((Y - yfit) ** 2))
        assert rmse < 3 * np.sqrt(np.mean(np.diag(R)))


class TestCV:
    def test_cv_em_shapes(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        opts = EMOpts(Niter=60, Nreps=1, disable_refine=True)
        models = cv_em(Y[:, :240], U[:, :240], 2, nfolds=2, opts=opts, rng=0)
        assert len(models) == 2
        for m in models:
            assert m["A"].shape == (2, 2)
            assert m["C"].shape == (3, 2)
            assert m["X"].shape == (2, 240)
            assert np.isfinite(m["logL"])


class TestEstimateParamsDirect:
    def test_perfect_states_recover_params(self):
        """With exact states and zero state uncertainty, the M-step reduces
        to least squares and must recover A, C, D accurately. A random
        (persistently exciting) input is used so the regression is
        well-conditioned (a step input leaves states ~constant, collinear
        with the input)."""
        A, B, C, D, Q, R = make_system(nx=2, ny=3, nu=1, seed=0)
        Y, X, U = simulate(A, B, C, D, Q, R, N=400, seed=2, step_input=False)
        N = Y.shape[1]
        Xs = X[:, :-1]  # states aligned with samples
        P = np.zeros((N, 2, 2))
        Pt = np.zeros((N - 1, 2, 2))
        est = estimate_params(Y, U, Xs, P, Pt)
        np.testing.assert_allclose(est.C, C, atol=0.05)
        np.testing.assert_allclose(est.D, D, atol=0.05)
        np.testing.assert_allclose(est.A, A, atol=0.05)
        np.testing.assert_allclose(est.R, R, atol=5e-3)
        np.testing.assert_allclose(est.Q, Q, atol=5e-4)

    def test_init_em_returns_consistent_shapes(self, sys_and_data):
        A, B, C, D, Q, R, Y, X, U = sys_and_data
        est = init_em(Y, U, 2, rng=0)
        assert est.A.shape == (2, 2)
        assert est.B.shape == (2, 1)
        assert est.C.shape == (3, 2)
        assert est.D.shape == (3, 1)
        assert est.x0.shape == (2,)
        assert est.P0.shape == (2, 2)

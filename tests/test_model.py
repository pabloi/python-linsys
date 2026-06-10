"""Tests for the object-oriented model classes (linsys.model)."""
import warnings

import numpy as np
import pytest

from linsys import kalman, stats
from linsys.em import EMOpts
from linsys.model import (DataFit, DataSet, FittedLinSys, InitCond, LinSys,
                          StateEstimate, TrainInfo)

from common import make_system


def make_model(**kw):
    A, B, C, D, Q, R = make_system(**kw)
    return LinSys(A, B, C, D, Q, R, name="true")


def make_dataset(model, N=120, seed=3, rng=11):
    rng_u = np.random.default_rng(seed)
    U = rng_u.standard_normal((model.ninputs, N))
    ds, st = model.simulate(U, rng=rng)
    return ds, st, U


# ---------------------------------------------------------------------------
# LinSys construction and properties
# ---------------------------------------------------------------------------
class TestLinSysBasics:
    def test_properties(self):
        m = make_model(nx=2, ny=3, nu=1)
        assert m.order == 2
        assert m.nx == 2
        assert m.ninputs == 1
        assert m.noutputs == 3
        np.testing.assert_allclose(np.sort(m.eigenvalues), [0.7, 0.95])
        np.testing.assert_allclose(np.sort(m.time_constants),
                                   np.sort(-1.0 / np.log([0.95, 0.7])))
        assert isinstance(m.hash, str) and len(m.hash) == 32

    def test_constructor_validation(self):
        A, B, C, D, Q, R = make_system()
        with pytest.raises(ValueError):
            LinSys(np.zeros((2, 3)), B, C, D, Q, R)  # A not square
        with pytest.raises(ValueError):
            LinSys(A, np.zeros((3, 1)), C, D, Q, R)  # B rows != nx
        with pytest.raises(ValueError):
            LinSys(A, B, C, np.zeros((2, 1)), Q, R)  # C/D rows mismatch
        with pytest.raises(ValueError):
            LinSys(A, B, C, D, np.eye(3), R)         # Q size != nx
        with pytest.raises(ValueError):
            LinSys(A, B, C, np.zeros((3, 2)), Q, R)  # B/D cols mismatch

    def test_to_from_dict(self):
        m = make_model()
        d = m.to_dict()
        assert set("ABCDQR").issubset(d) and "J" in d
        m2 = LinSys.from_dict(d)
        assert m2.hash == m.hash
        m3 = LinSys.from_dict({"J": m.A, "B": m.B, "C": m.C, "D": m.D,
                               "Q": m.Q, "R": m.R})
        assert m3.hash == m.hash

    def test_variants(self):
        m = make_model()
        assert np.all(m.deterministic().Q == 0)
        np.testing.assert_array_equal(m.deterministic().R, m.R)
        assert np.all(m.noiseless().R == 0)

    def test_noise_covar_det_predict_snr(self):
        m = make_model()
        # noise_covar(N) = sum_{k<N} A^k Q A^k'
        N = 4
        expect = sum(np.linalg.matrix_power(m.A, k) @ m.Q
                     @ np.linalg.matrix_power(m.A, k).T for k in range(N))
        np.testing.assert_allclose(m.noise_covar(N), expect)
        # det_predict: deterministic step response after N steps / at infinity
        I = np.eye(m.order)
        xN = np.zeros(m.order)
        for _ in range(N):
            xN = m.A @ xN + m.B[:, 0]
        np.testing.assert_allclose(m.det_predict(N), xN)
        np.testing.assert_allclose(m.det_predict(),
                                   np.linalg.solve(I - m.A, m.B[:, 0]))
        snr = m.snr(N)
        np.testing.assert_allclose(snr, xN ** 2 / np.diag(expect))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class TestSimulate:
    def test_simulate_returns_dataset_and_states(self):
        m = make_model(nx=2, ny=3, nu=1)
        ds, st, U = make_dataset(m, N=50)
        assert isinstance(ds, DataSet)
        assert ds.nsamp == 50 and ds.noutputs == 3 and ds.ninputs == 1
        assert isinstance(st, StateEstimate)
        assert st.state.shape == (2, 51)  # one extra (final) state

    def test_simulate_deterministic_noiseless(self):
        m = make_model()
        U = np.ones((1, 30))
        ds, st = m.simulate(U, None, True, True)
        # exact deterministic recursion
        x = np.zeros(m.order)
        for k in range(30):
            np.testing.assert_allclose(ds.out[:, k], m.C @ x + m.D @ U[:, k])
            x = m.A @ x + m.B @ U[:, k]
        np.testing.assert_allclose(st.state[:, -1], x)

    def test_simulate_reproducible(self):
        m = make_model()
        U = np.ones((1, 20))
        ds1, _ = m.simulate(U, rng=5)
        ds2, _ = m.simulate(U, rng=5)
        np.testing.assert_array_equal(ds1.out, ds2.out)

    def test_simulate_from_init_cond(self):
        m = make_model()
        x0 = np.array([1.0, -2.0])
        ds, st = m.simulate(np.zeros((1, 10)), InitCond(x0, np.eye(2)),
                            True, True)
        np.testing.assert_allclose(st.state[:, 0], x0)
        np.testing.assert_allclose(ds.out[:, 0], m.C @ x0)


# ---------------------------------------------------------------------------
# Kalman wrappers
# ---------------------------------------------------------------------------
class TestKalmanWrappers:
    def setup_method(self):
        self.m = make_model(nx=2, ny=3, nu=1)
        self.ds, _, _ = make_dataset(self.m, N=80)
        self.x0 = np.zeros(2)
        self.P0 = np.eye(2)

    def test_kfilter_matches_direct(self):
        res = self.m.kfilter(self.ds, InitCond(self.x0, self.P0))
        ref = kalman.filter_stationary(self.ds.out, self.m.A, self.m.C,
                                       self.m.Q, self.m.R, x0=self.x0,
                                       P0=self.P0, B=self.m.B, D=self.m.D,
                                       U=self.ds.in_)
        np.testing.assert_allclose(res.filtered.state, ref.X)
        np.testing.assert_allclose(res.filtered.covar, ref.P)
        np.testing.assert_allclose(res.one_ahead.state, ref.Xp)
        np.testing.assert_allclose(res.one_ahead.covar, ref.Pp)
        assert res.logL == pytest.approx(ref.logL)

    def test_kfilter_default_improper_prior(self):
        res = self.m.kfilter(self.ds)
        ref = kalman.filter_stationary(self.ds.out, self.m.A, self.m.C,
                                       self.m.Q, self.m.R, B=self.m.B,
                                       D=self.m.D, U=self.ds.in_)
        np.testing.assert_allclose(res.filtered.state, ref.X)
        assert res.logL == pytest.approx(ref.logL)

    def test_ksmooth_matches_direct(self):
        res = self.m.ksmooth(self.ds, InitCond(self.x0, self.P0))
        ref = kalman.smoother_stationary(self.ds.out, self.m.A, self.m.C,
                                         self.m.Q, self.m.R, x0=self.x0,
                                         P0=self.P0, B=self.m.B, D=self.m.D,
                                         U=self.ds.in_)
        np.testing.assert_allclose(res.smoothed.state, ref.Xs)
        np.testing.assert_allclose(res.smoothed.covar, ref.Ps)
        np.testing.assert_allclose(res.smoothed.lag_one_covar, ref.Pt)
        np.testing.assert_allclose(res.filtered.state, ref.Xf)
        np.testing.assert_allclose(res.one_ahead.state, ref.Xp)
        assert res.logL == pytest.approx(ref.logL)

    def test_logl_matches_stats(self):
        ic = InitCond(self.x0, self.P0)
        l1 = self.m.logL(self.ds, ic)
        l2 = stats.data_log_likelihood(self.ds.out, self.ds.in_, self.m.A,
                                       self.m.B, self.m.C, self.m.D, self.m.Q,
                                       self.m.R, self.x0, self.P0, "exact")
        assert l1 == pytest.approx(l2)
        # and both equal the filter's logL
        ref = self.m.kfilter(self.ds, ic)
        assert l1 == pytest.approx(ref.logL)

    def test_multiple_dataset_filter(self):
        ds2 = DataSet([self.ds.in_, self.ds.in_], [self.ds.out, self.ds.out])
        res = self.m.kfilter(ds2)
        assert isinstance(res.filtered, list) and len(res.filtered) == 2
        single = self.m.kfilter(self.ds)
        np.testing.assert_allclose(res.filtered[0].state,
                                   single.filtered.state)
        np.testing.assert_allclose(res.logL, [single.logL, single.logL])

    def test_predict_one_step(self):
        x = np.array([[1.0, 0.5], [2.0, -1.0]])  # 2 samples
        P = np.stack([np.eye(2), 2 * np.eye(2)])
        u = np.array([[0.3]])
        pred = self.m.predict(StateEstimate(x, P), u)
        np.testing.assert_allclose(pred.state,
                                   self.m.A @ x + self.m.B @ u)
        np.testing.assert_allclose(
            pred.covar[1], self.m.A @ P[1] @ self.m.A.T + self.m.Q)


# ---------------------------------------------------------------------------
# canonize / transform
# ---------------------------------------------------------------------------
def markov_params(m, n=6):
    return np.stack([m.C @ np.linalg.matrix_power(m.A, k) @ m.B
                     for k in range(n)])


class TestTransforms:
    def test_transform_preserves_io(self):
        m = make_model()
        V = np.array([[2.0, 1.0], [0.5, -1.0]])
        m2 = m.transform(V)
        np.testing.assert_allclose(markov_params(m2), markov_params(m),
                                   atol=1e-12)
        np.testing.assert_array_equal(m2.D, m.D)
        np.testing.assert_array_equal(m2.R, m.R)
        # Q transformed consistently: V Q V'
        np.testing.assert_allclose(m2.Q, V @ m.Q @ V.T)

    def test_canonize_preserves_io(self):
        m = make_model().transform(np.array([[1.0, 2.0], [0.0, 3.0]]))
        for method in (None, "canonicalAlt"):
            mc, V = m.canonize(method)
            np.testing.assert_allclose(markov_params(mc), markov_params(m),
                                       atol=1e-9)
            # canonical forms diagonalize A
            np.testing.assert_allclose(mc.A, np.diag(np.diag(mc.A)),
                                       atol=1e-9)
            # V is the applied transform
            np.testing.assert_allclose(mc.A, V @ m.A @ np.linalg.inv(V),
                                       atol=1e-9)

    def test_scale_and_shift_states(self):
        m = make_model()
        ms = m.scale(2.0)
        np.testing.assert_allclose(markov_params(ms), markov_params(m),
                                   atol=1e-12)
        K = np.array([[0.5], [-0.2]])
        msh = m.shift_states(K)
        # x' = x + K u: under constant input, the shifted model started at
        # x0' = x0 + K u0 produces the same output as the original.
        Uc = np.ones((1, 40))
        da, _ = m.simulate(Uc, None, True, True)
        db, _ = msh.simulate(Uc, InitCond(K @ Uc[:, 0], np.eye(2)), True, True)
        np.testing.assert_allclose(da.out, db.out, atol=1e-10)

    def test_pad_exclude_reduce(self):
        m = make_model(nx=2, ny=3, nu=1)
        p = m.pad([0])  # new output row 0, infinite variance
        assert p.noutputs == 4
        np.testing.assert_array_equal(p.C[0], 0)
        assert np.isinf(p.R[0, 0])
        np.testing.assert_allclose(p.C[1:], m.C)
        np.testing.assert_allclose(p.R[1:, 1:], m.R)
        back = p.exclude_output([0])
        np.testing.assert_allclose(back.C, m.C)
        np.testing.assert_allclose(back.R, m.R)
        # reduce drops inf-variance outputs and gives an nx-output model
        ds, _, _ = make_dataset(m, N=30)
        pds = DataSet(ds.in_, np.vstack([np.zeros((1, 30)), ds.out]))
        red, red_ds = p.reduce(pds)
        assert red.noutputs == m.order
        assert red_ds.noutputs == m.order
        # reduced model preserves the filter logL up to the margin: check
        # states instead (filtering equivalence)
        f1 = m.kfilter(ds, InitCond(np.zeros(2), np.eye(2)))
        f2 = red.kfilter(red_ds, InitCond(np.zeros(2), np.eye(2)))
        np.testing.assert_allclose(f1.filtered.state, f2.filtered.state,
                                   atol=1e-8)


# ---------------------------------------------------------------------------
# StateEstimate / InitCond
# ---------------------------------------------------------------------------
class TestStateEstimate:
    def test_construct_and_props(self):
        x = np.zeros((2, 10))
        P = np.stack([np.eye(2)] * 10)
        st = StateEstimate(x, P)
        assert st.nsamp == 10 and st.order == 2 and not st.is_multiple
        st2 = StateEstimate(x, np.eye(2))  # broadcast single covar
        assert st2.covar.shape == (10, 2, 2)

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            StateEstimate(np.zeros((2, 10)), np.stack([np.eye(3)] * 10))
        with pytest.raises(ValueError):
            StateEstimate(np.zeros((2, 10)), np.stack([np.eye(2)] * 9))
        with pytest.raises(ValueError):  # lag-one covar must have N-1 samples
            StateEstimate(np.zeros((2, 10)), np.stack([np.eye(2)] * 10),
                          np.stack([np.eye(2)] * 5))

    def test_non_psd_warns(self):
        with pytest.warns(UserWarning, match="not PSD"):
            StateEstimate(np.zeros((2, 3)), np.stack([-np.eye(2)] * 3))

    def test_get_sample_and_marginalize(self):
        x = np.arange(20, dtype=float).reshape(2, 10)
        P = np.stack([np.diag([1.0, 2.0])] * 10)
        st = StateEstimate(x, P)
        ic = st.get_sample(3)
        assert isinstance(ic, InitCond)
        np.testing.assert_array_equal(ic.state, x[:, 3])
        np.testing.assert_array_equal(ic.covar, P[3])
        mg = st.marginalize(1)
        assert mg.order == 1 and mg.nsamp == 10
        np.testing.assert_array_equal(mg.state[0], x[1])
        np.testing.assert_array_equal(mg.covar[:, 0, 0], P[:, 1, 1])

    def test_multiple(self):
        st = StateEstimate([np.zeros((2, 5)), np.ones((2, 7))],
                           [np.stack([np.eye(2)] * 5),
                            np.stack([np.eye(2)] * 7)])
        assert st.is_multiple
        np.testing.assert_array_equal(st.nsamp, [5, 7])
        s1 = st.extract_single(1)
        assert s1.nsamp == 7
        with pytest.raises(IndexError):
            st.extract_single(2)

    def test_init_cond_defaults(self):
        ic = InitCond()
        assert ic.state is None and ic.covar is None
        ic2 = InitCond(np.array([1.0, 2.0]))
        assert np.all(np.isinf(np.diag(ic2.covar)))
        assert ic2.order == 2 and ic2.nsamp == 1
        with pytest.raises(ValueError):
            InitCond(np.zeros((2, 3)))
        icm = InitCond([np.zeros(2), np.ones(2)],
                       [np.eye(2), np.eye(2)])
        assert icm.is_multiple
        np.testing.assert_array_equal(icm.extract_single(1).state, [1, 1])


# ---------------------------------------------------------------------------
# DataSet
# ---------------------------------------------------------------------------
class TestDataSet:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.U = rng.standard_normal((1, 20))
        self.Y = rng.standard_normal((3, 20))
        self.ds = DataSet(self.U, self.Y)

    def test_props(self):
        assert self.ds.nsamp == 20
        assert self.ds.ninputs == 1 and self.ds.noutputs == 3
        assert self.ds.non_nan_samp == 20
        Y = self.Y.copy()
        Y[:, 5] = np.nan
        Y[0, 7] = np.nan
        ds = DataSet(self.U, Y)
        assert ds.non_nan_samp == 18
        assert isinstance(self.ds.hash, str)
        assert self.ds.hash != ds.hash

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            DataSet(np.zeros((1, 10)), np.zeros((3, 11)))
        with pytest.raises(ValueError):
            DataSet([np.zeros((1, 10))], [np.zeros((3, 10)),
                                          np.zeros((3, 10))])

    def test_multiple_replication(self):
        ds = DataSet(self.U, [self.Y, self.Y + 1])
        assert ds.is_multiple and len(ds.in_) == 2
        np.testing.assert_array_equal(ds.extract_single(1).out, self.Y + 1)
        with pytest.raises(ValueError):
            self.ds.extract_single(0)

    def test_split(self):
        parts = self.ds.split([5, 12])
        assert [p.nsamp for p in parts] == [5, 7, 8]
        np.testing.assert_array_equal(np.hstack([p.out for p in parts]),
                                      self.Y)
        np.testing.assert_array_equal(np.hstack([p.in_ for p in parts]),
                                      self.U)
        multi = self.ds.split([5, 12], return_as_multi_set=True)
        assert multi.is_multiple and len(multi.out) == 3

    def test_block_split_partitions(self):
        parts = self.ds.block_split(3, 2)
        assert len(parts) == 2
        # Every sample of the original (except the discarded tail of 2)
        # appears in exactly one partition (non-NaN there, NaN elsewhere)
        count = np.zeros(20)
        for p in parts:
            # map partition samples back by matching input columns
            for k in range(p.nsamp):
                if not np.isnan(p.out[:, k]).all():
                    col = np.where((self.U == p.in_[:, k][:, None])
                                   .all(axis=0))[0][0]
                    np.testing.assert_array_equal(p.out[:, k], self.Y[:, col])
                    count[col] += 1
        assert count[:18].sum() == 18 and np.all(count[:18] == 1)
        assert count[18:].sum() == 0  # incomplete tail block discarded

    def test_alternate_folds(self):
        folds = self.ds.alternate(2)
        assert len(folds) == 2
        f0, f1 = folds[0].out, folds[1].out
        assert folds[0].nsamp == 20  # same shape, NaN-masked
        assert not np.isnan(f0[:, 0]).any() and np.isnan(f0[:, 1]).all()
        assert np.isnan(f1[:, 0]).all() and not np.isnan(f1[:, 1]).any()
        # union recovers all data
        merged = np.where(np.isnan(f0), f1, f0)
        np.testing.assert_array_equal(merged, self.Y)

    def test_reduce_and_projections(self):
        m = make_model(nx=2, ny=3, nu=1)
        ds, _, _ = make_dataset(m, N=40)
        red = ds.reduce([1])
        assert red.noutputs == 2
        np.testing.assert_array_equal(red.out, np.delete(ds.out, 1, axis=0))
        res, res_ls = ds.get_data_projections(m)
        assert res.shape == (2, 40) and res_ls.shape == (2, 40)

    def test_logl_delegates(self):
        m = make_model(nx=2, ny=3, nu=1)
        ds, _, _ = make_dataset(m, N=40)
        assert ds.logL(m) == pytest.approx(m.logL(ds))
        both = ds.logL([m, m])
        np.testing.assert_allclose(both, m.logL(ds))

    def test_estimate_var_and_flat_residuals(self):
        m = make_model(nx=2, ny=3, nu=1)
        U = np.ones((1, 60))
        ds, _ = m.simulate(U, rng=2)
        W = ds.estimate_var()
        assert W.shape == (3, 3)
        assert np.all(np.isfinite(W))
        r = ds.flat_residuals()
        assert r.shape == ds.out.shape


# ---------------------------------------------------------------------------
# DataFit
# ---------------------------------------------------------------------------
class TestDataFit:
    def setup_method(self):
        self.m = make_model(nx=2, ny=3, nu=1)
        self.ds, _, _ = make_dataset(self.m, N=60)
        self.ic = InitCond(np.zeros(2), np.eye(2))

    def test_ks_fit(self):
        dfit = self.m.fit(self.ds, self.ic)  # default KS
        assert dfit.fit_method == "KS"
        ref = self.m.ksmooth(self.ds, self.ic)
        np.testing.assert_allclose(dfit.state_estim.state,
                                   ref.smoothed.state)
        assert dfit.logL == pytest.approx(ref.logL)
        np.testing.assert_allclose(dfit.residual,
                                   dfit.output - self.ds.out)
        assert dfit.goodness_of_fit == pytest.approx(ref.logL)
        # KS-only noise estimates
        assert dfit.obs_noise.shape == self.ds.out.shape
        assert dfit.state_noise.shape == (2, 59)
        with pytest.raises(ValueError):
            dfit.n_ahead_output(1)  # no prediction from smoothed states

    def test_kf_fit_one_ahead(self):
        dfit = self.m.fit(self.ds, self.ic, "KF")
        out1 = dfit.one_ahead_output
        assert np.isnan(out1[:, 0]).all()  # first sample unpredictable
        # one-ahead from filtered states: C(A x_k + B u_k) + D u_{k+1}
        xf = dfit.state_estim.state
        k = 10
        expect = (self.m.C @ (self.m.A @ xf[:, k - 1]
                              + self.m.B @ self.ds.in_[:, k - 1])
                  + self.m.D @ self.ds.in_[:, k])
        np.testing.assert_allclose(out1[:, k], expect, atol=1e-10)
        np.testing.assert_allclose(dfit.one_ahead_residual[:, k],
                                   out1[:, k] - self.ds.out[:, k])
        with pytest.raises(ValueError):
            dfit.obs_noise

    def test_deterministic_residual(self):
        dfit = self.m.fit(self.ds, self.ic, "KF")
        sim, _ = self.m.simulate(self.ds.in_, self.ic, True, True)
        np.testing.assert_allclose(dfit.deterministic_residual,
                                   sim.out - self.ds.out)

    def test_residual_method(self):
        r = self.m.residual(self.ds, "det")
        assert r.shape == self.ds.out.shape
        r1 = self.m.residual(self.ds, "oneAhead")
        assert r1.shape == self.ds.out.shape
        assert np.isfinite(r1[:, 1:]).all()


# ---------------------------------------------------------------------------
# Fitting (EM) -> FittedLinSys / TrainInfo
# ---------------------------------------------------------------------------
class TestFit:
    @pytest.fixture(scope="class")
    def fitted(self):
        true = make_model(nx=1, ny=2, nu=1, qscale=1e-3, rscale=1e-2)
        U = np.ones((1, 150))
        ds, _ = true.simulate(U, rng=8)
        opts = EMOpts(Nreps=0, disable_refine=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LinSys.id(ds, 1, opts, rng=0)
        return true, ds, model

    def test_returns_fitted_linsys(self, fitted):
        true, ds, model = fitted
        assert isinstance(model, FittedLinSys)
        assert isinstance(model, LinSys)
        assert model.order == 1
        assert model.name == "rEM 1"
        assert np.isfinite(model.fitted_logL)
        # MLE initial condition stored
        assert model.init_cond_prior.state is not None
        assert model.init_cond_prior.covar.shape == (1, 1)

    def test_beats_wrong_model(self, fitted):
        true, ds, model = fitted
        wrong = LinSys(np.array([[0.2]]), true.B, true.C, np.zeros_like(true.D),
                       true.Q, true.R)
        assert model.fitted_logL > wrong.logL(ds)
        # and is close to (or better than) the true model's likelihood
        assert model.fitted_logL > true.logL(ds) - 20

    def test_train_info(self, fitted):
        true, ds, model = fitted
        ti = model.train_info
        assert isinstance(ti, TrainInfo)
        assert ti.method == "repeatedEM"
        assert ti.set_hash == ds.hash == model.data_set_hash
        assert isinstance(ti.options, EMOpts)
        assert model.data_set is ds
        assert model.data_set_non_nan_samples == 150

    def test_information_criteria(self, fitted):
        true, ds, model = fitted
        assert model.dof() > 0
        assert model.r_dof() == 3  # ny=2 full R: 2*3/2
        assert np.isfinite(model.BIC)
        assert np.isfinite(model.AIC)
        assert model.AICc > model.AIC
        assert model.BIC > model.AIC  # log(150) > 2

    def test_flat_model_and_lrt(self, fitted):
        true, ds, model = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat = LinSys.id(ds, 0)
        assert isinstance(flat, FittedLinSys)
        assert flat.name == "Flat"
        assert flat.order == 1 and np.all(flat.A == 0)
        assert np.isfinite(flat.fitted_logL)
        assert model.fitted_logL > flat.fitted_logL  # dynamics help
        p, chi, ddof = model.likelihood_ratio_test(flat)
        assert chi > 0 and ddof > 0 and 0 <= p <= 1

    def test_fit_linsys_entry_point(self, fitted):
        from linsys import fit_linsys
        true, ds, model = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2 = fit_linsys((ds.out, ds.in_), 0)
        assert isinstance(m2, FittedLinSys)
        assert m2.name == "Flat"

    def test_ss_id(self, fitted):
        true, ds, _ = fitted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LinSys.ss_id(ds, 1, ss_size=5)
        assert isinstance(m, FittedLinSys)
        assert m.order == 1
        assert m.method == "SS_i5"
        assert np.isfinite(m.goodness_of_fit)
        with pytest.raises(ValueError):
            m.dof()  # only defined for EM fits (as in MATLAB)
        with pytest.raises(NotImplementedError):
            LinSys.ss_id(ds, 1, method="subid")


# ---------------------------------------------------------------------------
# em_refine (smoke test)
# ---------------------------------------------------------------------------
class TestEmRefine:
    def test_refine_improves_perturbed_model(self):
        true = make_model(nx=1, ny=2, nu=1)
        U = np.ones((1, 60))
        ds, _ = true.simulate(U, rng=4)
        start = true._with_params(A=np.array([[0.8]]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refined = start.em_refine(ds)
        assert isinstance(refined, LinSys)
        assert refined.logL(ds) >= start.logL(ds)


# ---------------------------------------------------------------------------
# misc: upsample/downsample, summary table, bic_aic helper
# ---------------------------------------------------------------------------
class TestMisc:
    def test_unimplemented(self):
        m = make_model()
        with pytest.raises(NotImplementedError):
            m.upsample(2)
        with pytest.raises(NotImplementedError):
            m.downsample(2)

    def test_summary_table(self):
        m1 = make_model()
        m2, _ = m1.canonize()
        m2.name = "canon"
        tbl = LinSys.summary_table([m1, m2])
        assert tbl["tau"].shape == (2, 2)
        # equivalent models have the same time constants
        np.testing.assert_allclose(np.real(tbl["tau"][:, 0]),
                                   np.real(tbl["tau"][:, 1]), atol=1e-9)
        assert tbl["row_names"] == ["true", "canon"]

    def test_bic_aic_helper(self):
        m = make_model(nx=2, ny=3, nu=1)
        ds, _, _ = make_dataset(m, N=40)
        res = m.bic_aic(ds)
        assert np.isfinite(res.BIC) and np.isfinite(res.AIC)
        assert res.BIC < res.AIC  # higher-is-better convention, log(40) > 2

"""Tests for linsys.hmm (port of matlab-linsys/discrHMM)."""
import numpy as np
import pytest

from linsys.hmm import (column_normalize, hmm_stationary_inference,
                        hmm_stationary_inference_alt, hmm_logl, viterbi,
                        hmm_matrix_estim, hmm_em, discretize_obs)


def make_hmm():
    """3-state HMM with well-separated emissions (4 symbols).

    Column-stochastic convention: T[i, j] = p(next=i | curr=j),
    O[m, d] = p(y=m | x=d).
    """
    T = np.array([[0.90, 0.05, 0.05],
                  [0.05, 0.90, 0.05],
                  [0.05, 0.05, 0.90]])
    O = np.array([[0.85, 0.05, 0.05],
                  [0.05, 0.85, 0.05],
                  [0.05, 0.05, 0.85],
                  [0.05, 0.05, 0.05]])
    p0 = np.array([1.0, 0.0, 0.0])
    return T, O, p0


def simulate_hmm(T, O, p0, N, seed=0):
    rng = np.random.default_rng(seed)
    D = T.shape[0]
    M = O.shape[0]
    x = np.empty(N, dtype=int)
    y = np.empty(N, dtype=int)
    x[0] = rng.choice(D, p=p0)
    for k in range(N):
        if k > 0:
            x[k] = rng.choice(D, p=T[:, x[k - 1]])  # column-stochastic
        y[k] = rng.choice(M, p=O[:, x[k]])
    return x, y


class TestInference:
    def test_posteriors_sum_to_one(self):
        T, O, p0 = make_hmm()
        _, y = simulate_hmm(T, O, p0, 200, seed=1)
        res = hmm_stationary_inference(y, O, T, p0)
        np.testing.assert_allclose(res.p_updated.sum(axis=0), 1, atol=1e-12)
        np.testing.assert_allclose(res.p_predicted.sum(axis=0), 1, atol=1e-12)
        np.testing.assert_allclose(res.p_smoothed.sum(axis=0), 1, atol=1e-12)
        assert (res.p_smoothed >= 0).all()
        assert res.p_updated.shape == (3, 200)
        assert res.p_predicted.shape == (3, 201)

    def test_smoothed_map_accuracy(self):
        T, O, p0 = make_hmm()
        x, y = simulate_hmm(T, O, p0, 500, seed=2)
        res = hmm_stationary_inference(y, O, T, p0)
        acc = np.mean(np.argmax(res.p_smoothed, axis=0) == x)
        assert acc > 0.9

    def test_alt_matches_standard_for_one_obs_per_step(self):
        T, O, p0 = make_hmm()
        _, y = simulate_hmm(T, O, p0, 100, seed=3)
        res1 = hmm_stationary_inference(y, O, T, p0)
        res2 = hmm_stationary_inference_alt(y, np.arange(100), O, T, p0)
        np.testing.assert_allclose(res1.p_smoothed, res2.p_smoothed,
                                   atol=1e-12)


class TestViterbi:
    def test_recovers_states(self):
        T, O, p0 = make_hmm()
        x, y = simulate_hmm(T, O, p0, 500, seed=4)
        seq, logl = viterbi(y, T, O, p0)
        assert np.mean(seq == x) > 0.9
        assert np.isfinite(logl)

    def test_logl_is_joint_of_map_path(self):
        T, O, p0 = make_hmm()
        _, y = simulate_hmm(T, O, p0, 50, seed=5)
        seq, logl = viterbi(y, T, O, p0)
        # Recompute joint logL of the returned path by hand
        ll = np.log(p0[seq[0]]) + np.log(O[y[0], seq[0]])
        for k in range(1, len(y)):
            ll += np.log(T[seq[k], seq[k - 1]]) + np.log(O[y[k], seq[k]])
        np.testing.assert_allclose(logl, ll, atol=1e-10)
        # ... and it must not exceed the total data logL
        assert logl <= hmm_logl(y, O, T, p0) + 1e-10

    def test_uniform_prior_warning(self):
        T, O, _ = make_hmm()
        with pytest.warns(UserWarning):
            viterbi([0, 1, 2], T, O)


class TestFit:
    def test_matrix_estim_near_deterministic_posteriors(self):
        T, O, p0 = make_hmm()
        x, y = simulate_hmm(T, O, p0, 4000, seed=6)
        # Build (almost) deterministic posteriors from the true states
        D = T.shape[0]
        p = np.full((D, x.size), 1e-12)
        p[x, np.arange(x.size)] = 1.0
        p = column_normalize(p)
        That, Ohat = hmm_matrix_estim(p, y, n_symbols=O.shape[0])
        np.testing.assert_allclose(That.sum(axis=0), 1, atol=1e-12)
        np.testing.assert_allclose(Ohat.sum(axis=0), 1, atol=1e-12)
        assert np.abs(That - T).max() < 0.05
        assert np.abs(Ohat - O).max() < 0.05

    def test_em_improves_logl_and_recovers_transitions(self):
        T, O, p0 = make_hmm()
        x, y = simulate_hmm(T, O, p0, 2000, seed=7)
        # Start from a perturbed version of the truth
        rng = np.random.default_rng(8)
        T0 = column_normalize(T + 0.15 * rng.uniform(size=T.shape))
        O0 = column_normalize(O + 0.15 * rng.uniform(size=O.shape))
        res = hmm_em(y, p0, observation_matrix=O0, transition_matrix=T0,
                     max_iter=30)
        assert res.logl[-1] > res.logl[0]
        # logL roughly monotone (M-step is approximate; allow tiny dips)
        assert (np.diff(res.logl) > -1e-6).all()
        assert np.abs(res.transition_matrix - T).max() < 0.1
        assert np.abs(res.observation_matrix - O).max() < 0.1
        np.testing.assert_allclose(res.state_distr.sum(axis=0), 1, atol=1e-9)


class TestHelpers:
    def test_column_normalize(self):
        p = np.array([[1.0, 3.0], [1.0, 1.0]])
        out = column_normalize(p)
        np.testing.assert_allclose(out.sum(axis=0), 1)
        np.testing.assert_allclose(out[:, 1], [0.75, 0.25])

    def test_discretize_obs(self):
        obs = np.array([0.0, 0.5, 1.0])
        bins = discretize_obs(obs, nbins=10)
        assert bins[0] == 0 and bins[-1] == 9
        assert (bins >= 0).all() and (bins <= 9).all()

    def test_hmm_logl_finite_and_negative(self):
        T, O, p0 = make_hmm()
        _, y = simulate_hmm(T, O, p0, 100, seed=9)
        ll = hmm_logl(y, O, T, p0)
        assert np.isfinite(ll) and ll < 0

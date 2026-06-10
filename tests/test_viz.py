"""Smoke tests for linsys.viz (each function runs on synthetic data and
returns a matplotlib Figure)."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from linsys.viz import (sort_c, viz_data_fit, viz_data_likelihood,
                        viz_cv_data_likelihood, viz_data_res, viz_models,
                        viz_hmm_inference)

from common import make_system, simulate


@pytest.fixture(scope="module")
def models_and_data():
    A, B, C, D, Q, R = make_system(nx=2, ny=4, nu=1, seed=0)
    Y, X, U = simulate(A, B, C, D, Q, R, N=120, seed=1)
    m1 = dict(A=A, B=B, C=C, D=D, Q=Q, R=R, name="true")
    # A smaller (1-state) competing model:
    m2 = dict(A=A[:1, :1], B=B[:1, :], C=C[:, :1], D=D, Q=Q[:1, :1], R=R,
              name="small")
    return [m1, m2], Y, U


def _close(fig):
    import matplotlib.pyplot as plt
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_sort_c():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((6, 3))
    # Same columns permuted (and scaled) must map back to their source
    perm = [2, 0, 1]
    other = 2.0 * ref[:, perm]
    out = sort_c(ref, [other, ref])
    np.testing.assert_array_equal(out[0], perm)
    np.testing.assert_array_equal(out[1], [0, 1, 2])
    # Smaller model: indices stay within range
    out2 = sort_c(ref, [ref[:, :2]])
    assert len(out2[0]) == 2 and set(out2[0]) <= {0, 1, 2}


def test_viz_data_fit(models_and_data):
    models, Y, U = models_and_data
    _close(viz_data_fit(models, Y, U))


def test_viz_data_fit_single_model(models_and_data):
    models, Y, U = models_and_data
    _close(viz_data_fit(models[0], Y, U))


def test_viz_data_likelihood(models_and_data):
    models, Y, U = models_and_data
    _close(viz_data_likelihood(models, (Y, U)))
    _close(viz_data_likelihood(models, [(Y, U), (Y, U)]))


@pytest.mark.parametrize("method", ["logL", "oneAheadRMSE", "detRMSE"])
def test_viz_cv_data_likelihood(models_and_data, method):
    models, Y, U = models_and_data
    _close(viz_cv_data_likelihood(models, (Y, U), method=method))


def test_viz_cv_data_likelihood_folds(models_and_data):
    models, Y, U = models_and_data
    per_fold = [[m, m] for m in models]  # models[i][kd]
    _close(viz_cv_data_likelihood(per_fold, [(Y, U), (Y, U)]))


def test_viz_cv_mismatched_sizes_raises(models_and_data):
    models, Y, U = models_and_data
    with pytest.raises(ValueError):
        viz_cv_data_likelihood([[models[0]]], [(Y, U), (Y, U)])


def test_viz_data_res(models_and_data):
    models, Y, U = models_and_data
    _close(viz_data_res(models, Y, U))


def test_viz_models(models_and_data):
    models, _, _ = models_and_data
    _close(viz_models(models))
    _close(viz_models(models[0]))


def test_viz_hmm_inference():
    rng = np.random.default_rng(2)
    T = np.array([[0.9, 0.1], [0.1, 0.9]])
    O = np.array([[0.8, 0.1], [0.1, 0.8], [0.1, 0.1]])
    p_est = rng.uniform(size=(2, 50))
    p_est = p_est / p_est.sum(axis=0)
    obs = rng.integers(0, 3, size=50)
    _close(viz_hmm_inference(p_est, T, O, obs=obs))
    _close(viz_hmm_inference(p_est, T, O))

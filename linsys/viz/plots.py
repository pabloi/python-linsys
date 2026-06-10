"""Model/data visualization (port of matlab-linsys/misc/viz and
discrHMM/vizHMMInference.m).

Since the linsys model classes are not ported yet, models are plain dicts
with keys 'A' (or 'J'), 'B', 'C', 'D', 'Q', 'R' and an optional 'name'.
Data follows PORTING.md: Y is (ny, N), U is (nu, N), columns are time.

All functions import matplotlib lazily (Agg-safe, see
:func:`linsys.viz.helpers.get_plt`) and RETURN the figure object(s); they
never call ``plt.show()``. The visual content follows the MATLAB originals;
MATLAB-specific styling (normalized full-screen windows, color-order
hacking, uistack, ...) is simplified.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..utils import substitute_nans, logl_normal
from ..kalman import smoother_stationary
from .helpers import (get_plt, sort_c, model_mats, model_name,
                      one_ahead_output, deterministic_sim, filter_logl,
                      data_projections, red_blue_cmap)

__all__ = ["viz_data_fit", "viz_data_likelihood", "viz_cv_data_likelihood",
           "viz_data_res", "viz_models", "viz_hmm_inference"]


def _as_model_list(models):
    if isinstance(models, dict):
        return [models]
    return list(models)


def _imvec(ax, v, cmap, vmax):
    """Display a vector as a 1-column image (MATLAB imagesc of a column)."""
    v = np.asarray(v, dtype=float).reshape(-1, 1)
    ax.imshow(v, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])


def viz_data_fit(models, Y, U):
    """Overview of model fits to data (MATLAB: vizDataFit).

    Columns of the figure (as in MATLAB's 4-column layout):
    1. principal components of the data (uncentered PCA) as images,
    2. data projected on those PCs with each model's one-step-ahead (KF)
       output overlaid,
    3. error measures (per-sample RMSE time courses and total-RMSE bars for
       the deterministic and one-ahead predictions),
    4. smoothed states (with +- 1 std band) vs. the data projected onto the
       states, and
    5. deterministic (simulated) states vs. the same projection.

    The MATLAB second figure (output snapshots at hard-coded time points,
    returned only when two outputs are requested) is not ported.

    Returns the matplotlib Figure.
    """
    plt = get_plt()
    models = _as_model_list(models)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    ny, N = Y.shape
    cmap = red_blue_cmap()

    fits = [one_ahead_output(m, Y, U) for m in models]  # (yhat, smoother res)
    orders = [model_mats(m)[0].shape[0] for m in models]
    nx_max = max(orders)
    largest = int(np.argmax(orders))
    matches = sort_c(model_mats(models[largest])[2],
                     [model_mats(m)[2] for m in models])

    n_rows = max(nx_max, 5, 4)
    fig, axs = plt.subplots(n_rows, 5, figsize=(16, 2.2 * n_rows),
                            squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)

    # --- Columns 1-2: PCA of the data and one-ahead fits along the PCs
    Ysub = np.where(np.isnan(Y), 0.0, Y)
    cc, ss, _ = np.linalg.svd(Ysub, full_matrices=False)
    var = ss ** 2
    max_k = min(5, ny, n_rows)
    aC = np.abs(cc[:, :max_k]).max()
    for kk in range(max_k):
        ax = axs[kk, 0]
        ax.set_visible(True)
        _imvec(ax, cc[:, kk], cmap, aC)
        ax.set_ylabel(f"PC {kk + 1}, {100 * var[kk] / var.sum():.0f}%")
        ax = axs[kk, 1]
        ax.set_visible(True)
        ax.scatter(np.arange(N), cc[:, kk] @ Y, s=4, c="0.5")
        for i, (yhat, _) in enumerate(fits):
            ax.plot(cc[:, kk] @ yhat, lw=2, label=model_name(models[i], i))
        if kk == 0:
            ax.set_title("One-ahead (KF) output\nprojected onto data PCs")
            ax.legend(fontsize=6)

    # --- Column 3: error measures
    rr = Y - np.nanmean(Y, axis=1, keepdims=True)  # flat-model residuals
    residual_reference = np.sqrt(np.nansum(rr ** 2))
    labels = [model_name(m, i) for i, m in enumerate(models)]
    for ll, tag in enumerate(["Deterministic output error",
                              "KF prediction error"]):
        ax_tc = axs[2 * ll, 2]
        ax_bar = axs[2 * ll + 1, 2]
        ax_tc.set_visible(True)
        ax_bar.set_visible(True)
        for k, m in enumerate(models):
            if ll == 0:
                x0 = fits[k][1].Xs[:, 0]
                ysim, _ = deterministic_sim(m, U, x0)
                res = Y - ysim
            else:
                res = Y - fits[k][0]
            rmse_tc = np.nansum(res ** 2, axis=0)
            ax_tc.scatter(np.arange(N), np.sqrt(rmse_tc), s=3, alpha=.5)
            total = np.sqrt(np.nansum(res ** 2)) / residual_reference
            ax_bar.bar(k, total, edgecolor="k")
        ax_tc.set_title(f"{tag} (RMSE)", fontsize=8)
        ax_tc.set_yscale("log")
        ax_tc.grid(True)
        ax_bar.set_xticks(range(len(models)))
        ax_bar.set_xticklabels(labels, fontsize=6, rotation=90)
        ax_bar.grid(True)

    # --- Columns 4-5: smoothed and deterministic states vs projections
    for k, m in enumerate(models):
        A = model_mats(m)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            taus = -1.0 / np.log(np.sort(np.linalg.eigvals(A).real))
        projX = data_projections(m, Y, U)
        sres = fits[k][1]
        x0 = sres.Xs[:, 0]
        _, xsim = deterministic_sim(m, U, x0)
        for i in range(min(A.shape[0], n_rows)):
            row = matches[k][i] if matches[k][i] < n_rows else i
            nn = f"{model_name(m, k)}, tau={taus[i]:.3g}"
            for col, xx in ((3, sres.Xs), (4, xsim)):
                ax = axs[row, col]
                ax.set_visible(True)
                ax.scatter(np.arange(projX.shape[1]), projX[i, :], s=3,
                           alpha=.2)
                ax.plot(xx[i, :projX.shape[1]], lw=2, label=nn)
                if col == 3:
                    sd = np.sqrt(np.maximum(sres.Ps[:, i, i], 0))
                    ax.fill_between(np.arange(sres.Xs.shape[1]),
                                    sres.Xs[i, :] - sd, sres.Xs[i, :] + sd,
                                    alpha=.3)
                ax.grid(True)
                ax.set_ylabel(f"State {i + 1}", fontsize=7)
        if k == len(models) - 1:
            axs[0, 3].set_title("(Smoothed) states vs. projection",
                                fontsize=8)
            axs[0, 4].set_title("(Deterministic) states vs. projection",
                                fontsize=8)
            for row in range(n_rows):
                if axs[row, 3].get_visible():
                    axs[row, 3].legend(fontsize=5)

    # --- Smoothed-state one-ahead error z-scores (last row, col 5)
    ax = axs[n_rows - 1, 4]
    if not ax.get_visible():
        ax.set_visible(True)
        for k, m in enumerate(models):
            Q = model_mats(m)[4]
            sres = fits[k][1]
            st_err = sres.Xs - sres.Xp[:, :-1]
            _, z2 = logl_normal(st_err, Q)
            ax.plot(np.sqrt(z2), lw=1, label=model_name(m, k))
        ax.set_title("(KS) predicted state error (z-score)", fontsize=8)
        ax.grid(True)
    fig.tight_layout()
    return fig


def viz_data_likelihood(models, datasets):
    """Bar plot of the relative log-likelihood of each model on each dataset
    (MATLAB: vizDataLikelihood).

    ``datasets`` is a (Y, U) tuple or a list of them; one subplot row per
    dataset. As in MATLAB, each model is first filtered with the default
    improper prior, then re-fit using the resulting initial state (with P0 =
    Q, the same initial condition EM would set), and Delta logL (relative to
    the worst model) is shown.

    Returns the matplotlib Figure.
    """
    plt = get_plt()
    models = _as_model_list(models)
    if isinstance(datasets, tuple) and len(datasets) == 2 and \
            not isinstance(datasets[0], tuple):
        datasets = [datasets]
    fig, axs = plt.subplots(len(datasets), 1, figsize=(8, 3 * len(datasets)),
                            squeeze=False)
    labels = [model_name(m, i) for i, m in enumerate(models)]
    for kd, (Y, U) in enumerate(datasets):
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        U = np.atleast_2d(np.asarray(U, dtype=float))
        logl = []
        for m in models:
            A, B, C, D, Q, R = model_mats(m)
            first = smoother_stationary(Y, A, C, Q, R, B=B, D=D, U=U)
            logl.append(filter_logl(m, Y, U, x0=first.Xs[:, 0], P0=Q))
        yy = np.asarray(logl) - np.min(logl)
        ax = axs[kd, 0]
        for k in range(len(models)):
            ax.bar(k, yy[k])
            ax.text(k, .05 * max(yy[k], 1e-12), f"{yy[k]:.6g}", rotation=90,
                    fontsize=8, ha="center")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_title(r"$\Delta$ logL")
        ax.grid(True)
        if yy.max() > 0:
            ax.set_ylim(0, 1.1 * yy.max())
    fig.tight_layout()
    return fig


def viz_cv_data_likelihood(models, test_sets, method="logL"):
    """Cross-validated goodness-of-fit comparison (MATLAB:
    vizCVDataLikelihood).

    Parameters
    ----------
    models : list of lists, ``models[i][kd]`` = model i trained for fold kd
        (a flat list is accepted for a single test set).
    test_sets : (Y, U) tuple or list of them, one per fold; one subplot row
        per fold.
    method : 'logL' (filter log-likelihood on the held-out data),
        'oneAheadRMSE' (RMSE of the one-step-ahead Kalman prediction) or
        'detRMSE' (RMSE of the deterministic simulation from the smoothed
        initial state).

    Returns the matplotlib Figure.
    """
    plt = get_plt()
    if isinstance(test_sets, tuple) and len(test_sets) == 2 and \
            not isinstance(test_sets[0], tuple):
        test_sets = [test_sets]
    models = _as_model_list(models)
    if models and isinstance(models[0], dict):
        models = [[m] * len(test_sets) for m in models]
    if any(len(row) != len(test_sets) for row in models):
        raise ValueError("Number of models and testSets is not the same")

    fig, axs = plt.subplots(len(test_sets), 1,
                            figsize=(8, 3 * len(test_sets)), squeeze=False)
    for kd, (Y, U) in enumerate(test_sets):
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        U = np.atleast_2d(np.asarray(U, dtype=float))
        vals = []
        for row in models:
            m = row[kd]
            if method == "logL":
                vals.append(filter_logl(m, Y, U))
            elif method == "oneAheadRMSE":
                yhat, _ = one_ahead_output(m, Y, U)
                res = yhat - Y
                vals.append(float(np.sqrt(np.nanmean(np.nansum(res ** 2,
                                                               axis=0)))))
            elif method == "detRMSE":
                _, sres = one_ahead_output(m, Y, U)
                ysim, _ = deterministic_sim(m, U, sres.Xs[:, 0])
                res = ysim - Y
                vals.append(float(np.sqrt(np.nanmean(np.nansum(res ** 2,
                                                               axis=0)))))
            else:
                raise ValueError(f"unknown method '{method}'")
        yy = np.asarray(vals) - np.min(vals)
        ax = axs[kd, 0]
        labels = [model_name(row[kd], i) for i, row in enumerate(models)]
        for k in range(len(models)):
            ax.bar(k, yy[k], alpha=.5, edgecolor="w")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks([])
        ax.set_title(rf"$\Delta$ {method}")
        if yy.max() > 0:
            ax.set_ylim(0, 1.1 * yy.max())
    fig.tight_layout()
    return fig


def viz_data_res(models, Y, U, n_components=6):
    """Diagnostics of the one-step-ahead output residuals
    (MATLAB: vizDataRes).

    The residuals of the best model (highest logL; NOTE: the MATLAB source
    picks ``min(logL)``, which selects the WORST model under its own
    higher-is-better convention - ported as max) define a PCA basis; each
    model's residuals are projected on it and, per component, the figure
    shows: the PC vector (image), the projected residual time course, a
    normal QQ-plot, the autocorrelation (+-15 lags) and a histogram.

    Returns the matplotlib Figure.
    """
    plt = get_plt()
    from scipy import stats as sps
    models = _as_model_list(models)
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    cmap = red_blue_cmap()

    fits = [one_ahead_output(m, Y, U) for m in models]
    logls = [f[1].logL for f in fits]
    best = int(np.argmax(logls))
    res_best = Y - fits[best][0]
    res_best = substitute_nans(res_best.T).T  # interpolate NaNs
    pp, _, _ = np.linalg.svd(res_best, full_matrices=False)

    n_comp = min(n_components, Y.shape[0])
    fig, axs = plt.subplots(n_comp, 5, figsize=(15, 2.2 * n_comp),
                            squeeze=False)
    aP = .5 * np.abs(pp[:, :n_comp]).max()
    for i, m in enumerate(models):
        res = Y - fits[i][0]
        res = substitute_nans(res.T).T
        cc = np.linalg.lstsq(pp, res, rcond=None)[0]  # projections
        for kk in range(n_comp):
            if i == best:
                _imvec(axs[kk, 0], pp[:, kk], cmap, aP)
                axs[kk, 0].set_title(f"Residual PC {kk + 1} (best model)",
                                     fontsize=8)
            ax = axs[kk, 1]
            ax.scatter(np.arange(cc.shape[1]), cc[kk, :], s=3, alpha=.2)
            ax.set_title("PC of residual", fontsize=8)
            ax.grid(True)

            ax = axs[kk, 2]
            osm, osr = sps.probplot(cc[kk, :], dist="norm")[0]
            ax.scatter(osm, osr, s=3, alpha=.4)
            ax.set_title(f"QQ plot residual PC {kk + 1}", fontsize=8)

            ax = axs[kk, 3]
            x = cc[kk, :] - cc[kk, :].mean()
            r = np.correlate(x, x, mode="full")
            lags = np.arange(-(len(x) - 1), len(x))
            keep = np.abs(lags) <= 15
            ax.plot(lags[keep], r[keep])
            ax.set_xlabel("Delay (samp)", fontsize=7)
            ax.set_title("Residual autocorr", fontsize=8)
            ax.grid(True)

            ax = axs[kk, 4]
            ax.hist(cc[kk, :], bins=30, density=True, alpha=.4,
                    edgecolor="none")
            ax.set_title(f"Residual PC {kk + 1} histogram", fontsize=8)
    fig.tight_layout()
    return fig


def viz_models(models):
    """Side-by-side comparison of models, no data needed
    (MATLAB: vizModels).

    Simulates each model's step response (100 samples of 0, then 900 of 1 on
    every input) and shows, with state rows matched across models via
    :func:`linsys.viz.helpers.sort_c`: the state step responses, the poles
    as time constants (log scale), and images of the columns of C, the
    columns of D, R and Q.

    NOTE: the MATLAB source draws Q over R in the same subplot (the R panel
    is overwritten); here Q gets its own row. The SNR-based titles (which
    require the unported model class) are dropped.

    Returns the matplotlib Figure.
    """
    plt = get_plt()
    models = _as_model_list(models)
    cmap = red_blue_cmap()
    orders = [model_mats(m)[0].shape[0] for m in models]
    M = max(orders)
    largest = int(np.argmax(orders))
    matches = sort_c(model_mats(models[largest])[2],
                     [model_mats(m)[2] for m in models])
    Md = max(model_mats(m)[3].shape[1] for m in models)
    n_rows = M + Md + 3  # states, poles, D rows, R, Q
    n_cols = 2 + len(models)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(2.5 * n_cols, 1.8 * n_rows),
                            squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)

    # Step responses and pole plot (first two columns, merged visually)
    sims = []
    for k, m in enumerate(models):
        A, B, C, D, Q, R = model_mats(m)
        nu = B.shape[1]
        Ustep = np.hstack([np.zeros((nu, 100)), np.ones((nu, 900))])
        _, X2 = fwd_sim_states(m, Ustep)
        sims.append(X2)
        for i in range(A.shape[0]):
            row = matches[k][i] if matches[k][i] < M else i
            for col in (0, 1):
                ax = axs[row, col]
                ax.set_visible(True)
            ax = axs[row, 0]
            ax.plot(X2[i, :], lw=2, label=model_name(m, k))
            if k == 0:
                ax.set_ylabel(f"State {row + 1}")
                if row == 0:
                    ax.set_title("Step-response states", fontsize=9)
    for row in range(M):
        if axs[row, 0].get_visible():
            axs[row, 0].legend(fontsize=6)
            axs[row, 1].set_visible(False)

    ax = axs[M, 0]
    ax.set_visible(True)
    for k, m in enumerate(models):
        A = model_mats(m)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            taus = -1.0 / np.log(np.linalg.eigvals(A).astype(complex))
        ax.scatter(k + .2 * np.sign(taus.imag), np.abs(taus.real), zorder=3)
    ax.set_yscale("log")
    ax.set_title("Time constants", fontsize=9)
    ax.grid(True)

    # C, D, R, Q images per model column
    aC = .5 * max(np.abs(model_mats(m)[2]).max() for m in models)
    for i, m in enumerate(models):
        A, B, C, D, Q, R = model_mats(m)
        col = 2 + i
        for k in range(A.shape[0]):
            row = matches[i][k] if matches[i][k] < M else k
            ax = axs[row, col]
            ax.set_visible(True)
            _imvec(ax, C[:, k], cmap, aC)
            if i == 0:
                ax.set_ylabel(f"C_{k + 1}")
            if row == 0:
                ax.set_title(model_name(m, i), fontsize=9)
        for k in range(D.shape[1]):
            ax = axs[M + 1 + k, col]
            ax.set_visible(True)
            _imvec(ax, D[:, k], cmap, aC)
            if i == 0:
                ax.set_ylabel(f"D_{k + 1}")
        aR = np.mean(np.diag(R))
        ax = axs[M + 1 + Md, col]
        ax.set_visible(True)
        ax.imshow(R, cmap=cmap, vmin=-aR, vmax=aR)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("R")
        aQ = max(np.abs(model_mats(mm)[4]).max() for mm in models)
        if aQ == 0:
            aQ = .01 * aR / max(aC, 1e-12) ** 2
        ax = axs[M + 2 + Md, col]
        ax.set_visible(True)
        ax.imshow(Q, cmap=cmap, vmin=-aQ, vmax=aQ)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Q")
    fig.tight_layout()
    return fig


def fwd_sim_states(model, U):
    """Noise-free simulation helper used by viz_models."""
    return deterministic_sim(model, U, None)


def viz_hmm_inference(p_estimate, p_state_given_prev, p_obs_given_state,
                      obs=None, obs_times=None):
    """Visualize discrete-HMM inference (MATLAB: discrHMM/vizHMMInference).

    Four panels: the (column-normalized) transition matrix, the observation
    matrix, the posterior state distribution over time (heatmap with the MAP
    state overlaid) and, if given, a scatter of the observations over time.

    Returns the matplotlib Figure.
    """
    plt = get_plt()
    from ..hmm import column_normalize
    p_estimate = np.atleast_2d(np.asarray(p_estimate, dtype=float))
    T = column_normalize(np.asarray(p_state_given_prev, dtype=float))
    O = np.asarray(p_obs_given_state, dtype=float)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([.1, .69, .35, .25])
    im = ax.imshow(T, origin="lower", aspect="auto", cmap="Blues",
                   vmin=0, vmax=T.max())
    ax.set_title("Transition matrix")
    ax.set_xlabel("Current state")
    ax.set_ylabel("Next state")
    fig.colorbar(im, ax=ax)

    ax = fig.add_axes([.58, .69, .35, .25])
    im = ax.imshow(O, origin="lower", aspect="auto", cmap="Blues",
                   vmin=0, vmax=O.max())
    ax.set_title("Observation matrix")
    ax.set_xlabel("State")
    ax.set_ylabel("Obs")
    fig.colorbar(im, ax=ax)

    ax = fig.add_axes([.1, .37, .85, .25])
    im = ax.imshow(p_estimate, origin="lower", aspect="auto", cmap="Blues",
                   vmin=0, vmax=p_estimate.max())
    mle = np.argmax(p_estimate, axis=0)
    ax.plot(np.arange(p_estimate.shape[1]), mle, "r", lw=2)
    ax.set_title("State posterior and MAP")
    fig.colorbar(im, ax=ax)

    ax = fig.add_axes([.1, .05, .85, .25])
    if obs is not None:
        obs = np.asarray(obs).ravel()
        if obs_times is None:
            obs_times = np.arange(obs.size)
        ax.scatter(np.asarray(obs_times).ravel(), obs, s=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Observation")
    return fig

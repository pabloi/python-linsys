"""Self-contained numerical experiment for regularized Baum-Welch on a
discretized scalar drift-diffusion HMM.

Design-analysis companion to ``hmm_regularized_em.md``. This script does NOT
modify the ``linsys`` package: it imports the forward log-likelihood helper
from ``linsys.hmm`` and implements its own *exact* scaled forward-backward
(to obtain the pairwise xi-statistics that the package inference does not
currently expose) plus four M-step variants:

    (a) unconstrained   - free column-stochastic T / O (plain Baum-Welch)
    (b) banded          - support |i-j| <= k, renormalize columns
    (c) toeplitz        - T[i,j] = f(i-j), pooled along diagonals
    (d) parametric      - discretized AR(1)/drift-diffusion (a, q) for T,
                          Gaussian link (slope, intercept, r) for O

Conventions follow linsys.hmm: T is (D,D) column-stochastic with
T[i,j]=p(next=i|prev=j); O is (M,D) column-stochastic with O[m,d]=p(y=m|x=d).

Runtime target: < 2-3 minutes.
"""
from __future__ import annotations

import time
import numpy as np

from linsys.hmm import hmm_logl, column_normalize


# --------------------------------------------------------------------------
# Ground-truth generative model: discretized AR(1) drift-diffusion
# --------------------------------------------------------------------------
def gaussian_transition(N, a, q, mu):
    """Column-stochastic T from a discretized AR(1): next ~ N(mu+a(s-mu), q).

    Boundary handling is by truncation + per-column renormalization (a
    reflecting-like boundary), which breaks exact Toeplitz structure only in
    the few edge columns.
    """
    s = np.arange(N, dtype=float)
    m = mu + a * (s - mu)              # conditional mean per previous state j
    diff = s[:, None] - m[None, :]      # (i next, j prev)
    T = np.exp(-0.5 * diff ** 2 / q)
    return column_normalize(T)


def gaussian_emission(M, N, r, slope=1.0, intercept=0.0):
    """Column-stochastic O: y ~ N(intercept + slope*x, r) on a symbol grid."""
    sy = np.arange(M, dtype=float)
    sx = np.arange(N, dtype=float)
    mean = intercept + slope * sx
    diff = sy[:, None] - mean[None, :]  # (m symbol, d state)
    O = np.exp(-0.5 * diff ** 2 / r)
    return column_normalize(O)


def simulate(T, O, p0, n, seed):
    rng = np.random.default_rng(seed)
    D, M = T.shape[0], O.shape[0]
    x = np.empty(n, dtype=int)
    y = np.empty(n, dtype=int)
    x[0] = rng.choice(D, p=p0)
    for k in range(n):
        if k > 0:
            x[k] = rng.choice(D, p=T[:, x[k - 1]])
        y[k] = rng.choice(M, p=O[:, x[k]])
    return x, y


# --------------------------------------------------------------------------
# Exact scaled forward-backward -> gamma (smoothed marginals) and the
# accumulated pairwise statistic Xi[i,j] = sum_k p(x_k=j, x_{k+1}=i | Y).
# --------------------------------------------------------------------------
def forward_backward_xi(obs, T, O, p0):
    D = T.shape[0]
    N = obs.size
    alpha = np.empty((D, N))
    c = np.empty(N)                      # scaling factors = p(y_k | y_<k)
    a = O[obs[0], :] * p0
    c[0] = a.sum()
    alpha[:, 0] = a / c[0]
    for k in range(1, N):
        a = O[obs[k], :] * (T @ alpha[:, k - 1])
        c[k] = a.sum()
        alpha[:, k] = a / c[k]

    beta = np.empty((D, N))
    beta[:, N - 1] = 1.0
    for k in range(N - 2, -1, -1):
        b = T.T @ (O[obs[k + 1], :] * beta[:, k + 1])
        beta[:, k] = b / c[k + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=0, keepdims=True)

    # Xi[i,j] = T[i,j] * sum_k (O[y_{k+1},i] beta_{k+1,i}/c_{k+1}) alpha_{k,j}
    A = (O[obs[1:], :].T * beta[:, 1:]) / c[1:]   # (D, N-1), rows = next state i
    B = alpha[:, :-1]                             # (D, N-1), rows = prev state j
    Xi = T * (A @ B.T)
    loglik = np.sum(np.log(c))
    return gamma, Xi, loglik


# --------------------------------------------------------------------------
# M-step variants for T (given accumulated Xi) and O (given gamma + obs).
# --------------------------------------------------------------------------
def mstep_T_unconstrained(Xi, **kw):
    return column_normalize(Xi + 1e-300)


def mstep_T_banded(Xi, bandwidth=2, **kw):
    D = Xi.shape[0]
    i = np.arange(D)[:, None]
    j = np.arange(D)[None, :]
    mask = np.abs(i - j) <= bandwidth
    return column_normalize(Xi * mask + 1e-300)


def mstep_T_toeplitz(Xi, **kw):
    """Pool Xi along diagonals offset d=i-j, then rebuild banded-Toeplitz."""
    D = Xi.shape[0]
    offs = np.arange(-(D - 1), D)
    f = np.array([np.trace(Xi, offset=d) for d in offs], dtype=float)
    f = np.clip(f, 0, None)
    i = np.arange(D)[:, None]
    j = np.arange(D)[None, :]
    T = f[(i - j) + (D - 1)]
    return column_normalize(T + 1e-300)


def mstep_T_parametric(Xi, mu=None, **kw):
    """Weighted moment-match of AR(1) (a, q); rebuild Gaussian-kernel T."""
    D = Xi.shape[0]
    s = np.arange(D, dtype=float)
    if mu is None:
        mu = (D - 1) / 2.0
    u = s - mu
    # weighted regression of next-center on prev-center, weights = Xi[i,j]
    wj = Xi.sum(axis=0)                       # total weight per prev state j
    Suu = np.sum(wj * u ** 2)
    Suv = np.sum(Xi * (s[:, None] - mu) * u[None, :])
    a = Suv / (Suu + 1e-300)
    resid2 = np.sum(Xi * ((s[:, None] - mu) - a * u[None, :]) ** 2)
    q = resid2 / (Xi.sum() + 1e-300)
    q = max(q, 1e-3)
    T = gaussian_transition(D, a, q, mu)
    return T, {"a": a, "q": q}


def mstep_O_unconstrained(gamma, obs, M, **kw):
    D = gamma.shape[0]
    J = np.zeros((M, D))
    for m in range(M):
        J[m, :] = gamma[:, obs == m].sum(axis=1)
    return column_normalize(J + 1e-300)


def mstep_O_banded(gamma, obs, M, bandwidth=2, **kw):
    O = mstep_O_unconstrained(gamma, obs, M)
    D = O.shape[1]
    i = np.arange(M)[:, None]
    j = np.arange(D)[None, :]
    mask = np.abs(i - j) <= bandwidth
    return column_normalize(O * mask + 1e-300)


def mstep_O_toeplitz(gamma, obs, M, **kw):
    J = np.zeros((M, gamma.shape[0]))
    for m in range(M):
        J[m, :] = gamma[:, obs == m].sum(axis=1)
    D = gamma.shape[0]
    offs = np.arange(-(D - 1), M)
    f = np.array([np.trace(J, offset=-d) if False else
                  np.sum(J[np.clip(np.arange(D) + d, 0, M - 1), np.arange(D)]
                         * ((np.arange(D) + d >= 0) & (np.arange(D) + d < M)))
                  for d in offs], dtype=float)
    f = np.clip(f, 0, None)
    i = np.arange(M)[:, None]
    j = np.arange(D)[None, :]
    O = f[(i - j) + (D - 1)]
    return column_normalize(O + 1e-300)


def mstep_O_parametric(gamma, obs, M, **kw):
    """Weighted regression of observed symbol on state; Gaussian-link O."""
    D = gamma.shape[0]
    sx = np.arange(D, dtype=float)
    w = gamma                                   # (D, N)
    yk = obs.astype(float)                      # (N,)
    Wj = w.sum(axis=1)                          # weight per state
    Sx = np.sum(Wj * sx)
    Sxx = np.sum(Wj * sx ** 2)
    Sw = w.sum()
    Sy = np.sum(w * yk[None, :])
    Sxy = np.sum(w * sx[:, None] * yk[None, :])
    denom = Sw * Sxx - Sx ** 2
    slope = (Sw * Sxy - Sx * Sy) / (denom + 1e-300)
    intercept = (Sy - slope * Sx) / (Sw + 1e-300)
    pred = intercept + slope * sx               # mean symbol per state
    resid2 = np.sum(w * (yk[None, :] - pred[:, None]) ** 2)
    r = max(resid2 / (Sw + 1e-300), 1e-3)
    O = gaussian_emission(M, D, r, slope=slope, intercept=intercept)
    return O, {"slope": slope, "intercept": intercept, "r": r}


# --------------------------------------------------------------------------
# EM driver
# --------------------------------------------------------------------------
def run_em(obs, p0, T0, O0, M, T_step, O_step, bandwidth=2, max_iter=40,
           tol=1e-6):
    T, O = T0.copy(), O0.copy()
    ll_hist = []
    info = {}
    for _ in range(max_iter):
        gamma, Xi, ll = forward_backward_xi(obs, T, O, p0)
        ll_hist.append(ll)
        rT = T_step(Xi, bandwidth=bandwidth)
        T, tinfo = rT if isinstance(rT, tuple) else (rT, {})
        rO = O_step(gamma, obs, M, bandwidth=bandwidth)
        O, oinfo = rO if isinstance(rO, tuple) else (rO, {})
        info = {**tinfo, **{("O_" + k): v for k, v in oinfo.items()}}
        if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            break
    return T, O, np.array(ll_hist), info


def main():
    t_start = time.time()
    N = 50                 # number of states / bins
    M = 50                 # number of observation symbols
    N_train = 2000
    N_test = 1000
    mu = (N - 1) / 2.0
    a_true, q_true, r_true = 0.9, 4.0, 4.0

    T_true = gaussian_transition(N, a_true, q_true, mu)
    O_true = gaussian_emission(M, N, r_true, slope=1.0, intercept=0.0)
    p0 = np.zeros(N); p0[N // 2] = 1.0

    _, y_train = simulate(T_true, O_true, p0, N_train, seed=0)
    _, y_test = simulate(T_true, O_true, p0, N_test, seed=1)

    # Shared diffuse initialization (same for every variant)
    rng = np.random.default_rng(42)
    T0 = column_normalize(gaussian_transition(N, 0.5, 25.0, mu)
                          + 0.01 * rng.uniform(size=(N, N)))
    O0 = column_normalize(gaussian_emission(M, N, 25.0)
                          + 0.01 * rng.uniform(size=(M, N)))

    bandwidth = 5     # >= true diffusion scale (sqrt(q)=2 -> +/-2.5 sigma)
    variants = {
        "unconstrained": (mstep_T_unconstrained, mstep_O_unconstrained),
        "banded(k=5)":   (mstep_T_banded,        mstep_O_banded),
        "toeplitz":      (mstep_T_toeplitz,      mstep_O_toeplitz),
        "parametric":    (mstep_T_parametric,    mstep_O_parametric),
    }

    fro_true = np.linalg.norm(T_true)
    ll_test_true = hmm_logl(y_test, O_true, T_true, p0)
    ll_test_init = hmm_logl(y_test, O0, T0, p0)

    rows = []
    for name, (Ts, Os) in variants.items():
        t0 = time.time()
        T, O, ll, info = run_em(y_train, p0, T0, O0, M, Ts, Os,
                                bandwidth=bandwidth)
        dt = time.time() - t0
        fro = np.linalg.norm(T - T_true)
        ll_test = hmm_logl(y_test, O, T, p0)
        rows.append((name, len(ll), ll[-1] / N_train, ll_test / N_test,
                     fro, fro / fro_true, info, dt))

    print("\n=== Discretized drift-diffusion HMM: regularized Baum-Welch ===")
    print(f"N states = {N}, M symbols = {M}, N_train = {N_train}, "
          f"N_test = {N_test}")
    print(f"true (a,q,r) = ({a_true}, {q_true}, {r_true})")
    print(f"params: full T={N*N}, banded(k={bandwidth})~{N*(2*bandwidth+1)}, "
          f"toeplitz~{2*N}, parametric=2;  "
          f"effective transitions={N_train-1}")
    print(f"\ntrue-model  test logL/sample = {ll_test_true/N_test:.4f}")
    print(f"init        test logL/sample = {ll_test_init/N_test:.4f}")
    print("\n{:<14} {:>5} {:>12} {:>12} {:>10} {:>9}  {}".format(
        "variant", "iters", "train LL/n", "test LL/n", "||dT||_F",
        "rel", "recovered params"))
    for name, niter, lltr, llte, fro, rel, info, dt in rows:
        ip = ", ".join(f"{k}={v:.3f}" for k, v in info.items()) if info else "-"
        print("{:<14} {:>5} {:>12.4f} {:>12.4f} {:>10.4f} {:>8.2f}  {}".format(
            name, niter, lltr, llte, fro, rel, ip))
    print(f"\ntotal runtime: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()

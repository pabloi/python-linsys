"""Shared synthetic-data helpers for tests."""
import numpy as np

from linsys.utils import fwd_sim


def make_system(nx=2, ny=3, nu=1, seed=0, stable=(0.95, 0.7), qscale=1e-3,
                rscale=1e-2):
    rng = np.random.default_rng(seed)
    A = np.diag(stable[:nx]) if nx <= len(stable) else np.diag(
        np.linspace(0.95, 0.5, nx))
    B = rng.standard_normal((nx, nu))
    C = rng.standard_normal((ny, nx))
    D = rng.standard_normal((ny, nu))
    Q = qscale * np.eye(nx)
    R = rscale * np.eye(ny)
    return A, B, C, D, Q, R


def simulate(A, B, C, D, Q, R, N=300, seed=1, x0=None, step_input=True):
    nu = B.shape[1]
    U = np.ones((nu, N)) if step_input else \
        np.random.default_rng(seed + 1).standard_normal((nu, N))
    Y, X = fwd_sim(U, A, B, C, D, x0=x0, Q=Q, R=R, rng=seed)
    return Y, X, U


def naive_kalman_filter(Y, A, C, Q, R, x0, P0, B, D, U):
    """Reference textbook implementation (no tricks) for validation."""
    ny, N = Y.shape
    nx = A.shape[0]
    X = np.zeros((nx, N))
    P = np.zeros((N, nx, nx))
    Xp = np.zeros((nx, N + 1))
    Pp = np.zeros((N + 1, nx, nx))
    x, Pk = x0.copy(), P0.copy()
    Xp[:, 0] = x
    Pp[0] = Pk
    logL = 0.0
    for k in range(N):
        y = Y[:, k] - D @ U[:, k]
        if not np.isnan(y).any():
            S = C @ Pk @ C.T + R
            iS = np.linalg.inv(S)
            innov = y - C @ x
            K = Pk @ C.T @ iS
            x = x + K @ innov
            Pk = (np.eye(nx) - K @ C) @ Pk
            logL += (-0.5 * innov @ iS @ innov
                     - 0.5 * np.log(np.linalg.det(S))
                     - 0.5 * len(y) * np.log(2 * np.pi))
        X[:, k] = x
        P[k] = Pk
        x = A @ x + B @ U[:, k]
        Pk = A @ Pk @ A.T + Q
        Xp[:, k + 1] = x
        Pp[k + 1] = Pk
    return X, P, Xp, Pp, logL


def naive_rts_smoother(Y, A, C, Q, R, x0, P0, B, D, U):
    X, P, Xp, Pp, logL = naive_kalman_filter(Y, A, C, Q, R, x0, P0, B, D, U)
    N = Y.shape[1]
    nx = A.shape[0]
    Xs = X.copy()
    Ps = P.copy()
    Pt = np.zeros((N - 1, nx, nx))
    for k in range(N - 2, -1, -1):
        H = P[k] @ A.T @ np.linalg.inv(Pp[k + 1])
        Xs[:, k] = X[:, k] + H @ (Xs[:, k + 1] - Xp[:, k + 1])
        Ps[k] = P[k] + H @ (Ps[k + 1] - Pp[k + 1]) @ H.T
        Pt[k] = Ps[k + 1] @ H.T  # cov(x[k+1], x[k] | all data)
    return Xs, Ps, Pt, X, P, logL

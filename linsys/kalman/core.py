"""Elementary Kalman steps: update, predict, information form, model reduction.

Ports of KFupdate, KFpredict, infoUpdate, info2state, state2info, reduceModel.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

from ..utils import cholcov2, pinvchol, pinvchol2, _HALF_LOG_2PI

_LOG_2PI = 1.83787706640934529


def kf_update(C, R, y, x, P, reject_z2_threshold=None, want_logl=False):
    """Kalman measurement update (MATLAB: KFupdate).

    Returns (new_x, new_P, iL, rejected, logL) where iL = inv(chol(S)) for the
    innovation covariance S = R + C P C' (upper-triangular cS, cS.T@cS = S).
    logL is None unless want_logl.
    """
    S = R + C @ P @ C.T
    cS = cholcov2(S)
    iL = scipy.linalg.solve(cS, np.eye(S.shape[0]))
    CiL = C.T @ iL
    PCiL = P @ CiL

    innov = y - C @ x
    iLy = iL.T @ innov
    z2 = iLy @ iLy
    logL = None
    if want_logl or reject_z2_threshold is not None:
        half_logdet_sigma = np.sum(np.log(np.diag(cS)))
        logL = -0.5 * z2 - half_logdet_sigma - y.shape[0] * _HALF_LOG_2PI
    if reject_z2_threshold is not None and z2 > reject_z2_threshold:
        return x, P, iL, True, logL
    new_x = x + PCiL @ iLy
    K = PCiL @ iL.T
    H = np.eye(P.shape[0]) - PCiL @ CiL.T
    new_P = H @ P @ H.T + K @ R @ K.T  # Joseph-like form for PSD-ness
    return new_x, new_P, iL, False, logL


def kf_predict(A, Q, x, P, b=None):
    """Kalman prediction step (MATLAB: KFpredict)."""
    x = A @ x if b is None else A @ x + b
    P = A @ P @ A.T + Q
    return x, (P + P.T) / 2


def state_to_info(x0, P0):
    """Convert a (mean, covariance) pair to information form (i, I), handling
    infinite variances (MATLAB: state2info)."""
    x0 = np.asarray(x0, dtype=float).ravel().copy()
    P0 = np.atleast_2d(P0)
    n = P0.shape[0]
    dP = np.diag(P0)
    inf_var = np.isinf(dP)
    if inf_var.any():
        I = np.zeros((n, n))
        fin = np.where(~inf_var)[0]
        if fin.size:
            _, _, redI = pinvchol(P0[np.ix_(fin, fin)])
            I[np.ix_(fin, fin)] = redI
        x0[inf_var] = 0.0
    else:
        _, _, I = pinvchol(P0)
    i = I @ x0
    zero_var = np.where(dP == 0)[0]  # exactly-known states: infinite information
    if zero_var.size:
        I[zero_var, :] = 0.0
        I[:, zero_var] = 0.0
        I[zero_var, zero_var] = np.inf
    return i, I


def info_to_state(i, I):
    """Convert information pair (i, I) to (mean, covariance), restoring
    infinite variances where information is zero (MATLAB: info2state)."""
    _, _, P = pinvchol2(I)
    x = P @ i
    zero_info = np.where(np.diag(I) == 0)[0]
    if zero_info.size:
        P = P.copy()
        P[zero_info, zero_info] = np.inf
        x = x.copy()
        x[zero_info] = 0.0
    return x, P


def info_update(CtRinvC, CtRinvY, x, P, reject_z2_threshold=None,
                logdet_crc=None, inv_crc=None, old_I=None, want_logl=False):
    """Information-form measurement update (MATLAB: infoUpdate).

    Returns (new_i, new_I, new_x, new_P, logL, rejected, old_I).
    """
    rejected = False
    need_logl = want_logl or reject_z2_threshold is not None
    if need_logl and (inv_crc is None or logdet_crc is None):
        chol_inv_crc, _, inv_crc = pinvchol(CtRinvC)
        logdet_crc = -2 * np.sum(np.log(np.diag(chol_inv_crc)))
    chol_old_I = None
    if old_I is None:
        chol_old_I, _, old_I = pinvchol2(P)
        old_I = chol_old_I @ chol_old_I.T
    new_I = old_I + CtRinvC
    new_i = old_I @ x + CtRinvY

    chol_P, chol_inv_P, new_P = pinvchol2(new_I)
    new_x = new_P @ new_i
    logL = None
    if need_logl:
        if chol_old_I is None or chol_old_I.size == 0:
            logdet_old_I = 0.0
        else:
            d = np.diag(chol_old_I)
            logdet_old_I = np.log(d[d > 0]).sum()
        z = CtRinvY - CtRinvC @ x
        invP = inv_crc - new_P
        dP = np.diag(chol_P)
        logdetP = 2 * (np.sum(np.log(dP[dP > 0])) - logdet_old_I) + logdet_crc
        z2 = z @ (invP @ z)
        logL = -0.5 * z2 - 0.5 * logdetP - z.shape[0] * _HALF_LOG_2PI
        if reject_z2_threshold is not None and z2 > reject_z2_threshold:
            return old_I @ x, old_I, x, P, logL, True, old_I
    return new_i, new_I, new_x, new_P, logL, rejected, old_I


def reduce_model(C, R, Y, D=None):
    """Reduce an output of dimension ny > nx to an equivalent nx-dimensional
    one for filtering efficiency (MATLAB: reduceModel).

    Returns (Cnew, Rnew, Ynew, cRnew, logL_margin, Dnew): the reduced model
    satisfies Cnew = Rnew = C' inv(R) C and Ynew = C' inv(R) Y. logL_margin is
    a per-sample correction so that logL(original) = logL(reduced) + margin.
    """
    ny, nx = C.shape
    icR, cR, _ = pinvchol(R)
    J = C.T @ icR
    Yaux = icR.T @ Y
    Rnew = J @ J.T
    Ynew = J @ Yaux
    Cnew = Rnew
    icRnew, cRnew, _ = pinvchol(Rnew)

    dim_margin = ny - nx
    if dim_margin < 0:
        logl_margin = np.zeros(Y.shape[1] if Y.ndim > 1 else 1)
    else:
        logdet_margin = np.sum(np.log(np.diag(cRnew))) - np.sum(np.log(np.diag(cR)))
        aux = icRnew.T @ Ynew
        z2_margin = np.sum(Yaux ** 2, axis=0) - np.sum(aux ** 2, axis=0)
        logl_margin = -0.5 * (z2_margin + dim_margin * _LOG_2PI) + logdet_margin
    Dnew = None
    if D is not None:
        Dnew = (C.T @ icR) @ (icR.T @ D)
    return Cnew, Rnew, Ynew, cRnew, logl_margin, Dnew

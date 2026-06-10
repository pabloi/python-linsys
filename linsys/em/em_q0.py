"""EM with Q fixed to zero (port of matlab-linsys/EM/EM_Q0.m).

The MATLAB EM_Q0 is a stale standalone copy of the EM loop (it calls
estimateParams without the now-required opts argument, so it errors as-is,
and it relies on an approximate logL). Since the modern EM machinery already
supports fixing parameters, this port implements EM_Q0 as em() with
fix_q = 0, which is the documented intent ("a true EM implementation to do
LTI-SSM identification, imposing Q=0").

With Q = 0 the states are a deterministic trajectory of the initial state,
so the initial uncertainty must be fixed externally: the default M-step
P0 = Q would give P0 = 0 and the smoothed states could never move away from
the initial guess, while an improper (infinite) prior makes the backward
smoothing pass degenerate when Q = 0. We default to a flat-but-proper
fix_p0 = 100*I, which is large relative to the state scale set by the
initialization (states are normalized to row-norm 100, i.e. typical
magnitude ~100/sqrt(N)).
"""
from __future__ import annotations

import numpy as np

from .em import EMResult, em
from .opts import EMOpts


def em_q0(Y, U, x_guess, opts: EMOpts | None = None, rng=None) -> EMResult:
    """EM identification imposing Q = 0 (deterministic states) (MATLAB: EM_Q0).

    Same interface as em(). opts.fix_q is overridden to zeros and stable_a
    is enforced; if fix_p0 was not provided it is set to a flat proper prior
    (100*I, see module docstring). Default Niter is 101 (as in MATLAB
    EM_Q0), unless opts.Niter is set. Note that with Q = 0 the E/M steps are
    not an exact EM ascent, so logL may oscillate; the best model found is
    returned.
    """
    o = EMOpts() if opts is None else opts.copy()
    o.fix_q = np.nan  # special flag: fix Q to zeros (resolved in process_em_opts)
    if o.fix_p0 is None:
        if np.isscalar(x_guess):
            nx = int(x_guess)
        else:
            nx = (np.atleast_2d(x_guess[0]) if isinstance(x_guess, list)
                  else np.atleast_2d(x_guess)).shape[0]
        o.fix_p0 = 1e2 * np.eye(nx)  # flat (but proper) prior
    if o.Niter is None:
        o.Niter = 101
    # With Q = 0 an unstable A iterate makes filtering diverge (the M-step
    # for A is not likelihood-based in this degenerate case), so stability
    # is enforced by default:
    o.stable_a = True
    return em(Y, U, x_guess, o, rng=rng)

"""Smooth/dynamic PCA (port of matlab-linsys/sPCA).

MATLAB -> Python mapping:
    sPCAv8.m          -> spca.spca           (sPCAv1-v7 are in old/, skipped)
    estimateDynv3b.m  -> estimate_dyn.estimate_dyn (alias estimate_dyn_v3b)
    estimateDynv4.m   -> estimate_dyn.estimate_dyn_v4
    CVsPCA.m          -> cv.cv_spca
    chngInitState.m   -> spca._chng_init_state (private helper)
"""
from .cv import cv_spca
from .estimate_dyn import (
    EstimateDynResult, estimate_dyn, estimate_dyn_v3b, estimate_dyn_v4,
)
from .spca import SPCAModel, spca

__all__ = [
    "SPCAModel", "spca", "cv_spca",
    "EstimateDynResult", "estimate_dyn", "estimate_dyn_v3b",
    "estimate_dyn_v4",
]

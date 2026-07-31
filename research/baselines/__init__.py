"""Baselines package: SOTA optimal / near-optimal tree-ensemble CF methods.

Every backend conforms to :class:`baselines.base.BaselineExplainer` and returns
a :class:`baselines.base.CFResult`, so the experiment harness can call them
interchangeably on the same forest, query, distance, and time limit.

See BASELINES.md for the sourcing brief and STATUS.md for per-method provenance.
"""
from __future__ import annotations

from .base import BaselineExplainer, BaselineUnavailable, CFResult
from .brute_force import BruteForceExplainer
from .cfmaps import CounterfactualMapsExplainer
from .milp_soft import MilpSoftExplainer
from .ocean import (
    OceanCPExplainer,
    OceanMaxSATExplainer,
    OceanMILPExplainer,
    OceanSATExplainer,
)
from .registry import (
    OPTIMAL_METHODS,
    BackendStatus,
    all_keys,
    build,
    build_available,
)
from .calf_method import CALFExplainer

__all__ = [
    "BaselineExplainer",
    "BaselineUnavailable",
    "CFResult",
    "BruteForceExplainer",
    "CounterfactualMapsExplainer",
    "MilpSoftExplainer",
    "CALFExplainer",
    "OceanMILPExplainer",
    "OceanCPExplainer",
    "OceanMaxSATExplainer",
    "OceanSATExplainer",
    "all_keys",
    "build",
    "build_available",
    "BackendStatus",
    "OPTIMAL_METHODS",
]

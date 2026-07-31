"""Backend discovery: build whichever explainers are actually available.

The harness asks the registry for a set of explainers on a given forest; any
backend whose dependency is missing (Gurobi, OCEAN, OR-Tools, …) is *skipped*
with a recorded reason rather than crashing the run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from .base import BaselineExplainer, BaselineUnavailable
from .brute_force import BruteForceExplainer
from .cfmaps import CounterfactualMapsExplainer
from .milp_soft import MilpSoftExplainer
from .ocean import (
    OceanCPExplainer,
    OceanMaxSATExplainer,
    OceanMILPExplainer,
    OceanSATExplainer,
)
from .calf_method import CALFExplainer

# Registry order = the order methods appear in the smoke-test table.
_REGISTRY: dict[str, Type[BaselineExplainer]] = {
    "calf": CALFExplainer,
    # Ablation aliases: the same engine under distinct method names, so one
    # suite config can run several calf arms side by side (which arm each is
    # comes from its method_kwargs in the config, e.g. dual_round_polish /
    # alpha_warm).
    "calf_prev": CALFExplainer,
    "calf_polish": CALFExplainer,
    "calf_warm": CALFExplainer,
    "calf_warm_early": CALFExplainer,
    "calf_warmlp": CALFExplainer,
    "calf_warmlp64": CALFExplainer,
    "calf_budget": CALFExplainer,
    "calf_esc": CALFExplainer,
    "brute_force": BruteForceExplainer,
    "counterfactual_maps": CounterfactualMapsExplainer,
    "milp_soft": MilpSoftExplainer,
    "ocean_milp": OceanMILPExplainer,
    "ocean_cp": OceanCPExplainer,
    # Thread-count ablation aliases: same OCEAN backends, but the suite passes
    # `threads: 4` via method_kwargs so one config can run a 4-thread arm next to
    # the single-threaded default (recorded under a distinct method name).
    "ocean_milp_t4": OceanMILPExplainer,
    "ocean_cp_t4": OceanCPExplainer,
    "ocean_maxsat": OceanMaxSATExplainer,
    "ocean_sat": OceanSATExplainer,
}

# Methods that certify the true global optimum under the shared metric (used by
# the smoke test to decide which methods must agree on the objective).
OPTIMAL_METHODS = frozenset(
    {"calf", "brute_force", "counterfactual_maps", "milp_soft",
     "ocean_milp", "ocean_cp", "ocean_milp_t4", "ocean_cp_t4", "ocean_maxsat"}
)


@dataclass
class BackendStatus:
    key: str
    available: bool
    explainer: BaselineExplainer | None
    reason: str | None = None


def all_keys() -> list[str]:
    return list(_REGISTRY)


def build(key: str, rf, **kw) -> BaselineExplainer:
    """Construct one explainer; raises BaselineUnavailable if its deps are missing."""
    if key not in _REGISTRY:
        raise KeyError(f"unknown baseline {key!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key](rf, **kw)


def build_available(
    rf, keys: list[str] | None = None, *, per_backend_kwargs: dict | None = None, **common_kw
) -> list[BackendStatus]:
    """Build every requested backend, skipping unavailable ones.

    ``per_backend_kwargs`` maps a key to extra constructor kwargs (e.g.
    ``{"calf": {"max_iters": 200_000}}``); merged over ``common_kw``.
    """
    keys = keys or all_keys()
    per_backend_kwargs = per_backend_kwargs or {}
    out: list[BackendStatus] = []
    for key in keys:
        kw = {**common_kw, **per_backend_kwargs.get(key, {})}
        try:
            expl = build(key, rf, **kw)
            out.append(BackendStatus(key=key, available=True, explainer=expl))
        except BaselineUnavailable as exc:
            out.append(BackendStatus(key=key, available=False, explainer=None, reason=str(exc)))
        except Exception as exc:  # construction bug in an available backend
            out.append(
                BackendStatus(
                    key=key, available=False, explainer=None,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return out

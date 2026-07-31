"""Result type for a counterfactual extraction run."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExtractionResult:
    x: np.ndarray | None       # the counterfactual point, or None if none found
    cost: float                # scaled-L1 cost of x (inf if not found)
    optimality_gap: float      # 0.0 => certified optimal
    iters: int                 # search nodes popped
    found: bool
    target_class: int

    @property
    def proven_optimal(self) -> bool:
        return self.found and self.optimality_gap == 0.0

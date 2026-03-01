"""Result data containers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SubcaseResult:
    subcase_id: int = 0
    # SOL 101
    displacements: Dict[int, np.ndarray] = field(default_factory=dict)
    spc_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    # SOL 103
    eigenvalues: Optional[np.ndarray] = None       # omega^2 values
    frequencies: Optional[np.ndarray] = None        # Hz
    mode_shapes: List[Dict[int, np.ndarray]] = field(default_factory=list)
    eigenvectors_full: List[np.ndarray] = field(default_factory=list)


@dataclass
class ResultData:
    title: str = ""
    subcases: List[SubcaseResult] = field(default_factory=list)

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
    # SOL 144
    trim_variables: Optional[Dict[str, float]] = None
    aero_pressures: Optional[np.ndarray] = None
    aero_forces: Optional[np.ndarray] = None
    aero_boxes: Optional[list] = None
    # SOL 144 nodal loads (Dict[node_id, np.ndarray(6)])
    nodal_aero_forces: Optional[Dict[int, np.ndarray]] = None
    nodal_inertial_forces: Optional[Dict[int, np.ndarray]] = None
    nodal_combined_forces: Optional[Dict[int, np.ndarray]] = None
    trim_balance: Optional[Dict[str, float]] = None  # 6-DOF balance check


@dataclass
class ResultData:
    title: str = ""
    subcases: List[SubcaseResult] = field(default_factory=list)

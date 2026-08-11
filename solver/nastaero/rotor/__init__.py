"""Rotor aerodynamics package for VTOL loads analysis.

Provides Blade Element Momentum Theory (BEMT) solvers for computing
rotor thrust, torque, and hub loads in hover, climb, and forward flight.
"""
from __future__ import annotations

from .airfoil import RotorAirfoil
from .blade import BladeDef
from .bemt_solver import BEMTSolver, RotorLoads

__all__ = ["RotorAirfoil", "BladeDef", "BEMTSolver", "RotorLoads"]

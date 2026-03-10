"""Rotor blade airfoil aerodynamic model.

Provides Cl, Cd, Cm lookup for blade element analysis. Supports both
linear model (default) and tabulated data. Default values correspond
to NACA 0012 at moderate Reynolds numbers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RotorAirfoil:
    """2D airfoil aerodynamic coefficient model.

    Attributes
    ----------
    Cl_alpha : float
        Lift curve slope (per radian). Default 2π (thin airfoil).
    alpha_0 : float
        Zero-lift angle of attack (radians). Default 0 (symmetric).
    Cd_0 : float
        Minimum drag coefficient.
    Cd_1 : float
        Linear drag term: Cd = Cd_0 + Cd_1*|alpha| + Cd_2*alpha^2.
    Cd_2 : float
        Quadratic drag term.
    Cm_0 : float
        Pitching moment coefficient at alpha=0. Default 0 (symmetric).
    alpha_stall : float
        Stall angle of attack (radians). Beyond this, Cl is capped.
    alpha_table : np.ndarray, optional
        Tabulated alpha values (radians) for custom data.
    Cl_table : np.ndarray, optional
        Tabulated Cl values corresponding to alpha_table.
    Cd_table : np.ndarray, optional
        Tabulated Cd values corresponding to alpha_table.
    """
    Cl_alpha: float = 2.0 * np.pi
    alpha_0: float = 0.0
    Cd_0: float = 0.008
    Cd_1: float = 0.0
    Cd_2: float = 0.3
    Cm_0: float = 0.0
    alpha_stall: float = np.radians(12.0)
    alpha_table: Optional[np.ndarray] = None
    Cl_table: Optional[np.ndarray] = None
    Cd_table: Optional[np.ndarray] = None

    def cl(self, alpha: float) -> float:
        """Lift coefficient at angle of attack.

        Parameters
        ----------
        alpha : float
            Angle of attack in radians.

        Returns
        -------
        float
            Lift coefficient.
        """
        if self.alpha_table is not None and self.Cl_table is not None:
            return float(np.interp(alpha, self.alpha_table, self.Cl_table))

        # Linear model with stall clamp
        cl_val = self.Cl_alpha * (alpha - self.alpha_0)
        cl_max = self.Cl_alpha * (self.alpha_stall - self.alpha_0)
        return float(np.clip(cl_val, -cl_max, cl_max))

    def cd(self, alpha: float) -> float:
        """Drag coefficient at angle of attack.

        Parameters
        ----------
        alpha : float
            Angle of attack in radians.

        Returns
        -------
        float
            Drag coefficient.
        """
        if self.alpha_table is not None and self.Cd_table is not None:
            return float(np.interp(alpha, self.alpha_table, self.Cd_table))

        return self.Cd_0 + self.Cd_1 * abs(alpha) + self.Cd_2 * alpha ** 2

    def cm(self, alpha: float) -> float:
        """Pitching moment coefficient (about quarter-chord).

        Parameters
        ----------
        alpha : float
            Angle of attack in radians.

        Returns
        -------
        float
            Pitching moment coefficient.
        """
        return self.Cm_0

    @classmethod
    def naca0012(cls) -> RotorAirfoil:
        """Create NACA 0012 airfoil model (default parameters)."""
        return cls(
            Cl_alpha=2 * np.pi * 0.9,  # ~5.65 /rad (typical for NACA 0012)
            alpha_0=0.0,
            Cd_0=0.008,
            Cd_1=0.0,
            Cd_2=0.30,
            Cm_0=0.0,
            alpha_stall=np.radians(12.0),
        )

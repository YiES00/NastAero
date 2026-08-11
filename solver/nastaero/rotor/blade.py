"""Rotor blade geometry definition.

Defines chord, twist, and airfoil distributions along the blade span
for use with BEMT solvers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .airfoil import RotorAirfoil


@dataclass
class BladeDef:
    """Rotor blade geometry and aerodynamic properties.

    Attributes
    ----------
    radius : float
        Blade radius (m) from hub center to tip.
    root_cutout : float
        Non-lifting root cutout as fraction of radius (0-1). Default 0.15.
    n_elements : int
        Number of blade elements for BEMT discretization. Default 20.
    chord_dist : np.ndarray, optional
        Chord distribution [n_stations x 2]: (r/R, chord_m).
        If None, uses constant chord from mean_chord.
    twist_dist : np.ndarray, optional
        Twist distribution [n_stations x 2]: (r/R, twist_rad).
        If None, uses linear twist from twist_root to twist_tip.
    mean_chord : float
        Mean blade chord (m). Used if chord_dist is None. Default 0.05.
    twist_root : float
        Blade twist at root (radians). Default 12 deg.
    twist_tip : float
        Blade twist at tip (radians). Default 3 deg.
    airfoil : RotorAirfoil
        Airfoil properties (uniform along span).
    """
    radius: float = 0.6
    root_cutout: float = 0.15
    n_elements: int = 20
    chord_dist: Optional[np.ndarray] = None
    twist_dist: Optional[np.ndarray] = None
    mean_chord: float = 0.05
    twist_root: float = np.radians(12.0)
    twist_tip: float = np.radians(3.0)
    airfoil: RotorAirfoil = field(default_factory=RotorAirfoil.naca0012)

    def get_stations(self) -> np.ndarray:
        """Get radial station positions (r/R) for BEMT elements.

        Returns
        -------
        np.ndarray
            Midpoint r/R values of each annular element, shape (n_elements,).
        """
        r_start = self.root_cutout
        r_end = 1.0
        edges = np.linspace(r_start, r_end, self.n_elements + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def get_dr(self) -> float:
        """Get radial width of each element (m).

        Returns
        -------
        float
            Element width in meters.
        """
        return self.radius * (1.0 - self.root_cutout) / self.n_elements

    def chord_at(self, r_over_R: float) -> float:
        """Chord length at radial station.

        Parameters
        ----------
        r_over_R : float
            Normalized radial position (0 to 1).

        Returns
        -------
        float
            Chord in meters.
        """
        if self.chord_dist is not None:
            return float(np.interp(r_over_R, self.chord_dist[:, 0],
                                   self.chord_dist[:, 1]))
        return self.mean_chord

    def twist_at(self, r_over_R: float) -> float:
        """Blade twist angle at radial station.

        Parameters
        ----------
        r_over_R : float
            Normalized radial position (0 to 1).

        Returns
        -------
        float
            Twist angle in radians (positive nose-up).
        """
        if self.twist_dist is not None:
            return float(np.interp(r_over_R, self.twist_dist[:, 0],
                                   self.twist_dist[:, 1]))
        # Linear twist
        t = (r_over_R - self.root_cutout) / (1.0 - self.root_cutout)
        t = np.clip(t, 0.0, 1.0)
        return self.twist_root + t * (self.twist_tip - self.twist_root)

    @property
    def solidity(self) -> float:
        """Rotor solidity σ = N_b * c / (π * R).

        Uses mean chord. Note: n_blades not stored here — call from outside.
        """
        return self.mean_chord / (np.pi * self.radius)

    def blade_solidity(self, n_blades: int) -> float:
        """Rotor solidity σ = N_b * c / (π * R)."""
        return n_blades * self.mean_chord / (np.pi * self.radius)

"""Blade Element Momentum Theory (BEMT) solver for axial flight.

Computes rotor thrust, torque, and power in hover and axial climb/descent
using combined blade element and momentum theory with Prandtl tip-loss
correction.

References
----------
- Leishman, J.G., "Principles of Helicopter Aerodynamics", 2nd ed., Ch. 3
- Johnson, W., "Rotorcraft Aeromechanics", Ch. 3-4
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .blade import BladeDef


@dataclass
class RotorLoads:
    """Integrated rotor hub loads.

    All forces/moments are in the rotor shaft frame:
    - Z-axis: along shaft (positive = thrust direction, upward for lift rotor)
    - X-axis: forward (for H-force)
    - Y-axis: lateral

    Attributes
    ----------
    thrust : float
        Axial thrust force (N). Positive along shaft axis.
    torque : float
        Shaft torque (N·m). Positive = reaction torque opposing rotation.
    power : float
        Shaft power (W). P = Q × Ω.
    H_force : float
        In-plane H-force (N) in forward flight. Zero in hover.
    roll_moment : float
        Rolling moment about X-axis (N·m).
    pitch_moment : float
        Pitching moment about Y-axis (N·m).
    CT : float
        Thrust coefficient: T / (ρ A Ω²R²).
    CQ : float
        Torque coefficient: Q / (ρ A Ω²R³).
    CP : float
        Power coefficient: P / (ρ A Ω³R³). Equal to CQ.
    collective_rad : float
        Collective pitch used (radians).
    """
    thrust: float = 0.0
    torque: float = 0.0
    power: float = 0.0
    H_force: float = 0.0
    roll_moment: float = 0.0
    pitch_moment: float = 0.0
    CT: float = 0.0
    CQ: float = 0.0
    CP: float = 0.0
    collective_rad: float = 0.0


class BEMTSolver:
    """BEMT solver for axial flight (hover and climb).

    Uses iterative solution of the combined momentum-blade element
    equations with Prandtl tip-loss factor.

    Parameters
    ----------
    blade : BladeDef
        Blade geometry and airfoil properties.
    n_blades : int
        Number of blades.
    max_iter : int
        Maximum iterations for induction factor convergence.
    tol : float
        Convergence tolerance for induction factor.
    """

    def __init__(self, blade: BladeDef, n_blades: int = 4,
                 max_iter: int = 100, tol: float = 1e-6):
        self.blade = blade
        self.n_blades = n_blades
        self.max_iter = max_iter
        self.tol = tol

    def _prandtl_tip_loss(self, r_over_R: float, phi: float) -> float:
        """Prandtl tip-loss factor.

        Parameters
        ----------
        r_over_R : float
            Normalized radial position.
        phi : float
            Inflow angle (radians).

        Returns
        -------
        float
            Tip-loss factor F (0 to 1).
        """
        if abs(phi) < 1e-10 or r_over_R >= 0.999:
            return 1e-4
        f_arg = self.n_blades * (1.0 - r_over_R) / (2.0 * r_over_R * abs(np.sin(phi)))
        f_arg = min(f_arg, 20.0)  # Prevent overflow
        F = (2.0 / np.pi) * np.arccos(np.exp(-f_arg))
        return max(F, 1e-4)

    def _solve_element(self, r_over_R: float, omega: float, V_climb: float,
                       rho: float, collective: float
                       ) -> tuple[float, float, float, float]:
        """Solve BEMT for a single blade element.

        Parameters
        ----------
        r_over_R : float
            Normalized radial position.
        omega : float
            Rotational speed (rad/s).
        V_climb : float
            Climb velocity (m/s), positive upward.
        rho : float
            Air density (kg/m³).
        collective : float
            Collective pitch (radians) added to blade twist.

        Returns
        -------
        dT : float
            Elemental thrust (N).
        dQ : float
            Elemental torque (N·m).
        a : float
            Converged axial induction factor.
        phi : float
            Converged inflow angle (radians).
        """
        R = self.blade.radius
        r = r_over_R * R
        dr = self.blade.get_dr()
        c = self.blade.chord_at(r_over_R)
        theta = self.blade.twist_at(r_over_R) + collective
        sigma_r = self.n_blades * c / (2.0 * np.pi * r) if r > 0 else 0.0

        U_T = omega * r  # Tangential velocity
        if U_T < 1e-6:
            return 0.0, 0.0, 0.0, 0.0

        # Initial guess for axial induction
        a = 0.05

        for _ in range(self.max_iter):
            # Axial velocity component (inflow)
            U_P = V_climb + a * omega * R  # Induced velocity ~ a*ΩR for hover

            # Inflow angle
            phi = np.arctan2(U_P, U_T) if U_T > 0 else np.pi / 2
            alpha = theta - phi

            # Airfoil coefficients
            cl = self.blade.airfoil.cl(alpha)
            cd = self.blade.airfoil.cd(alpha)

            # Tip-loss factor
            F = self._prandtl_tip_loss(r_over_R, phi)

            # Element force coefficients
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            # CT/sigma for this element (from blade element theory)
            # dT = 0.5 * rho * (U_T² + U_P²) * c * (Cl*cos_phi - Cd*sin_phi) * dr * Nb
            W2 = U_T ** 2 + U_P ** 2

            # Momentum theory: dT = 4πr * F * ρ * vi * (V_climb + vi) * dr
            # Combined: solve for 'a' via Newton or relaxation
            if abs(sin_phi) > 1e-10:
                # New induction factor from blade element result
                Cn = cl * cos_phi + cd * sin_phi  # Normal force coeff
                a_new = sigma_r * Cn / (4.0 * F * sin_phi ** 2)

                # Glauert correction for high induction (a > 0.4)
                if a_new > 0.4:
                    # Use empirical Glauert correction
                    K = 4.0 * F * sin_phi ** 2 / sigma_r
                    a_new = 0.5 * (2.0 + K * (1.0 - 2.0 * 0.4)
                                   - np.sqrt((K * (1.0 - 2.0 * 0.4) + 2.0) ** 2
                                             + 4.0 * (K * 0.4 ** 2 - 1.0)))
                    a_new = max(a_new, 0.0)
            else:
                a_new = a

            # Relaxation
            a_old = a
            a = 0.3 * a_new + 0.7 * a_old

            if abs(a - a_old) < self.tol:
                break

        # Final loads
        U_P = V_climb + a * omega * R
        phi = np.arctan2(U_P, U_T) if U_T > 0 else np.pi / 2
        alpha = theta - phi
        cl = self.blade.airfoil.cl(alpha)
        cd = self.blade.airfoil.cd(alpha)

        W2 = U_T ** 2 + U_P ** 2
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        dL = 0.5 * rho * W2 * c * cl * dr  # Lift per unit span per blade
        dD = 0.5 * rho * W2 * c * cd * dr  # Drag per unit span per blade

        dT = self.n_blades * (dL * cos_phi - dD * sin_phi)
        dQ = self.n_blades * (dL * sin_phi + dD * cos_phi) * r

        return dT, dQ, a, phi

    def solve(self, rpm: float, V_inf: float, rho: float,
              collective_rad: float = 0.0) -> RotorLoads:
        """Solve BEMT for entire rotor in axial flight.

        Parameters
        ----------
        rpm : float
            Rotor speed in RPM.
        V_inf : float
            Freestream velocity along shaft axis (m/s).
            Positive = climb, negative = descent.
        rho : float
            Air density (kg/m³).
        collective_rad : float
            Additional collective pitch (radians). Added to blade twist.

        Returns
        -------
        RotorLoads
            Integrated hub loads.
        """
        omega = rpm * 2.0 * np.pi / 60.0
        R = self.blade.radius
        A = np.pi * R ** 2

        stations = self.blade.get_stations()
        total_T = 0.0
        total_Q = 0.0

        for r_over_R in stations:
            dT, dQ, _, _ = self._solve_element(
                r_over_R, omega, V_inf, rho, collective_rad)
            total_T += dT
            total_Q += dQ

        total_P = total_Q * omega

        # Non-dimensionalize
        denom_T = rho * A * (omega * R) ** 2 if omega > 0 else 1.0
        denom_Q = rho * A * (omega * R) ** 2 * R if omega > 0 else 1.0
        CT = total_T / denom_T if denom_T > 0 else 0.0
        CQ = total_Q / denom_Q if denom_Q > 0 else 0.0
        CP = CQ  # CP = CQ in coefficient form

        return RotorLoads(
            thrust=total_T,
            torque=total_Q,
            power=total_P,
            H_force=0.0,  # Zero in axial flight
            roll_moment=0.0,
            pitch_moment=0.0,
            CT=CT,
            CQ=CQ,
            CP=CP,
            collective_rad=collective_rad,
        )

    def solve_for_thrust(self, target_thrust_N: float, rpm: float,
                         rho: float, V_inf: float = 0.0,
                         pitch_bounds: tuple[float, float] = (
                             np.radians(-5), np.radians(25)),
                         ) -> RotorLoads:
        """Find collective pitch to achieve target thrust.

        Uses bisection to find the collective pitch angle that produces
        the desired thrust level.

        Parameters
        ----------
        target_thrust_N : float
            Desired thrust (N).
        rpm : float
            Rotor speed (RPM).
        rho : float
            Air density (kg/m³).
        V_inf : float
            Axial velocity (m/s). Default 0 (hover).
        pitch_bounds : tuple
            Min/max collective pitch search range (radians).

        Returns
        -------
        RotorLoads
            Loads at the collective that achieves target thrust.
        """
        lo, hi = pitch_bounds

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            loads = self.solve(rpm, V_inf, rho, collective_rad=mid)
            if loads.thrust < target_thrust_N:
                lo = mid
            else:
                hi = mid
            if abs(loads.thrust - target_thrust_N) / max(abs(target_thrust_N), 1.0) < 0.001:
                break

        return self.solve(rpm, V_inf, rho, collective_rad=0.5 * (lo + hi))

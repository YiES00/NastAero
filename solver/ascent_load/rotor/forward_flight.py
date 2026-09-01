"""Forward flight BEMT with Glauert skewed-wake correction.

Extends the axial BEMT solver for forward flight conditions where the
rotor operates in edgewise flow. Uses Glauert correction for skewed
wake and Pitt-Peters dynamic inflow model for azimuthally-averaged loads.

References
----------
- Leishman, "Principles of Helicopter Aerodynamics", Ch. 3-4
- Pitt & Peters, "Theoretical prediction of dynamic inflow derivatives",
  Vertica, 5(1), 1981
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import logging

from .blade import BladeDef
from .bemt_solver import BEMTSolver, RotorLoads

logger = logging.getLogger(__name__)
from ..config import logger


class ForwardFlightBEMT:
    """BEMT solver for forward flight with skewed wake.

    Computes rotor loads by azimuthal averaging of blade element forces
    over one revolution, including the Glauert correction for non-axial
    inflow.

    Parameters
    ----------
    blade : BladeDef
        Blade geometry definition.
    n_blades : int
        Number of blades.
    n_azimuth : int
        Number of azimuthal stations for averaging. Default 36 (10° steps).
    """

    def __init__(self, blade: BladeDef, n_blades: int = 4,
                 n_azimuth: int = 36):
        self.blade = blade
        self.n_blades = n_blades
        self.n_azimuth = n_azimuth
        self._axial_solver = BEMTSolver(blade, n_blades)

    def _glauert_inflow(self, mu: float, lambda_c: float,
                        CT: float) -> float:
        """Glauert mean inflow ratio for forward flight.

        λ_i = CT / (2 * sqrt(μ² + λ²))

        where λ = λ_c + λ_i (total inflow).

        Parameters
        ----------
        mu : float
            Advance ratio V·cos(α) / (ΩR).
        lambda_c : float
            Climb inflow ratio V·sin(α) / (ΩR).
        CT : float
            Thrust coefficient (initial estimate).

        Returns
        -------
        float
            Induced inflow ratio λ_i.
        """
        if CT < 1e-10:
            return 0.0

        # Iterative solution: λ_i = CT / (2√(μ² + (λ_c + λ_i)²))
        lambda_i = 0.05  # Initial guess
        for _ in range(50):
            denom = np.sqrt(mu ** 2 + (lambda_c + lambda_i) ** 2)
            if denom < 1e-10:
                break
            lambda_i_new = CT / (2.0 * denom)
            if abs(lambda_i_new - lambda_i) < 1e-8:
                break
            lambda_i = 0.5 * lambda_i_new + 0.5 * lambda_i

        return lambda_i

    def _pitt_peters_inflow(self, r_over_R: float, psi: float,
                            mu: float, lambda_i_mean: float,
                            chi: float) -> float:
        """Pitt-Peters first-harmonic inflow model.

        λ_i(r, ψ) = λ_0 + λ_1c · (r/R) · cos(ψ) + λ_1s · (r/R) · sin(ψ)

        The first harmonic inflow variation accounts for the skewed wake.

        Parameters
        ----------
        r_over_R : float
            Normalized radial position.
        psi : float
            Blade azimuth angle (radians). ψ=0 is downwind.
        mu : float
            Advance ratio.
        lambda_i_mean : float
            Mean induced inflow ratio.
        chi : float
            Wake skew angle (radians).

        Returns
        -------
        float
            Local induced inflow ratio at (r, ψ).
        """
        # Pitt-Peters: longitudinal variation
        kx = (15.0 * np.pi / 23.0) * np.tan(chi / 2.0)
        # Lateral variation (small for symmetric rotor)
        ky = 0.0

        lambda_i = lambda_i_mean * (1.0 + kx * r_over_R * np.cos(psi)
                                    + ky * r_over_R * np.sin(psi))
        return max(lambda_i, 0.0)

    def solve(self, rpm: float, V_inf: float, alpha_shaft: float,
              rho: float, collective_rad: float = 0.0) -> RotorLoads:
        """Solve rotor loads in forward flight.

        Parameters
        ----------
        rpm : float
            Rotor speed (RPM).
        V_inf : float
            Freestream velocity magnitude (m/s).
        alpha_shaft : float
            Shaft tilt angle relative to freestream (radians).
            alpha=π/2 is pure axial (hover), alpha=0 is edgewise.
        rho : float
            Air density (kg/m³).
        collective_rad : float
            Collective pitch offset (radians).

        Returns
        -------
        RotorLoads
            Azimuthally-averaged hub loads.
        """
        omega = rpm * 2.0 * np.pi / 60.0
        if omega < 1e-6:
            return RotorLoads()

        R = self.blade.radius
        A = np.pi * R ** 2

        # Decompose freestream into axial and in-plane components
        V_axial = V_inf * np.sin(alpha_shaft)  # Along shaft
        V_plane = V_inf * np.cos(alpha_shaft)  # Perpendicular to shaft

        mu = V_plane / (omega * R)  # Advance ratio
        lambda_c = V_axial / (omega * R)  # Climb inflow ratio

        # For low advance ratios, use axial solver directly
        if mu < 0.05:
            return self._axial_solver.solve(rpm, V_axial, rho, collective_rad)

        # Initial CT estimate from axial solver at mean inflow
        axial_result = self._axial_solver.solve(rpm, V_axial, rho, collective_rad)
        CT_est = axial_result.CT

        # CT–lambda 고정점 폐합: 블레이드 적분이 주는 CT로 Glauert 평균
        # 유입을 다시 풀어 잔차가 수렴할 때까지 반복한다. 1회 계산만
        # 하면 초기 축류 추정과 최종 블레이드 하중이 폐합되지 않아
        # 중간 전진비에서 유입이 10~20% 어긋난다(2026-08 검토 지적).
        lambda_i_mean = self._glauert_inflow(mu, lambda_c, CT_est)
        # 웜스타트: 직전 해의 유입을 초기값으로 쓰면(이분법 반복 중
        # collective만 조금씩 변함) 폐합이 1~2회에 수렴한다.
        warm = getattr(self, "_lambda_cache", None)
        if warm is not None and abs(warm[0] - mu) < 0.05:
            lambda_i_mean = warm[1]
        for _closure_it in range(12):
            chi = (np.arctan2(mu, lambda_c + lambda_i_mean)
                   if (lambda_c + lambda_i_mean) > 1e-10 else np.pi / 4)
            CT_bl = self._blade_CT(omega, R, A, rho, mu, lambda_c,
                                   lambda_i_mean, chi, V_plane,
                                   collective_rad)
            lam_new = self._glauert_inflow(mu, lambda_c, CT_bl)
            if abs(lam_new - lambda_i_mean) <= 1e-5 * max(
                    abs(lambda_i_mean), 1e-6):
                lambda_i_mean = lam_new
                CT_est = CT_bl
                break
            # 완화 고정점 (진동 방지)
            lambda_i_mean = 0.5 * (lambda_i_mean + lam_new)
            CT_est = CT_bl
        else:
            logger.debug("FF BEMT closure not converged (mu=%.3f)", mu)

        self._lambda_cache = (mu, lambda_i_mean)

        # Wake skew angle
        lambda_total = lambda_c + lambda_i_mean
        chi = np.arctan2(mu, lambda_total) if lambda_total > 1e-10 else np.pi / 4

        # Azimuthal averaging
        psi_stations = np.linspace(0, 2 * np.pi, self.n_azimuth, endpoint=False)
        r_stations = self.blade.get_stations()
        dr = self.blade.get_dr()

        total_T = 0.0
        total_Q = 0.0
        total_H = 0.0
        total_Mx = 0.0  # Roll moment
        total_My = 0.0  # Pitch moment

        for psi in psi_stations:
            for r_over_R in r_stations:
                r = r_over_R * R
                c = self.blade.chord_at(r_over_R)
                theta = self.blade.twist_at(r_over_R) + collective_rad

                # Local induced inflow with Pitt-Peters correction
                lambda_i_local = self._pitt_peters_inflow(
                    r_over_R, psi, mu, lambda_i_mean, chi)

                # Velocity components at blade element
                U_T = omega * r + V_plane * np.sin(psi)  # Tangential
                U_P = (lambda_c + lambda_i_local) * omega * R  # Perpendicular (axial inflow)
                U_R = V_plane * np.cos(psi)  # Radial (no aerodynamic effect)

                if abs(U_T) < 1e-6:
                    continue

                # Local angle of attack
                phi = np.arctan2(U_P, U_T)
                alpha = theta - phi

                # Airfoil coefficients
                cl = self.blade.airfoil.cl(alpha)
                cd = self.blade.airfoil.cd(alpha)

                # Element forces (per blade, per unit azimuth)
                W2 = U_T ** 2 + U_P ** 2
                dL = 0.5 * rho * W2 * c * cl * dr
                dD = 0.5 * rho * W2 * c * cd * dr

                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                # Thrust component (along shaft)
                dFz = dL * cos_phi - dD * sin_phi
                # In-plane component (tangential)
                dFx = dL * sin_phi + dD * cos_phi

                # Accumulate (all blades, azimuthal average)
                total_T += self.n_blades * dFz / self.n_azimuth
                total_Q += self.n_blades * dFx * r / self.n_azimuth

                # H-force: in-plane forward force
                total_H += self.n_blades * dFx * np.sin(psi) / self.n_azimuth

                # Hub moments from blade flapping
                total_Mx += self.n_blades * dFz * r * np.sin(psi) / self.n_azimuth
                total_My += self.n_blades * dFz * r * np.cos(psi) / self.n_azimuth

        total_P = total_Q * omega

        # Non-dimensionalize
        denom_T = rho * A * (omega * R) ** 2
        denom_Q = denom_T * R
        CT = total_T / denom_T if denom_T > 0 else 0.0
        CQ = total_Q / denom_Q if denom_Q > 0 else 0.0

        return RotorLoads(
            thrust=total_T,
            torque=total_Q,
            power=total_P,
            H_force=total_H,
            roll_moment=total_Mx,
            pitch_moment=total_My,
            CT=CT,
            CQ=CQ,
            CP=CQ,
            collective_rad=collective_rad,
        )

    def _blade_CT(self, omega, R, A, rho, mu, lambda_c, lambda_i_mean,
                  chi, V_plane, collective_rad):
        """폐합 반복용 블레이드 적분 추력계수 (본 해석 루프와 동일 물리).

        본 루프와 같은 국부 유입·받음각·힘 분해를 방위각×반경으로
        벡터화했다(폐합 반복이 이중 파이썬 루프면 수십 배 느려진다).
        """
        psi = np.linspace(0, 2 * np.pi, self.n_azimuth,
                          endpoint=False)[:, None]          # (na,1)
        rR = np.asarray(self.blade.get_stations())[None, :]  # (1,nr)
        dr = self.blade.get_dr()
        r = rR * R
        c = np.array([self.blade.chord_at(x) for x in rR[0]])[None, :]
        theta = (np.array([self.blade.twist_at(x) for x in rR[0]])[None, :]
                 + collective_rad)
        kx = (15.0 * np.pi / 23.0) * np.tan(chi / 2.0)
        lam_loc = np.maximum(
            lambda_i_mean * (1.0 + kx * rR * np.cos(psi)), 0.0)
        U_T = omega * r + V_plane * np.sin(psi)
        U_P = (lambda_c + lam_loc) * omega * R
        mask = np.abs(U_T) >= 1e-6
        phi = np.arctan2(U_P, U_T)
        alpha = theta - phi
        cl = self.blade.airfoil.cl_array(alpha)
        cd = self.blade.airfoil.cd_array(alpha)
        W2 = U_T ** 2 + U_P ** 2
        dL = 0.5 * rho * W2 * c * cl * dr
        dD = 0.5 * rho * W2 * c * cd * dr
        dFz = np.where(mask, dL * np.cos(phi) - dD * np.sin(phi), 0.0)
        total_T = self.n_blades * dFz.sum() / self.n_azimuth
        denom_T = rho * A * (omega * R) ** 2
        return total_T / denom_T if denom_T > 0 else 0.0

    def solve_for_thrust(self, target_thrust_N: float, rpm: float,
                         V_inf: float, alpha_shaft: float, rho: float,
                         pitch_bounds: tuple[float, float] = (
                             np.radians(-5), np.radians(25)),
                         ) -> RotorLoads:
        """Find collective pitch for target thrust in forward flight.

        Parameters
        ----------
        target_thrust_N : float
            Desired thrust (N).
        rpm : float
            Rotor speed (RPM).
        V_inf : float
            Freestream velocity (m/s).
        alpha_shaft : float
            Shaft angle (radians).
        rho : float
            Air density (kg/m³).
        pitch_bounds : tuple
            Collective pitch search range (radians).

        Returns
        -------
        RotorLoads
            Loads at collective that achieves target thrust.
        """
        lo, hi = pitch_bounds

        for _ in range(50):
            mid = 0.5 * (lo + hi)
            loads = self.solve(rpm, V_inf, alpha_shaft, rho,
                               collective_rad=mid)
            if loads.thrust < target_thrust_N:
                lo = mid
            else:
                hi = mid
            if abs(loads.thrust - target_thrust_N) / max(abs(target_thrust_N), 1.0) < 0.001:
                break

        result = self.solve(rpm, V_inf, alpha_shaft, rho,
                            collective_rad=0.5 * (lo + hi))
        result.thrust_target_N = float(target_thrust_N)
        if abs(result.thrust - target_thrust_N) > 0.02 * max(abs(target_thrust_N), 1.0):
            result.thrust_saturated = True
            logger.warning(
                "Forward-flight BEMT thrust saturated: target %.1f N, "
                "achieved %.1f N (collective %.1f deg, rpm %.0f, V %.1f m/s)",
                target_thrust_N, result.thrust,
                np.degrees(result.collective_rad), rpm, V_inf)
        return result

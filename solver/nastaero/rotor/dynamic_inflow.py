"""Dynamic inflow models for hover-gust and transient rotor analysis.

Implements the Pitt-Peters first-order dynamic inflow model for
single-rotor and aggregate multi-rotor (decoupled) response. Used for
hover-gust load case generation in the dissertation Chapter 7.

Vertical-axis dynamic inflow follows the Pitt-Peters first-order ODE;
the in-plane (lateral) gust response is treated as a quasi-steady
H-force linearization (no separate lateral inflow ODE) using the
momentum-flux coefficient ``ρ A ν_steady`` (m_dot · v_rel). This is
consistent with first-order rotor lateral derivatives at hover and
remains decoupled across rotors (alpha_ij = 0).

References
----------
- Pitt, D. M. & Peters, D. A. (1981). "Theoretical Prediction of
  Dynamic-Inflow Derivatives." Vertica, 5(1), 21-34.
- Johnson, W. (2013). Rotorcraft Aeromechanics. Cambridge Univ. Press.
- Leishman, J. G. (2006). Principles of Helicopter Aerodynamics, 2nd ed.
  (Chapter 4: Rotor in-plane forces — H-force derivation).
- Yeo, H. & Johnson, W. (2022). "Comprehensive Analysis of Rotorcraft
  and Distributed Electric Propulsion Vehicle Inflow Models." AHS Forum.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


# ============================================================
# Single-rotor Pitt-Peters dynamic inflow (1st-order ODE)
# ============================================================

@dataclass
class PittPetersInflow:
    """Pitt-Peters first-order dynamic inflow model for a single rotor.

    Solves the 1st-order ODE:
        tau_nu * d(nu)/dt + nu(t) = nu_eff(t)

    where:
        nu(t) is the time-varying induced velocity
        nu_eff(t) = nu_steady - w_g(t) - z_dot(t)
        nu_steady = sqrt(T / (2 rho A))   (steady momentum theory)
        w_g(t) is the vertical gust velocity at the rotor disc,
            POSITIVE DOWNWARD: a down-gust augments the flow through
            the disc exactly as a climb does, so both w_g and z_dot
            reduce the induced-velocity requirement
        z_dot(t) is the body vertical velocity (positive = climbing)

    Time constant tau_nu = [4/(3*pi)] * R / nu_steady (Pitt-Peters 1981).

    유도: 원판의 겉보기 질량 m_a = (8/3) rho R^3 을 동결 질량유량
    추력 기울기 dT/d(nu) = 2 rho A U (A = pi R^2) 로 나눈다.
        tau = (8/3) rho R^3 / (2 rho pi R^2 U) = 4/(3 pi) * R / U
            = 0.4244 * R / U
    무차원 Pitt-Peters로도 같다. M11 = 8/(3 pi) 를 호버 유입류 이득
    L^-1_11 = 2 lambda_h 로 나누면 무차원 시상수 0.4244/lambda_h 이고,
    lambda_h = nu_s/(Omega R) 을 대입하면 0.4244 R/nu_s 가 된다.
    (v1.2까지 쓰이던 0.85는 무차원 계수 8/(3 pi)를 차원 겉보기 질량으로
    혼동한 값으로, 정확히 2배 과대였다.)

    계수 선택에 관한 주의 (2026-08 검토 반영). 위 8/(3 pi)는
    Pitt-Peters (1981) Table 3의 '비보정(uncorrected)' 균일 유입
    겉보기 질량이다. 같은 표는 압력분포를 보정한 값
    M11 = 128/(75 pi) (tau = 64/(75 pi) R/U = 0.2716 R/U)도 제시하며
    이후의 권장 해석행렬은 보정값을 쓴다. 본 모델은 불투과 원판
    겉보기 질량과 동결 질량유량 기울기라는 자기일관한 1차
    유도를 따르는 것이므로 비보정 계수를 채택하되, 두 계수는
    모델 불확실성 범위로 함께 보고한다(ILC-8 기준 tau 13.7 ms 대
    8.8 ms; 돌풍 주기 T_g = 0.5 s 대비 어느 쪽도 tau << T_g라
    최대하중 민감도는 2% 미만 — 논문 3 민감도 절 참조).

    모델 정체성. 이 클래스는 표준 Pitt-Peters 유한상태 유입
    모형(하중계수 C가 입력, 유도유동 lambda가 상태)의 완전 구현이
    아니라, 그 겉보기 질량 계수를 쓰는 1차 지연 돌풍
    대체모형(apparent-mass gust surrogate)이다. 상태 강제는
    nu_eff = nu_s - w_g - z_dot 로 이루어지고 추력 증분은 상태로부터
    역산된다. 전진비행 확장도 wake-skew 결합 L 행렬이 아니라
    유효속도 U = sqrt(V^2 + nu^2)의 스칼라 축약이다.

    Parameters
    ----------
    rotor_radius : float
        Rotor radius (m).
    rho : float
        Air density (kg/m^3). Default 1.225 (sea level ISA).
    T_steady : float
        Steady-state thrust (N) used to compute nu_steady.

    Attributes
    ----------
    A : float
        Disc area (m^2).
    nu_steady : float
        Steady induced velocity (m/s).
    tau_nu : float
        Inflow time constant (s).
    """
    rotor_radius: float
    rho: float = 1.225
    T_steady: float = 0.0
    V_forward: float = 0.0

    def __post_init__(self):
        self.A = np.pi * self.rotor_radius ** 2
        if self.T_steady > 0:
            # 호버 씨드에서 전진비행 운동량 이론의 음함수
            #   T = 2 rho A U nu,  U = sqrt(V_f^2 + nu^2)
            # 를 고정점 반복으로 푼다 (동결 질량유량 선형화 관례:
            # V_f = 0이면 U = nu_steady로 기존 호버식과 정확히 일치).
            nu = np.sqrt(self.T_steady / (2.0 * self.rho * self.A))
            for _ in range(50):
                U = np.sqrt(self.V_forward ** 2 + nu ** 2)
                nu_new = self.T_steady / (2.0 * self.rho * self.A * U)
                if abs(nu_new - nu) < 1e-12:
                    nu = nu_new
                    break
                nu = 0.5 * (nu + nu_new)
            self.nu_steady = float(nu)
        else:
            self.nu_steady = 0.0
        # 질량 유량 파라미터 — 시간상수와 추력 감도의 공통 스케일
        self.mass_flow = float(np.sqrt(self.V_forward ** 2
                                       + self.nu_steady ** 2))
        if self.mass_flow > 0:
            self.tau_nu = (4.0 / (3.0 * np.pi)) * self.rotor_radius / self.mass_flow
        else:
            # Fallback for trivial / numerical cases
            self.tau_nu = 0.05  # 50 ms typical

    @property
    def thrust_slope(self) -> float:
        """추력-유입류 감도 dT/d(nu) = 2 rho A U (동결 질량유량).

        V_forward = 0에서 2 rho A nu_steady로 기존 호버 선형화와
        정확히 일치한다."""
        return 2.0 * self.rho * self.A * self.mass_flow

    def derivative(self, nu: float, w_g: float, z_dot: float = 0.0) -> float:
        """RHS of the inflow ODE: d(nu)/dt = (nu_eff - nu) / tau_nu.

        Parameters
        ----------
        nu : float
            Current inflow value (m/s).
        w_g : float
            Vertical gust velocity (m/s, positive = DOWNWARD gust).
        z_dot : float
            Body vertical velocity (m/s, positive = climbing).

        Returns
        -------
        float
            d(nu)/dt (m/s^2).
        """
        nu_eff = self.nu_steady - w_g - z_dot
        return (nu_eff - nu) / self.tau_nu

    def thrust_perturbation(self, nu: float) -> float:
        """Compute thrust perturbation Delta_T from inflow perturbation.

        Linearized around steady operating point (frozen mass flow):
            Delta_T = 2 * rho * A * U * (nu - nu_steady)
        where U = sqrt(V_forward^2 + nu_steady^2); at hover
        (V_forward = 0) this reduces to the classical
        2 * rho * A * nu_steady * Delta_nu.

        Parameters
        ----------
        nu : float
            Current inflow (m/s).

        Returns
        -------
        float
            Delta_T (N). Positive = thrust increase.
        """
        return self.thrust_slope * (nu - self.nu_steady)

    def lateral_H_force(self, v_rel: float) -> float:
        """Quasi-steady in-plane H-force from lateral relative wind.

        Momentum-flux derivation: the mass flow through the hovering
        disc is m_dot = rho * A * nu_steady. Turning that stream tube
        by the lateral relative wind v_rel = v_gust - y_dot_body
        imparts the in-plane momentum flux

            H = m_dot * v_rel = rho * A * nu_steady * v_rel

        This is a first-order momentum bound; blade-element H-force
        contributions (profile drag tilt, cyclic flapping) are not
        modelled. The lateral component is treated as quasi-steady
        (no separate lateral inflow ODE).

        Sign convention: positive v_rel (e.g. wind blowing in +Y at the
        disc) generates positive H (force on rotor in +Y, reaction on
        body in +Y as well).

        Parameters
        ----------
        v_rel : float
            Lateral relative wind (m/s) at the rotor disc,
            v_rel = v_gust_at_disc - y_dot_body.

        Returns
        -------
        float
            H-force (N).
        """
        return self.rho * self.A * self.nu_steady * v_rel


# ============================================================
# Multi-rotor aggregate response (decoupled, alpha_ij = 0)
# ============================================================

def _evaluate_gust(gust_func, t: float, x: float) -> Tuple[float, float]:
    """Normalize a gust callable to always return (w_g, v_g) m/s.

    The gust callable can be one of:

    - ``lambda t, x: w_g``                — vertical-only, returns scalar
    - ``lambda t, x: (w_g, v_g)``         — vertical + lateral
    - ``lambda t, x: {"w": ..., "v": ...}`` — keyword form

    This helper makes vertical-only gust callables backward-compatible
    with the lateral-aware path.
    """
    out = gust_func(t, x)
    if isinstance(out, tuple):
        if len(out) == 2:
            return float(out[0]), float(out[1])
        if len(out) == 1:
            return float(out[0]), 0.0
        raise ValueError(f"gust_func tuple has unexpected length {len(out)}")
    if isinstance(out, dict):
        return float(out.get("w", 0.0)), float(out.get("v", 0.0))
    # Scalar => vertical only
    return float(out), 0.0


@dataclass
class MultiRotorAggregate:
    """Multi-rotor aggregate inflow / gust response (decoupled).

    Each rotor's vertical inflow evolves independently via the
    Pitt-Peters first-order ODE; lateral gust response is modelled as
    a quasi-steady H-force linearization (no separate lateral inflow
    ODE), consistent with the first-order momentum-theory coefficient
    ``ρ A ν_steady``. Inflow coupling between rotors (alpha_ij != 0)
    is intentionally left as future work and is not modelled here.

    The aggregate response captures:

      1. Each rotor's independent vertical inflow ODE
      2. Body vertical and (optionally) lateral acceleration feedback
      3. Time-delayed gust arrival at each rotor
         (Delta_t_i = x_i / V_wind)
      4. Per-rotor lateral H-force from quasi-steady gust + body Y/dot
         (only when ``enable_lateral=True``)

    State vector
    ------------
    - Vertical-only (default, ``enable_lateral=False``):
        y = [nu_1, ..., nu_N, z, z_dot]                  length N + 2
    - Vertical + lateral (``enable_lateral=True``):
        y = [nu_1, ..., nu_N, z, z_dot, y_body, y_dot]   length N + 4

    Parameters
    ----------
    rotors : list of PittPetersInflow
        One Pitt-Peters model per rotor (typically 6 or 8).
    rotor_x_positions : np.ndarray
        Longitudinal position of each rotor (m). Used for time delay
        of the vertical gust component.
    body_mass : float
        Total body mass (kg) for body Z (and optionally Y) dynamics.
    enable_lateral : bool, optional
        When True, the state includes body lateral position and rate,
        and lateral gust components from the gust callable are applied
        as H-forces on the body. Default False (back-compat).
    rotor_y_positions : np.ndarray, optional
        Lateral position of each rotor (m). Required only for
        diagnostic moment computation; not used by the body Y ODE.
    V_tail_attribution : float, optional
        Fraction (0..1) of aggregate H-force assumed to be transmitted
        to the vertical tail. Configuration-dependent; default 0.30
        for the NASA L+C / GACOMP class layouts in the dissertation.
    V_tail_arm : float, optional
        Vertical arm (m) from rotor plane to V-tail aerodynamic centre
        for the diagnostic moment ``V_tail_attribution * H_total *
        V_tail_arm``. Default 1.5 m.
    """
    rotors: list  # List[PittPetersInflow]
    rotor_x_positions: np.ndarray
    body_mass: float
    enable_lateral: bool = False
    rotor_y_positions: Optional[np.ndarray] = None
    V_tail_attribution: float = 0.30
    V_tail_arm: float = 1.5

    def __post_init__(self):
        self.n_rotors = len(self.rotors)
        if len(self.rotor_x_positions) != self.n_rotors:
            raise ValueError(
                f"rotor_x_positions length ({len(self.rotor_x_positions)}) "
                f"!= n_rotors ({self.n_rotors})"
            )
        if self.rotor_y_positions is not None:
            if len(self.rotor_y_positions) != self.n_rotors:
                raise ValueError(
                    f"rotor_y_positions length "
                    f"({len(self.rotor_y_positions)}) != n_rotors "
                    f"({self.n_rotors})"
                )

    def state_dim(self) -> int:
        """Total state vector dimension.

        - Vertical only: ``n_rotors + 2``  ([nu_i, z, z_dot])
        - Lateral on:    ``n_rotors + 4``  ([nu_i, z, z_dot, y, y_dot])
        """
        return self.n_rotors + (4 if self.enable_lateral else 2)

    # ── Index helpers (keep state layout self-documenting) ──
    @property
    def _idx_z(self) -> int:
        return self.n_rotors

    @property
    def _idx_z_dot(self) -> int:
        return self.n_rotors + 1

    @property
    def _idx_y(self) -> int:
        if not self.enable_lateral:
            raise AttributeError("Lateral state disabled.")
        return self.n_rotors + 2

    @property
    def _idx_y_dot(self) -> int:
        if not self.enable_lateral:
            raise AttributeError("Lateral state disabled.")
        return self.n_rotors + 3

    def derivative(
        self,
        y: np.ndarray,
        t: float,
        gust_func: Callable[[float, float], float],
        V_wind: float = 10.0,
        g: float = 9.81,
    ) -> np.ndarray:
        """RHS of the coupled body-rotor ODE.

        See class docstring for the state-vector layout. The
        ``gust_func`` may return either a scalar ``w_g`` (vertical
        gust only, back-compat) or a ``(w_g, v_g)`` tuple
        (vertical + lateral); :func:`_evaluate_gust` normalizes both
        forms.

        Parameters
        ----------
        y : np.ndarray
            Current state vector (length :meth:`state_dim`).
        t : float
            Current time (s).
        gust_func : callable
            ``gust_func(t, x_position)`` returning either ``w_g`` (m/s)
            or ``(w_g, v_g)`` for vertical + lateral gust.
        V_wind : float
            Wind speed for time-delay computation (m/s). Default 10.
        g : float
            Gravity (m/s^2). Reserved for future use.

        Returns
        -------
        np.ndarray
            d(y)/dt vector (length :meth:`state_dim`).
        """
        if len(y) != self.state_dim():
            raise ValueError(f"State dim mismatch: {len(y)} != {self.state_dim()}")

        dydt = np.zeros_like(y)
        z_dot = y[self._idx_z_dot]
        y_dot = y[self._idx_y_dot] if self.enable_lateral else 0.0

        delta_T_total = 0.0
        H_total = 0.0

        for i, rotor in enumerate(self.rotors):
            x_i = self.rotor_x_positions[i]

            # Time-delayed gust arrival (Δt = x/V_wind is significant
            # for UAM, ~0.1-1 s, not O(1 ms))
            delta_t_i = x_i / V_wind
            t_eff = t - delta_t_i
            if t_eff >= 0.0:
                w_g_i, v_g_i = _evaluate_gust(gust_func, t_eff, x_i)
            else:
                w_g_i, v_g_i = 0.0, 0.0

            # ── Vertical inflow ODE ──
            dydt[i] = rotor.derivative(y[i], w_g_i, z_dot)
            delta_T_total += rotor.thrust_perturbation(y[i])

            # ── Lateral quasi-steady H-force ──
            if self.enable_lateral:
                # Relative lateral wind at the disc: gust minus body
                # lateral velocity. Body lateral motion subtracts from
                # the relative wind exactly as z_dot does for vertical.
                v_rel = v_g_i - y_dot
                H_total += rotor.lateral_H_force(v_rel)

        # ── Body Z dynamics (always) ──
        dydt[self._idx_z] = z_dot
        dydt[self._idx_z_dot] = delta_T_total / self.body_mass

        # ── Body Y dynamics (lateral path) ──
        if self.enable_lateral:
            dydt[self._idx_y] = y_dot
            dydt[self._idx_y_dot] = H_total / self.body_mass

        return dydt

    def initial_state(self) -> np.ndarray:
        """Initial state: each rotor at nu_steady, body at rest."""
        y0 = np.zeros(self.state_dim())
        for i, rotor in enumerate(self.rotors):
            y0[i] = rotor.nu_steady
        return y0

    # ────────────────────────────────────────────────────────
    # Diagnostics — H-force, V-tail loads (post-processing)
    # ────────────────────────────────────────────────────────

    def lateral_H_force_total(
        self,
        gust_func: Callable[[float, float], float],
        t: float,
        y_dot_body: float = 0.0,
        V_wind: float = 10.0,
    ) -> float:
        """Aggregate lateral H-force at time ``t`` (post-processing).

        Sums ``rotor.lateral_H_force(v_g - y_dot_body)`` over all
        rotors, applying the per-rotor gust time delay ``x_i / V_wind``.
        Useful for off-line analysis without integrating the full ODE.

        Returns
        -------
        float
            Aggregate H-force (N).
        """
        H = 0.0
        for i, rotor in enumerate(self.rotors):
            x_i = self.rotor_x_positions[i]
            t_eff = t - x_i / V_wind
            if t_eff < 0.0:
                continue
            _, v_g_i = _evaluate_gust(gust_func, t_eff, x_i)
            H += rotor.lateral_H_force(v_g_i - y_dot_body)
        return H

    def V_tail_lateral_moment(
        self,
        gust_func: Callable[[float, float], float],
        t: float,
        y_dot_body: float = 0.0,
        V_wind: float = 10.0,
    ) -> Tuple[float, float]:
        """V-tail lateral load and moment from aggregate H-force.

        Applies the configurable attribution
        ``F_VT = V_tail_attribution * H_total`` and the V-tail arm
        ``V_tail_arm``. These coefficients are configuration-dependent
        — for the NASA L+C / GACOMP class layouts used in the
        dissertation, the defaults ``0.30`` and ``1.5 m`` reflect a
        nominal rear-mounted V-tail; OEMs should override.

        Returns
        -------
        (F_VT, M_VT) : tuple of float
            Lateral force (N) and bending moment (N·m) attributed to
            the V-tail.
        """
        H = self.lateral_H_force_total(gust_func, t, y_dot_body, V_wind)
        F_VT = self.V_tail_attribution * H
        M_VT = F_VT * self.V_tail_arm
        return F_VT, M_VT


# ============================================================
# Standard gust profiles (1-cosine, step, etc.)
# ============================================================

def make_one_cosine_gust(
    U_de: float,
    T_g: float,
    direction: str = "vertical",
) -> Callable[[float, float], float]:
    """Create a 1-cosine discrete gust profile.

    w_g(t) = (U_de / 2) * [1 - cos(2*pi*t / T_g)]   for 0 <= t <= T_g
           = 0                                       otherwise

    Standard FAR 25 gust: U_de = 7.62 m/s (25 fps reduced) at design cruise.

    Parameters
    ----------
    U_de : float
        Derived gust velocity (m/s). Sign convention: + = downward (인플로 ODE가 소비하는 부호).
    T_g : float
        Gust period (s). Typical 0.5 s for sharp UAM hover gust.
    direction : str
        "vertical" or "lateral" (informational only; the ODE uses w_g).

    Returns
    -------
    callable
        gust_func(t, x) -> w_g (m/s).
    """
    def gust_func(t: float, x: float) -> float:
        if 0.0 <= t <= T_g:
            return (U_de / 2.0) * (1.0 - np.cos(2.0 * np.pi * t / T_g))
        return 0.0

    return gust_func


def make_step_gust(U_de: float, t_onset: float = 0.0) -> Callable[[float, float], float]:
    """Step gust: w_g = U_de for t >= t_onset, else 0."""
    def gust_func(t: float, x: float) -> float:
        return U_de if t >= t_onset else 0.0
    return gust_func


def make_one_cosine_gust_3d(
    U_de: float,
    V_de: float,
    T_g: float,
) -> Callable[[float, float], Tuple[float, float]]:
    """Create a 1-cosine discrete gust with vertical + lateral components.

    Both components share the same shape and period; only their
    amplitudes differ:

        w_g(t) = (U_de / 2) * (1 - cos(2π t / T_g))   (vertical)
        v_g(t) = (V_de / 2) * (1 - cos(2π t / T_g))   (lateral)

    for ``0 <= t <= T_g`` and zero otherwise. Either amplitude can be
    set to 0 to disable that direction.

    Parameters
    ----------
    U_de : float
        Vertical derived gust velocity (m/s, + = downward — 인플로 ODE 규약).
    V_de : float
        Lateral derived gust velocity (m/s, + = +Y direction).
    T_g : float
        Gust period (s). Default 0.5 s for sharp UAM hover gust.

    Returns
    -------
    callable
        ``gust_func(t, x) -> (w_g, v_g)``. Consumed directly by
        :class:`MultiRotorAggregate` when ``enable_lateral=True``.
    """
    two_pi = 2.0 * np.pi

    def gust_func(t: float, x: float) -> Tuple[float, float]:
        if 0.0 <= t <= T_g:
            half_minus_cos = 0.5 * (1.0 - np.cos(two_pi * t / T_g))
            return U_de * half_minus_cos, V_de * half_minus_cos
        return 0.0, 0.0

    return gust_func


# ============================================================
# Test stub (runs as smoke test)
# ============================================================

def _smoke_test():
    """Quick smoke test for the module."""
    rotor = PittPetersInflow(rotor_radius=0.75, T_steady=2330.0)
    print(f"Pitt-Peters single rotor:")
    print(f"  nu_steady = {rotor.nu_steady:.2f} m/s")
    print(f"  tau_nu    = {rotor.tau_nu*1000:.1f} ms")

    # 1-cosine gust U_de=7.62 m/s, T_g=0.5 s
    gust = make_one_cosine_gust(U_de=7.62, T_g=0.5)
    print(f"\n1-cosine gust at t=0.25s, x=0: {gust(0.25, 0.0):.3f} m/s")

    # Aggregate model with 8 rotors
    rotors = [PittPetersInflow(rotor_radius=0.75, T_steady=2330.0) for _ in range(8)]
    x_pos = np.array([3.0, 6.0, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0])
    agg = MultiRotorAggregate(rotors=rotors, rotor_x_positions=x_pos,
                               body_mass=1900.0)
    y0 = agg.initial_state()
    print(f"\n8-rotor aggregate:")
    print(f"  state dim   = {agg.state_dim()}")
    print(f"  initial nu  = {y0[0]:.2f} m/s")

    # Verify time-delay magnitude (the v0.1 error correction)
    V_wind = 10.0
    delta_t_max = max(x_pos) / V_wind
    print(f"\n  Delta_t at V_wind={V_wind} m/s, x_max={max(x_pos)} m:")
    print(f"    Delta_t = {delta_t_max:.2f} s   (NOT O(1 ms) as v0.1 v0.2 erroneously stated)")

    # ── Lateral integration smoke test ──
    print("\n8-rotor aggregate WITH lateral gust (integrated):")
    agg_lat = MultiRotorAggregate(
        rotors=rotors,
        rotor_x_positions=x_pos,
        body_mass=1900.0,
        enable_lateral=True,
    )
    print(f"  state dim   = {agg_lat.state_dim()}  (expected {8 + 4})")
    g3d = make_one_cosine_gust_3d(U_de=0.0, V_de=7.62, T_g=0.5)
    # Scan H_total over time to find the aggregate peak (after delays)
    V_wind = 10.0
    t_grid = np.linspace(0.0, 1.5, 301)
    H_curve = np.array([agg_lat.lateral_H_force_total(g3d, t=ti, V_wind=V_wind)
                         for ti in t_grid])
    t_peak = t_grid[int(np.argmax(np.abs(H_curve)))]
    H_peak = agg_lat.lateral_H_force_total(g3d, t=t_peak, V_wind=V_wind)
    F_VT, M_VT = agg_lat.V_tail_lateral_moment(g3d, t=t_peak, V_wind=V_wind)
    print(f"  Peak H_total at t={t_peak:.3f}s for V_de=7.62 m/s, V_wind={V_wind}:")
    print(f"    H_total   = {H_peak/1000:.3f} kN")
    print(f"    F_VT      = {F_VT/1000:.3f} kN  (attribution {agg_lat.V_tail_attribution:.2f})")
    print(f"    M_VT      = {M_VT/1000:.3f} kN·m (arm {agg_lat.V_tail_arm:.1f} m)")


if __name__ == "__main__":
    _smoke_test()

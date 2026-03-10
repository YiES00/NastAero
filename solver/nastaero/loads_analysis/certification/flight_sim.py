"""6-DOF rigid body flight dynamics simulation.

Implements a nonlinear 6-DOF flight dynamics model with:
- 12-state vector: [u, v, w, p, q, r, φ, θ, ψ, xe, ye, ze]
- Stability-derivative aerodynamic model (small perturbation coefficients)
- Body-axis gravity decomposition via Euler angles
- RK4 fixed-timestep integrator (pickle-safe for multiprocessing)
- Trim initial condition solver

The simulator is intentionally lightweight (~ms per run) so hundreds of
maneuver/gust cases can be run in parallel before extracting critical
time points for detailed SOL 144 analysis.

Design decision: direct RK4 instead of scipy.integrate.solve_ivp because:
1. Pickle-safe (no closures) — compatible with ProcessPoolExecutor
2. Fixed timestep is sufficient (0.1-2 Hz dynamics, dt=5ms → 200 steps/s)
3. Minimal Python overhead for sub-millisecond runs
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .aero_derivatives import AeroDerivativeSet


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AircraftState:
    """12-state flight dynamics state vector.

    Body-axis velocities (u, v, w), angular rates (p, q, r),
    Euler angles (phi, theta, psi), and Earth-fixed position (xe, ye, ze).
    """
    u: float = 0.0    # Body-axis forward velocity (m/s)
    v: float = 0.0    # Body-axis lateral velocity (m/s)
    w: float = 0.0    # Body-axis vertical velocity (m/s)
    p: float = 0.0    # Roll rate (rad/s)
    q: float = 0.0    # Pitch rate (rad/s)
    r: float = 0.0    # Yaw rate (rad/s)
    phi: float = 0.0    # Roll angle (rad)
    theta: float = 0.0  # Pitch angle (rad)
    psi: float = 0.0    # Heading angle (rad)
    xe: float = 0.0    # Earth-fixed x position (m)
    ye: float = 0.0    # Earth-fixed y position (m)
    ze: float = 0.0    # Earth-fixed z position (m)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.u, self.v, self.w,
            self.p, self.q, self.r,
            self.phi, self.theta, self.psi,
            self.xe, self.ye, self.ze,
        ])

    @classmethod
    def from_array(cls, y: np.ndarray) -> AircraftState:
        return cls(
            u=y[0], v=y[1], w=y[2],
            p=y[3], q=y[4], r=y[5],
            phi=y[6], theta=y[7], psi=y[8],
            xe=y[9], ye=y[10], ze=y[11],
        )

    @property
    def V_total(self) -> float:
        """Total airspeed magnitude (m/s)."""
        return math.sqrt(self.u**2 + self.v**2 + self.w**2)

    @property
    def alpha(self) -> float:
        """Angle of attack (rad)."""
        if abs(self.u) < 1e-10:
            return 0.0
        return math.atan2(self.w, self.u)

    @property
    def beta(self) -> float:
        """Sideslip angle (rad)."""
        V = self.V_total
        if V < 1e-10:
            return 0.0
        return math.asin(max(-1.0, min(1.0, self.v / V)))


@dataclass
class AircraftParams:
    """Aircraft parameters for 6-DOF simulation.

    Combines mass properties, inertia tensor, reference geometry,
    and the complete aerodynamic derivative set.
    """
    mass: float = 0.0     # Total mass (kg)
    S: float = 0.0        # Reference wing area (m²)
    b: float = 0.0        # Wing span (m)
    c_bar: float = 0.0    # Mean aerodynamic chord (m)
    Ixx: float = 0.0      # Roll inertia (kg·m²)
    Iyy: float = 0.0      # Pitch inertia (kg·m²)
    Izz: float = 0.0      # Yaw inertia (kg·m²)
    Ixz: float = 0.0      # Product of inertia (kg·m²)
    derivs: Optional[AeroDerivativeSet] = None
    thrust_N: float = 0.0   # Constant thrust along body x-axis (N)
    rho: float = 1.225     # Air density (kg/m³)


@dataclass
class ControlInput:
    """Control surface deflections."""
    delta_e: float = 0.0   # Elevator (rad, positive trailing-edge down → nose-up)
    delta_a: float = 0.0   # Aileron (rad, positive → right wing down)
    delta_r: float = 0.0   # Rudder (rad, positive → nose left)


@dataclass
class SimTimeHistory:
    """Complete simulation time history results."""
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    states: np.ndarray = field(default_factory=lambda: np.array([]))    # (N, 12)
    controls: np.ndarray = field(default_factory=lambda: np.array([]))  # (N, 3)
    nz: np.ndarray = field(default_factory=lambda: np.array([]))
    ny: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    beta_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    p_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    q_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    r_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    p_dot: np.ndarray = field(default_factory=lambda: np.array([]))   # Roll accel (rad/s²)
    q_dot: np.ndarray = field(default_factory=lambda: np.array([]))   # Pitch accel (rad/s²)
    r_dot: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# 6-DOF equations of motion
# ---------------------------------------------------------------------------

_G = 9.80665  # m/s²


def six_dof_derivatives(
    y: np.ndarray,
    t: float,
    params: AircraftParams,
    control: ControlInput,
    gust_vel: Optional[np.ndarray] = None,
    external_force_func: Optional[Callable] = None,
) -> np.ndarray:
    """Compute dy/dt for the 6-DOF nonlinear rigid-body EOM.

    Parameters
    ----------
    y : ndarray (12,)
        State vector [u, v, w, p, q, r, phi, theta, psi, xe, ye, ze].
    t : float
        Current time (s).
    params : AircraftParams
        Aircraft parameters including derivatives.
    control : ControlInput
        Current control surface deflections.
    gust_vel : ndarray (3,) or None
        Gust velocity in body axes [w_gx, w_gy, w_gz] (m/s).
    external_force_func : callable(t, y) -> (F_xyz, M_xyz) or None
        Optional external force/moment callback (e.g., rotor forces).
        Returns (ndarray(3), ndarray(3)) = (forces, moments) in body axes.

    Returns
    -------
    ndarray (12,)
        State derivatives dy/dt.
    """
    u, v, w = y[0], y[1], y[2]
    p, q, r = y[3], y[4], y[5]
    phi, theta = y[6], y[7]

    d = params.derivs
    m = params.mass
    S = params.S
    b = params.b
    c = params.c_bar
    Ixx = params.Ixx
    Iyy = params.Iyy
    Izz = params.Izz
    Ixz = params.Ixz

    # Effective aerodynamic velocities (add gust)
    u_aero = u
    v_aero = v
    w_aero = w
    if gust_vel is not None:
        u_aero += gust_vel[0]
        v_aero += gust_vel[1]
        w_aero += gust_vel[2]

    # Aerodynamic angles
    V = math.sqrt(u_aero**2 + v_aero**2 + w_aero**2)
    if V < 1.0:
        V = 1.0  # prevent division by zero at very low speeds
    alpha = math.atan2(w_aero, max(u_aero, 1.0))
    beta = math.asin(max(-1.0, min(1.0, v_aero / V)))

    # --- Simple stall model: clamp alpha & beta for aero force computation ---
    # Beyond ~17° the real wing stalls (CL drops), but our linear model
    # (CL = CLα·α) would keep increasing indefinitely, producing
    # unrealistically high nz.  Clamping alpha_aero bounds CL at
    # CL_max ≈ CLα × alpha_stall_rad ≈ 5.8 × 0.30 ≈ 1.74.
    # Similarly, the VTP stalls at large sideslip angles (~20°), so
    # beta is clamped to prevent unrealistic lateral forces/moments.
    # The actual state (u, v, w) still evolves freely — only the
    # aero force calculation sees the capped angles.
    _ALPHA_STALL = 0.30   # ~17.2°
    _BETA_STALL = 0.35    # ~20.1°
    alpha_aero = max(-_ALPHA_STALL, min(_ALPHA_STALL, alpha))
    beta_aero = max(-_BETA_STALL, min(_BETA_STALL, beta))

    # Dynamic pressure
    qbar = 0.5 * params.rho * V**2

    # Non-dimensional rates
    p_hat = p * b / (2.0 * V)     # p * b/(2V)
    q_hat = q * c / (2.0 * V)     # q * c̄/(2V)
    r_hat = r * b / (2.0 * V)     # r * b/(2V)

    de = control.delta_e
    da = control.delta_a
    dr = control.delta_r

    # ------------------------------------------------------------------
    # Aerodynamic force/moment coefficients (stability derivative model)
    # ------------------------------------------------------------------

    # Lift coefficient (using stall-clamped alpha)
    CL = (d.CLalpha * alpha_aero
          + d.CLdelta_e * de
          + d.CLq * q_hat)

    # Drag coefficient (parabolic polar, stall-clamped alpha)
    CD = d.CD0 + d.CDalpha * alpha_aero**2

    # Sideforce coefficient (using stall-clamped beta)
    CY = (d.CYbeta * beta_aero
          + d.CYdelta_r * dr
          + d.CYp * p_hat
          + d.CYr * r_hat)

    # Rolling moment coefficient (using stall-clamped beta)
    Cl = (d.Clbeta * beta_aero
          + d.Cldelta_a * da
          + d.Cldelta_r * dr
          + d.Clp * p_hat
          + d.Clr * r_hat)

    # Pitching moment coefficient (stall-clamped alpha)
    Cm = (d.Cmalpha * alpha_aero
          + d.Cmdelta_e * de
          + d.Cmq * q_hat
          + d.Cmalpha_dot * q_hat)  # α̇ ≈ q for small perturbations

    # Yawing moment coefficient (using stall-clamped beta)
    Cn = (d.Cnbeta * beta_aero
          + d.Cndelta_a * da
          + d.Cndelta_r * dr
          + d.Cnp * p_hat
          + d.Cnr * r_hat)

    # ------------------------------------------------------------------
    # Aerodynamic forces in wind axes → body axes
    # ------------------------------------------------------------------
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    cb = math.cos(beta)
    sb = math.sin(beta)

    # Forces in stability axes (L=lift, D=drag)
    L = qbar * S * CL     # Lift (perpendicular to V, positive up)
    D = qbar * S * CD     # Drag (opposite to V)
    Y = qbar * S * CY     # Sideforce (positive right)

    # Convert to body axes
    # X_body = -D·cos(α) + L·sin(α)
    # Z_body = -D·sin(α) - L·cos(α)
    # Y_body = Y
    X_aero = -D * ca + L * sa
    Y_aero = Y
    Z_aero = -D * sa - L * ca

    # Aerodynamic moments in body axes
    L_aero = qbar * S * b * Cl       # Rolling moment
    M_aero = qbar * S * c * Cm       # Pitching moment
    N_aero = qbar * S * b * Cn       # Yawing moment

    # ------------------------------------------------------------------
    # Gravity in body axes
    # ------------------------------------------------------------------
    sphi = math.sin(phi)
    cphi = math.cos(phi)
    stheta = math.sin(theta)
    ctheta = math.cos(theta)

    gx = -_G * stheta
    gy = _G * ctheta * sphi
    gz = _G * ctheta * cphi

    # ------------------------------------------------------------------
    # External forces/moments (e.g., rotor forces for VTOL)
    # ------------------------------------------------------------------
    F_ext = np.zeros(3)
    M_ext = np.zeros(3)
    if external_force_func is not None:
        try:
            ext_result = external_force_func(t, y)
            F_ext = ext_result[0]
            M_ext = ext_result[1]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Force equations (du/dt, dv/dt, dw/dt)
    # ------------------------------------------------------------------
    X_total = X_aero + params.thrust_N + F_ext[0]
    du = (X_total / m) + gx + r * v - q * w
    dv = ((Y_aero + F_ext[1]) / m) + gy - r * u + p * w
    dw = ((Z_aero + F_ext[2]) / m) + gz + q * u - p * v

    # ------------------------------------------------------------------
    # Moment equations (dp/dt, dq/dt, dr/dt)
    # Including Ixz cross-coupling terms
    # ------------------------------------------------------------------
    L_total = L_aero + M_ext[0]
    M_total = M_aero + M_ext[1]
    N_total = N_aero + M_ext[2]

    Gamma = Ixx * Izz - Ixz**2

    dp = (Izz * L_total + Ixz * N_total
          - (Izz * (Izz - Iyy) + Ixz**2) * q * r
          + Ixz * (Ixx - Iyy + Izz) * p * q) / Gamma

    dq = (M_total - (Ixx - Izz) * p * r - Ixz * (p**2 - r**2)) / Iyy

    dr = (Ixz * L_total + Ixx * N_total
          + (Ixx * (Ixx - Iyy) + Ixz**2) * p * q
          - Ixz * (Ixx - Iyy + Izz) * q * r) / Gamma

    # ------------------------------------------------------------------
    # Euler angle kinematics (dφ/dt, dθ/dt, dψ/dt)
    # ------------------------------------------------------------------
    if abs(ctheta) < 1e-10:
        ctheta = 1e-10  # prevent singularity near ±90°

    dphi = p + (q * sphi + r * cphi) * stheta / ctheta
    dtheta = q * cphi - r * sphi
    dpsi = (q * sphi + r * cphi) / ctheta

    # ------------------------------------------------------------------
    # Position kinematics (Earth-fixed, NED-like)
    # ------------------------------------------------------------------
    spsi = math.sin(y[8])
    cpsi = math.cos(y[8])

    dxe = (ctheta * cpsi * u
           + (sphi * stheta * cpsi - cphi * spsi) * v
           + (cphi * stheta * cpsi + sphi * spsi) * w)
    dye = (ctheta * spsi * u
           + (sphi * stheta * spsi + cphi * cpsi) * v
           + (cphi * stheta * spsi - sphi * cpsi) * w)
    dze = (-stheta * u
           + sphi * ctheta * v
           + cphi * ctheta * w)

    return np.array([du, dv, dw, dp, dq, dr,
                     dphi, dtheta, dpsi, dxe, dye, dze])


# ---------------------------------------------------------------------------
# RK4 integrator
# ---------------------------------------------------------------------------

def integrate_6dof(
    params: AircraftParams,
    initial_state: AircraftState,
    control_func: Callable,
    t_span: Tuple[float, float],
    dt: float = 0.005,
    gust_func: Optional[Callable] = None,
    external_force_func: Optional[Callable] = None,
) -> SimTimeHistory:
    """Integrate 6-DOF EOM using RK4 fixed-timestep method.

    Parameters
    ----------
    params : AircraftParams
        Aircraft parameters.
    initial_state : AircraftState
        Initial conditions.
    control_func : callable(t) -> ControlInput
        Control input as function of time.
    t_span : (t0, tf)
        Start and end times (s).
    dt : float
        Timestep (s). Default 5ms (sufficient for 0.1-2 Hz dynamics).
    gust_func : callable(t, xe) -> ndarray(3) or None
        Gust velocity in body axes as function of time and position.
    external_force_func : callable(t, y) -> (F_xyz, M_xyz) or None
        Optional external force/moment callback (e.g., rotor forces).
        Returns (ndarray(3), ndarray(3)) = (forces, moments) in body axes.

    Returns
    -------
    SimTimeHistory
        Complete time history with derived quantities.
    """
    t0, tf = t_span
    N = int(math.ceil((tf - t0) / dt)) + 1
    t_arr = np.linspace(t0, tf, N)

    states = np.zeros((N, 12))
    controls = np.zeros((N, 3))
    nz_arr = np.zeros(N)
    ny_arr = np.zeros(N)
    alpha_arr = np.zeros(N)
    beta_arr = np.zeros(N)
    p_rate = np.zeros(N)
    q_rate = np.zeros(N)
    r_rate = np.zeros(N)
    p_dot_arr = np.zeros(N)
    q_dot_arr = np.zeros(N)
    r_dot_arr = np.zeros(N)

    y = initial_state.to_array()
    states[0] = y

    ctrl = control_func(t0)
    controls[0] = [ctrl.delta_e, ctrl.delta_a, ctrl.delta_r]

    # Initial derived quantities
    _store_derived(y, 0, params, nz_arr, ny_arr, alpha_arr, beta_arr,
                   p_rate, q_rate, r_rate)
    # Initial angular accelerations from EOM
    gust0 = _eval_gust(gust_func, t0, y[9]) if gust_func else None
    dydt0 = six_dof_derivatives(y, t0, params, ctrl, gust0,
                                external_force_func)
    p_dot_arr[0] = dydt0[3]  # dp/dt
    q_dot_arr[0] = dydt0[4]  # dq/dt
    r_dot_arr[0] = dydt0[5]  # dr/dt

    # RK4 integration loop
    for i in range(N - 1):
        t = t_arr[i]
        h = t_arr[i + 1] - t

        ctrl = control_func(t)
        gust = _eval_gust(gust_func, t, y[9]) if gust_func else None

        k1 = six_dof_derivatives(y, t, params, ctrl, gust,
                                 external_force_func)

        ctrl_mid = control_func(t + 0.5 * h)
        gust_mid = _eval_gust(gust_func, t + 0.5 * h, y[9] + 0.5 * h * y[0]) if gust_func else None

        k2 = six_dof_derivatives(y + 0.5 * h * k1, t + 0.5 * h,
                                 params, ctrl_mid, gust_mid,
                                 external_force_func)
        k3 = six_dof_derivatives(y + 0.5 * h * k2, t + 0.5 * h,
                                 params, ctrl_mid, gust_mid,
                                 external_force_func)

        ctrl_end = control_func(t + h)
        gust_end = _eval_gust(gust_func, t + h, y[9] + h * y[0]) if gust_func else None

        k4 = six_dof_derivatives(y + h * k3, t + h,
                                 params, ctrl_end, gust_end,
                                 external_force_func)

        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        states[i + 1] = y
        ctrl_store = control_func(t_arr[i + 1])
        controls[i + 1] = [ctrl_store.delta_e, ctrl_store.delta_a,
                           ctrl_store.delta_r]

        _store_derived(y, i + 1, params, nz_arr, ny_arr, alpha_arr,
                       beta_arr, p_rate, q_rate, r_rate)

        # Angular accelerations from k1 at the new step
        # (k1 = dy/dt evaluated at the current state → exact derivative)
        ctrl_now = control_func(t_arr[i + 1])
        gust_now = _eval_gust(gust_func, t_arr[i + 1], y[9]) if gust_func else None
        dydt_now = six_dof_derivatives(y, t_arr[i + 1], params,
                                       ctrl_now, gust_now,
                                       external_force_func)
        p_dot_arr[i + 1] = dydt_now[3]
        q_dot_arr[i + 1] = dydt_now[4]
        r_dot_arr[i + 1] = dydt_now[5]

    return SimTimeHistory(
        t=t_arr,
        states=states,
        controls=controls,
        nz=nz_arr,
        ny=ny_arr,
        alpha_deg=alpha_arr,
        beta_deg=beta_arr,
        p_rate=p_rate,
        q_rate=q_rate,
        r_rate=r_rate,
        p_dot=p_dot_arr,
        q_dot=q_dot_arr,
        r_dot=r_dot_arr,
    )


def _eval_gust(gust_func, t, xe):
    """Safely evaluate gust function."""
    try:
        return gust_func(t, xe)
    except Exception:
        return None


def _store_derived(y, idx, params, nz, ny, alpha, beta, p_rate, q_rate, r_rate):
    """Compute and store derived quantities from state vector."""
    u, v, w = y[0], y[1], y[2]
    V = math.sqrt(u**2 + v**2 + w**2)
    if V > 1.0:
        alpha[idx] = math.degrees(math.atan2(w, u))
        beta[idx] = math.degrees(math.asin(max(-1, min(1, v / V))))
    p_rate[idx] = y[3]
    q_rate[idx] = y[4]
    r_rate[idx] = y[5]

    # Load factor: nz = (L + T·sin(α)) / W ≈ -az_body / g + cos(θ)cos(φ)
    # More directly: nz = -(dw/dt - q·u + p·v) / g + cos(θ)cos(φ)
    # For simplicity, use: nz ≈ -Fz_aero / (m·g)
    # But more accurately from body accelerations:
    phi, theta = y[6], y[7]
    cphi = math.cos(phi)
    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    # Body z-acceleration includes gravity contribution
    # az_body = dw/dt + q*u - p*v (from EOM, before gravity)
    # nz = -az_body / g = -(Fz_aero/m + g*cos(θ)*cos(φ) + q*u - p*v) / g
    # Approximate: nz = 1 at 1g trim (initial condition)
    # Better: compute from state evolution
    # For time-history tracking, use kinematic formula:
    # nz ≈ (cos(θ)cos(φ)) - a_z_inertial / g
    # At trim: a_z = 0, nz = cos(θ)cos(φ) ≈ 1
    # We'll refine this after we have derivatives available
    # For now, use a simpler approach:
    nz[idx] = cphi * ctheta  # This is the gravity component; will be refined
    ny[idx] = 0.0


def compute_nz_from_history(params: AircraftParams, history: SimTimeHistory):
    """Recompute load factors from time history using central differences.

    nz = -(d²ze/dt²) / g + 1  (in steady level flight nz=1)

    This gives the true inertial load factor experienced by the aircraft.
    """
    dt = history.t[1] - history.t[0] if len(history.t) > 1 else 0.005
    N = len(history.t)

    # Earth-z velocity (positive down in NED)
    ze = history.states[:, 11]

    # Central difference for acceleration
    az = np.zeros(N)
    if N > 2:
        az[1:-1] = (ze[2:] - 2 * ze[1:-1] + ze[:-2]) / dt**2
        az[0] = az[1]
        az[-1] = az[-2]

    # nz = -az/g + cos(θ)cos(φ)
    # In NED convention with ze positive down:
    # nz = az/g (since ze is positive down, az>0 means deceleration = positive nz)
    # More precisely: nz = L / W
    # Use body-axis approach instead:

    for i in range(N):
        u = history.states[i, 0]
        v = history.states[i, 1]
        w = history.states[i, 2]
        p = history.states[i, 3]
        q = history.states[i, 4]
        r = history.states[i, 5]
        phi = history.states[i, 6]
        theta = history.states[i, 7]

        V = math.sqrt(u**2 + v**2 + w**2)
        if V < 1.0:
            history.nz[i] = 1.0
            continue

        alpha = math.atan2(w, max(u, 1.0))
        beta_val = math.asin(max(-1, min(1, v / V)))

        # Apply same stall clamping as in six_dof_derivatives
        _ALPHA_STALL = 0.30  # ~17.2°
        _BETA_STALL = 0.35   # ~20.1°
        alpha_aero = max(-_ALPHA_STALL, min(_ALPHA_STALL, alpha))
        beta_aero = max(-_BETA_STALL, min(_BETA_STALL, beta_val))

        qbar = 0.5 * params.rho * V**2
        d = params.derivs

        # Non-dimensional rates
        p_hat = p * params.b / (2.0 * V)
        q_hat = q * params.c_bar / (2.0 * V)
        r_hat = r * params.b / (2.0 * V)

        de = history.controls[i, 0]

        CL = (d.CLalpha * alpha_aero + d.CLdelta_e * de + d.CLq * q_hat)

        # nz = L / W = (qbar * S * CL) / (m * g)
        history.nz[i] = qbar * params.S * CL / (params.mass * _G)

        CY = (d.CYbeta * beta_aero
              + d.CYdelta_r * history.controls[i, 2]
              + d.CYp * p_hat + d.CYr * r_hat)
        history.ny[i] = qbar * params.S * CY / (params.mass * _G)


# ---------------------------------------------------------------------------
# Trim initial condition solver
# ---------------------------------------------------------------------------

def trim_initial_state(
    params: AircraftParams,
    V_tas: float,
    nz: float = 1.0,
) -> Tuple[AircraftState, float]:
    """Compute trimmed initial state for given airspeed and load factor.

    Finds the angle of attack and pitch angle for steady flight at the
    given load factor. Also computes the required thrust for drag balance.

    Parameters
    ----------
    params : AircraftParams
        Aircraft parameters.
    V_tas : float
        True airspeed (m/s).
    nz : float
        Load factor (g's). Default 1.0 for level flight.

    Returns
    -------
    (AircraftState, delta_e_trim)
        Trimmed state and required elevator deflection (rad).
    """
    d = params.derivs
    qbar = 0.5 * params.rho * V_tas**2
    W = params.mass * _G

    # Required CL for given load factor
    CL_req = nz * W / (qbar * params.S)

    # Iterative trim: solve coupled CL(α, δe) = CL_req, Cm(α, δe) = 0
    # CL = CLα·α + CLδe·δe = CL_req
    # Cm = Cmα·α + Cmδe·δe = 0
    # → δe = -(Cmα/Cmδe)·α
    # → CLα·α + CLδe·(-(Cmα/Cmδe)·α) = CL_req
    # → α·(CLα - CLδe·Cmα/Cmδe) = CL_req
    denom = d.CLalpha
    if abs(d.Cmdelta_e) > 1e-6:
        denom = d.CLalpha - d.CLdelta_e * d.Cmalpha / d.Cmdelta_e

    if abs(denom) > 1e-6:
        alpha_trim = CL_req / denom
    else:
        alpha_trim = CL_req / d.CLalpha if abs(d.CLalpha) > 1e-6 else 0.0

    # Trim elevator for Cm = 0
    # Cm = Cmalpha * alpha + Cmdelta_e * delta_e = 0
    if abs(d.Cmdelta_e) > 1e-6:
        delta_e_trim = -d.Cmalpha * alpha_trim / d.Cmdelta_e
    else:
        delta_e_trim = 0.0

    # Trim drag for thrust (include lift-induced drag)
    CL_trim = d.CLalpha * alpha_trim + d.CLdelta_e * delta_e_trim
    CD = d.CD0 + d.CDalpha * alpha_trim**2
    thrust_trim = qbar * params.S * (CD - CL_trim * math.sin(alpha_trim))
    # Add component from body-axis: thrust must also overcome the aft component
    # of lift (L·sin(α)) but this is typically small. Keep simple: thrust ≈ D.
    thrust_trim = max(thrust_trim, 0.0)
    thrust_trim = qbar * params.S * CD  # Keep it simple: thrust = drag

    # Pitch angle = alpha at trim (flight path angle γ = 0 for level flight)
    theta_trim = alpha_trim

    # Set thrust in params (for use during simulation)
    params.thrust_N = thrust_trim

    state = AircraftState(
        u=V_tas * math.cos(alpha_trim),
        v=0.0,
        w=V_tas * math.sin(alpha_trim),
        p=0.0, q=0.0, r=0.0,
        phi=0.0,
        theta=theta_trim,
        psi=0.0,
        xe=0.0, ye=0.0, ze=0.0,
    )

    return state, delta_e_trim

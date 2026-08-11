"""Unit tests for dynamic_inflow module.

Verifies:
1. Pitt-Peters momentum theory consistency (ν = √(T/(2ρA)))
2. Time constant scaling τ_ν = 0.85 r / ν
3. Thrust perturbation linearity
4. Multi-rotor aggregate state dimension
5. Body acceleration feedback
6. Gust time-delay magnitude (Δt = x / V_wind)
7. 1-cosine gust profile correctness
"""
from __future__ import annotations

import numpy as np
import pytest

from nastaero.rotor.dynamic_inflow import (
    PittPetersInflow,
    MultiRotorAggregate,
    make_one_cosine_gust,
    make_step_gust,
    make_one_cosine_gust_3d,
    _evaluate_gust,
)


# ============================================================
# 1. Single rotor momentum theory consistency
# ============================================================

class TestPittPetersMomentumTheory:
    """Verify ν = √(T/(2ρA)) and τ_ν = 0.85 r / ν."""

    def test_inflow_NASA_LC_lift_rotor(self):
        """NASA L+C / GACOMP lift rotor: T=2330 N, r=0.75 m → ν=23.2 m/s."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
        # Manual momentum theory
        A = np.pi * 0.75 ** 2
        nu_expected = np.sqrt(2330.0 / (2.0 * 1.225 * A))
        assert nu_expected == pytest.approx(23.2, rel=0.005)
        assert rotor.nu_steady == pytest.approx(nu_expected, rel=1e-9)

    def test_time_constant_scaling(self):
        """τ_ν = 0.85 · r / ν (Pitt-Peters 1981)."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
        tau_expected = 0.85 * 0.75 / rotor.nu_steady
        assert rotor.tau_nu == pytest.approx(tau_expected, rel=1e-9)
        # Numerical magnitude check (~27.5 ms for GACOMP rotor)
        assert rotor.tau_nu == pytest.approx(0.0275, rel=0.05)

    def test_radius_consistency_for_small_inflow(self):
        """If ν=16.5 m/s claimed, what radius would actually give it?"""
        # ν = √(T/(2ρπr²)) → r = √(T/(2ρπν²))
        # For T=2330 N, ρ=1.225, ν_target=16.5 m/s:
        T, rho, nu = 2330.0, 1.225, 16.5
        r_required = np.sqrt(T / (2 * rho * np.pi * nu ** 2))
        # Should be about 1.05 m, NOT 0.75 m
        assert r_required == pytest.approx(1.05, rel=0.02)
        # This documents that v0.5.0 had a math error in Ch7 for r=0.75m

    def test_thrust_perturbation_linearity(self):
        """Δ_T = 2ρA·ν_steady·Δ_ν."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
        A = np.pi * 0.75 ** 2
        delta_nu = 1.0  # m/s perturbation
        delta_T_expected = 2.0 * 1.225 * A * rotor.nu_steady * delta_nu
        delta_T_computed = rotor.thrust_perturbation(rotor.nu_steady + delta_nu)
        assert delta_T_computed == pytest.approx(delta_T_expected, rel=1e-9)

    def test_zero_thrust_fallback(self):
        """T=0 should give a fallback τ_ν, not divide-by-zero."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=0.0)
        assert rotor.nu_steady == 0.0
        assert rotor.tau_nu > 0.0
        assert rotor.tau_nu < 1.0  # reasonable fallback


# ============================================================
# 2. ODE derivative correctness
# ============================================================

class TestPittPetersODE:

    def test_steady_state_zero_derivative(self):
        """At ν = ν_eff, dν/dt should be 0."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
        # No gust, no body motion: ν_eff = ν_steady
        d_nu = rotor.derivative(nu=rotor.nu_steady, w_g=0.0, z_dot=0.0)
        assert abs(d_nu) < 1e-9

    def test_downward_gust_decreases_inflow(self):
        """w_g > 0 (downward gust, augments inflow like a climb)
        → ν_eff < ν_steady → dν/dt < 0 at steady ν."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
        d_nu = rotor.derivative(nu=rotor.nu_steady, w_g=5.0, z_dot=0.0)
        assert d_nu < 0

    def test_body_descent_increases_effective_inflow(self):
        """z_dot > 0 (body moving up) means rotor sees more downward flow,
        decreasing effective ν_eff. Convention: ν_eff = ν_steady - w_g - z_dot."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
        d_nu = rotor.derivative(nu=rotor.nu_steady, w_g=0.0, z_dot=2.0)
        assert d_nu < 0


# ============================================================
# 3. Multi-rotor aggregate response
# ============================================================

class TestMultiRotorAggregate:

    def setup_method(self):
        self.rotors = [
            PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
            for _ in range(8)
        ]
        self.x_pos = np.array([3.0, 6.0, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0])
        self.agg = MultiRotorAggregate(
            rotors=self.rotors, rotor_x_positions=self.x_pos, body_mass=1900.0
        )

    def test_state_dim_n_rotors_plus_2(self):
        """State = n_rotors inflows + (z, z_dot)."""
        assert self.agg.state_dim() == 8 + 2

    def test_initial_state_is_steady_inflow(self):
        y0 = self.agg.initial_state()
        for i, rotor in enumerate(self.rotors):
            assert y0[i] == pytest.approx(rotor.nu_steady)
        # Body z and z_dot should be 0
        assert y0[8] == 0.0
        assert y0[9] == 0.0

    def test_no_gust_no_body_motion_zero_derivative(self):
        """At steady state with no gust, dy/dt = 0."""
        y0 = self.agg.initial_state()
        no_gust = lambda t, x: 0.0
        dydt = self.agg.derivative(y0, t=0.5, gust_func=no_gust, V_wind=10.0)
        # All inflow derivatives = 0
        assert np.allclose(dydt[:8], 0.0, atol=1e-9)
        # z_dot = 0 (input was at rest)
        assert dydt[8] == 0.0
        # z_double_dot = 0 (no thrust perturbation)
        assert abs(dydt[9]) < 1e-9

    def test_uniform_upward_gust_uniform_response(self):
        """All rotors at same x see the same gust → uniform inflow change."""
        # Override x_pos so all rotors see gust at same time
        agg = MultiRotorAggregate(
            rotors=self.rotors,
            rotor_x_positions=np.zeros(8),  # all at x=0
            body_mass=1900.0,
        )
        y0 = agg.initial_state()
        const_gust = lambda t, x: 5.0  # constant 5 m/s downward gust
        dydt = agg.derivative(y0, t=1.0, gust_func=const_gust, V_wind=10.0)
        # All inflow derivatives should be equal
        assert np.allclose(dydt[:8], dydt[0])
        # All should be negative (gust reduces effective inflow)
        assert dydt[0] < 0

    def test_body_feedback_consistency(self):
        """z_double_dot should be sum(Delta_T) / mass."""
        # Apply nu_perturbation to all rotors
        y = self.agg.initial_state()
        delta_nu = 1.0
        for i in range(8):
            y[i] += delta_nu

        # Expected total Delta_T
        delta_T_per_rotor = self.rotors[0].thrust_perturbation(
            self.rotors[0].nu_steady + delta_nu
        )
        delta_T_total = 8 * delta_T_per_rotor
        z_ddot_expected = delta_T_total / 1900.0

        no_gust = lambda t, x: 0.0
        dydt = self.agg.derivative(y, t=0.5, gust_func=no_gust, V_wind=10.0)
        assert dydt[9] == pytest.approx(z_ddot_expected, rel=1e-6)

    def test_state_dim_mismatch_raises(self):
        """Wrong-size state vector should raise ValueError."""
        bad_y = np.zeros(5)  # too small
        no_gust = lambda t, x: 0.0
        with pytest.raises(ValueError):
            self.agg.derivative(bad_y, t=0.0, gust_func=no_gust)


# ============================================================
# 4. Gust time delay magnitude (the v0.5.0 error correction)
# ============================================================

class TestGustTimeDelay:
    """Verify Δt = x / V_wind is on the order of seconds, not milliseconds."""

    def test_time_delay_is_seconds_not_ms(self):
        """For UAM-typical x=3-6m and V_wind=10 m/s, Δt should be 0.3-0.6 s."""
        x_separation = 6.0  # meters
        V_wind = 10.0  # m/s
        dt = x_separation / V_wind
        # NOT O(1 ms) as v0.1 erroneously stated; it's O(0.1-1 s)
        assert dt >= 0.3
        assert dt <= 1.5

    def test_time_delay_v_wind_inversely_proportional(self):
        """Δt should scale as 1/V_wind."""
        x = 5.0
        dt_low = x / 5.0
        dt_high = x / 15.0
        assert dt_low > dt_high
        assert dt_low / dt_high == pytest.approx(3.0, rel=1e-9)


# ============================================================
# 5. Gust profile shapes
# ============================================================

class TestGustProfiles:

    def test_one_cosine_gust_shape(self):
        """1-cosine gust: w_g(t) = (Ude/2)·(1 - cos(2π·t/T_g)) for t in [0, T_g]."""
        Ude, T_g = 7.62, 0.5
        gust = make_one_cosine_gust(Ude, T_g)
        # At t=0: w_g = 0
        assert gust(0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
        # At t = T_g/2: w_g = Ude (peak)
        assert gust(T_g / 2.0, 0.0) == pytest.approx(Ude, rel=1e-9)
        # At t = T_g: w_g = 0
        assert gust(T_g, 0.0) == pytest.approx(0.0, abs=1e-9)
        # Outside [0, T_g]: w_g = 0
        assert gust(-0.1, 0.0) == 0.0
        assert gust(T_g + 0.1, 0.0) == 0.0

    def test_step_gust_onset(self):
        """Step gust: w_g = U_de for t >= t_onset, else 0."""
        gust = make_step_gust(U_de=5.0, t_onset=0.2)
        assert gust(0.0, 0.0) == 0.0
        assert gust(0.1, 0.0) == 0.0
        assert gust(0.2, 0.0) == 5.0
        assert gust(1.0, 0.0) == 5.0


# ============================================================
# 8. Lateral gust integration — H-force, body Y dynamics
# ============================================================

class TestLateralHForce:
    """Quasi-steady H-force linearization: H = 2 ρ A ν_steady · v_rel."""

    def test_zero_lateral_wind_gives_zero_H(self):
        rotor = PittPetersInflow(rotor_radius=0.75, T_steady=2330.0)
        assert rotor.lateral_H_force(0.0) == 0.0

    def test_H_force_coefficient_matches_momentum_theory(self):
        """Per-rotor H = m_dot * v_rel = ρ A ν_steady · v_rel."""
        rotor = PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
        A = np.pi * 0.75 ** 2
        coeff = 1.225 * A * rotor.nu_steady
        for v in [-5.0, -1.0, 0.5, 7.62, 12.0]:
            assert rotor.lateral_H_force(v) == pytest.approx(coeff * v, rel=1e-9)

    def test_H_force_linearity(self):
        """H is linear in v_rel (no quadratic terms)."""
        rotor = PittPetersInflow(rotor_radius=0.75, T_steady=2330.0)
        for k in [-3.0, -0.5, 2.0, 7.62]:
            assert rotor.lateral_H_force(2.0 * k) == pytest.approx(
                2.0 * rotor.lateral_H_force(k), rel=1e-9
            )

    def test_H_force_sign_convention(self):
        """+v_rel → +H (force on rotor in +Y)."""
        rotor = PittPetersInflow(rotor_radius=0.75, T_steady=2330.0)
        assert rotor.lateral_H_force(+1.0) > 0
        assert rotor.lateral_H_force(-1.0) < 0


class TestEvaluateGust:
    """The gust normalizer must accept scalar / tuple / dict forms."""

    def test_scalar_gust_becomes_vertical_only(self):
        gust = lambda t, x: 1.5
        w, v = _evaluate_gust(gust, 0.0, 0.0)
        assert (w, v) == (1.5, 0.0)

    def test_tuple_gust_unpacks(self):
        gust = lambda t, x: (3.0, -2.0)
        w, v = _evaluate_gust(gust, 0.0, 0.0)
        assert (w, v) == (3.0, -2.0)

    def test_dict_gust_unpacks(self):
        gust = lambda t, x: {"w": 4.0, "v": 6.0}
        w, v = _evaluate_gust(gust, 0.0, 0.0)
        assert (w, v) == (4.0, 6.0)


class TestOneCosineGust3D:
    """3D 1-cosine gust profile."""

    def test_zero_at_endpoints(self):
        g = make_one_cosine_gust_3d(U_de=7.62, V_de=5.0, T_g=0.5)
        for t in [0.0, 0.5]:
            w, v = g(t, 0.0)
            assert w == pytest.approx(0.0, abs=1e-9)
            assert v == pytest.approx(0.0, abs=1e-9)

    def test_peak_at_half_period(self):
        g = make_one_cosine_gust_3d(U_de=7.62, V_de=5.0, T_g=0.5)
        w, v = g(0.25, 0.0)
        assert w == pytest.approx(7.62, rel=1e-9)
        assert v == pytest.approx(5.0, rel=1e-9)

    def test_outside_window_returns_zero(self):
        g = make_one_cosine_gust_3d(U_de=1.0, V_de=2.0, T_g=0.5)
        for t in [-0.1, 0.6, 5.0]:
            w, v = g(t, 0.0)
            assert (w, v) == (0.0, 0.0)


class TestMultiRotorAggregateLateral:
    """Lateral path: state dim, body Y ODE, H-force aggregation, V-tail."""

    def _make_8(self, enable_lateral=False, **kw):
        rotors = [PittPetersInflow(rotor_radius=0.75, rho=1.225, T_steady=2330.0)
                  for _ in range(8)]
        x_pos = np.array([3.0, 6.0, 3.0, 6.0, 3.0, 6.0, 3.0, 6.0])
        return MultiRotorAggregate(
            rotors=rotors,
            rotor_x_positions=x_pos,
            body_mass=1900.0,
            enable_lateral=enable_lateral,
            **kw,
        )

    def test_state_dim_vertical_only(self):
        agg = self._make_8(enable_lateral=False)
        assert agg.state_dim() == 8 + 2

    def test_state_dim_with_lateral(self):
        agg = self._make_8(enable_lateral=True)
        assert agg.state_dim() == 8 + 4

    def test_back_compat_scalar_gust_still_works(self):
        """Vertical-only path must accept scalar gust callable unchanged."""
        agg = self._make_8(enable_lateral=False)
        gust = make_one_cosine_gust(U_de=7.62, T_g=0.5)
        y0 = agg.initial_state()
        dydt = agg.derivative(y0, t=0.9, gust_func=gust, V_wind=10.0)
        assert dydt.shape == (10,)
        # Some derivative should be non-zero once gust has propagated
        assert np.any(np.abs(dydt[:8]) > 1e-6)

    def test_lateral_only_gust_keeps_inflow_steady(self):
        """Pure lateral gust → vertical inflow stays at nu_steady."""
        agg = self._make_8(enable_lateral=True)
        gust = make_one_cosine_gust_3d(U_de=0.0, V_de=7.62, T_g=0.5)
        y0 = agg.initial_state()
        dydt = agg.derivative(y0, t=0.9, gust_func=gust, V_wind=10.0)
        # Inflows are at nu_steady, vertical gust is zero, z_dot=0 → dν/dt = 0
        assert np.allclose(dydt[:8], 0.0, atol=1e-9)
        # But body Y must have non-zero acceleration
        assert abs(dydt[agg._idx_y_dot]) > 1e-3

    def test_lateral_H_force_total_matches_per_rotor_sum(self):
        """H_total at the synchronized gust peak ≈ 8 × per-rotor H."""
        # Use V_wind so large that delays are negligible relative to T_g
        agg = self._make_8(enable_lateral=True)
        gust = make_one_cosine_gust_3d(U_de=0.0, V_de=7.62, T_g=0.5)
        # Pick t at gust peak with negligible delay (V_wind=1e6)
        H_total = agg.lateral_H_force_total(gust, t=0.25, V_wind=1.0e6)
        per_rotor = agg.rotors[0].lateral_H_force(7.62)
        assert H_total == pytest.approx(8.0 * per_rotor, rel=1e-9)

    def test_V_tail_moment_uses_attribution_and_arm(self):
        """F_VT = α H_total, M_VT = F_VT · arm."""
        agg = self._make_8(enable_lateral=True,
                            V_tail_attribution=0.4, V_tail_arm=1.2)
        gust = make_one_cosine_gust_3d(U_de=0.0, V_de=7.62, T_g=0.5)
        H_total = agg.lateral_H_force_total(gust, t=0.25, V_wind=1.0e6)
        F_VT, M_VT = agg.V_tail_lateral_moment(gust, t=0.25, V_wind=1.0e6)
        assert F_VT == pytest.approx(0.4 * H_total, rel=1e-9)
        assert M_VT == pytest.approx(F_VT * 1.2, rel=1e-9)

    def test_body_y_dot_subtracts_from_relative_wind(self):
        """If body lateral velocity equals gust, relative wind = 0 → H = 0."""
        agg = self._make_8(enable_lateral=True)
        gust = make_one_cosine_gust_3d(U_de=0.0, V_de=7.62, T_g=0.5)
        # State: nu_steady on inflows, body z=0, z_dot=0, y=0, y_dot = 7.62
        y0 = agg.initial_state()
        y0[agg._idx_y_dot] = 7.62
        # At t such that gust at every rotor = 7.62 (use huge V_wind)
        # Force evaluate gust at t=0.25, V_wind=1e6 -> all rotors see 7.62
        # The ODE applies the same V_wind; cannot inject 1e6 here without
        # also using it on the derivative call. Use derivative() directly
        # with V_wind=1e6.
        dydt = agg.derivative(y0, t=0.25, gust_func=gust, V_wind=1.0e6)
        # H_total -> 0 -> body y_dotdot -> 0
        assert dydt[agg._idx_y_dot] == pytest.approx(0.0, abs=1e-6)

    def test_initial_state_zero_in_body(self):
        """All body components start at zero regardless of lateral flag."""
        agg_v = self._make_8(enable_lateral=False)
        agg_l = self._make_8(enable_lateral=True)
        for agg in (agg_v, agg_l):
            y0 = agg.initial_state()
            assert np.all(y0[agg.n_rotors:] == 0.0)
            assert np.allclose(y0[:agg.n_rotors], agg.rotors[0].nu_steady, rtol=1e-9)

    def test_initial_state_length_matches_state_dim(self):
        for flag in (False, True):
            agg = self._make_8(enable_lateral=flag)
            assert agg.initial_state().shape == (agg.state_dim(),)

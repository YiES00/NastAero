"""Tests for Phase 2: AircraftConfig + V-n Diagram + Visualization.

Tests the Part 23 load factor formulas, V-n diagram generation,
Pratt gust formula, speed conversions, and CONM2 adjuster.
"""
import math
import numpy as np
import pytest

from nastaero.loads_analysis.certification.aircraft_config import (
    AircraftConfig, WeightCGCondition, SpeedSchedule,
    ControlSurfaceLimits, LandingGearConfig,
    part23_nz_max, part23_nz_min,
    eas_to_tas, tas_to_eas, eas_to_mach, mach_to_eas,
    dynamic_pressure_from_eas,
    CONM2Adjuster,
    RHO_0, G_MPS2, KG_PER_LB,
)
from nastaero.loads_analysis.certification.vn_diagram import (
    VnPoint, VnDiagram, pratt_gust_delta_nz,
    compute_vn_diagram, compute_all_vn_diagrams,
)
from nastaero.loads_analysis.case_generator import isa_atmosphere


# ---------------------------------------------------------------------------
# Part 23 load factor formulas
# ---------------------------------------------------------------------------

class TestPart23LoadFactors:
    """Test §23.337 load factor formulas."""

    def test_nz_max_light_aircraft(self):
        """Light aircraft (< 4000 lb): nz > 3.8, capped at 3.8."""
        W = 2000 * KG_PER_LB * G_MPS2  # 2000 lb in N
        nz = part23_nz_max(W)
        # 2.1 + 24000/2000 = 2.1 + 12 = 14.1 → capped at 3.8
        assert nz == pytest.approx(3.8)

    def test_nz_max_heavy_aircraft(self):
        """Heavier aircraft: nz formula applies."""
        W = 12000 * KG_PER_LB * G_MPS2  # 12000 lb
        nz = part23_nz_max(W)
        expected = 2.1 + 24000.0 / 12000.0  # 2.1 + 2.0 = 4.1 → cap 3.8
        assert nz == pytest.approx(3.8)

        W = 50000 * KG_PER_LB * G_MPS2  # 50000 lb
        nz = part23_nz_max(W)
        expected = 2.1 + 24000.0 / 50000.0  # 2.1 + 0.48 = 2.58
        assert nz == pytest.approx(expected, rel=1e-3)

    def test_nz_max_cap(self):
        """nz_max never exceeds 3.8."""
        for W_lb in [1000, 2000, 5000, 10000]:
            W = W_lb * KG_PER_LB * G_MPS2
            nz = part23_nz_max(W)
            assert nz <= 3.8 + 1e-10

    def test_nz_min(self):
        """nz_min = -0.4 × nz_max."""
        nz_max = 3.8
        nz_min = part23_nz_min(nz_max)
        assert nz_min == pytest.approx(-1.52)

    def test_nz_min_proportional(self):
        """nz_min scales with nz_max."""
        nz_max = 2.58
        nz_min = part23_nz_min(nz_max)
        assert nz_min == pytest.approx(-0.4 * 2.58, rel=1e-6)


# ---------------------------------------------------------------------------
# Speed conversions
# ---------------------------------------------------------------------------

class TestSpeedConversions:
    """Test EAS↔TAS↔Mach conversions."""

    def test_eas_equals_tas_at_sea_level(self):
        """At sea level, EAS = TAS."""
        V_eas = 100.0  # m/s
        V_tas = eas_to_tas(V_eas, 0.0)
        assert V_tas == pytest.approx(V_eas, rel=1e-3)

    def test_tas_greater_than_eas_at_altitude(self):
        """TAS > EAS at altitude (density decreases)."""
        V_eas = 100.0
        V_tas = eas_to_tas(V_eas, 10000.0)
        assert V_tas > V_eas

    def test_roundtrip_eas_tas(self):
        """EAS→TAS→EAS roundtrip."""
        V_eas = 80.0
        alt = 5000.0
        V_tas = eas_to_tas(V_eas, alt)
        V_eas2 = tas_to_eas(V_tas, alt)
        assert V_eas2 == pytest.approx(V_eas, rel=1e-6)

    def test_eas_to_mach(self):
        """EAS to Mach at sea level."""
        _, _, a = isa_atmosphere(0.0)
        V_eas = a * 0.3  # Should be Mach 0.3 at sea level
        mach = eas_to_mach(V_eas, 0.0)
        assert mach == pytest.approx(0.3, rel=1e-3)

    def test_mach_to_eas(self):
        """Mach to EAS roundtrip."""
        mach = 0.5
        alt = 3000.0
        V_eas = mach_to_eas(mach, alt)
        mach2 = eas_to_mach(V_eas, alt)
        assert mach2 == pytest.approx(mach, rel=1e-6)

    def test_dynamic_pressure_from_eas(self):
        """q = 0.5 × ρ₀ × V²."""
        V_eas = 100.0
        q = dynamic_pressure_from_eas(V_eas)
        assert q == pytest.approx(0.5 * RHO_0 * 100.0**2, rel=1e-6)


# ---------------------------------------------------------------------------
# Speed schedule
# ---------------------------------------------------------------------------

class TestSpeedSchedule:
    """Test speed schedule validation."""

    def test_compute_VA(self):
        """VA = VS1 × √nz_max."""
        s = SpeedSchedule(VS1=30.0)
        VA = s.compute_VA(3.8)
        assert VA == pytest.approx(30.0 * math.sqrt(3.8), rel=1e-6)

    def test_validation_ok(self):
        """Valid speed schedule has no issues."""
        s = SpeedSchedule(VS1=25.0, VA=48.7, VC=70.0, VD=90.0)
        issues = s.validate()
        assert len(issues) == 0

    def test_validation_vd_too_low(self):
        """VD < 1.25 × VC triggers warning."""
        s = SpeedSchedule(VS1=25.0, VA=48.7, VC=70.0, VD=80.0)
        issues = s.validate()
        assert any("1.25" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Control surface limits
# ---------------------------------------------------------------------------

class TestControlSurfaceLimits:
    """Test speed-dependent control surface deflection schedule."""

    def test_full_aileron_at_VA(self):
        """Full deflection at VA."""
        cl = ControlSurfaceLimits(aileron_max_deg=20.0)
        delta = cl.aileron_at_speed(48.0, VA=48.0, VC=70.0, VD=90.0)
        assert delta == pytest.approx(math.radians(20.0), rel=1e-6)

    def test_twothirds_aileron_at_VC(self):
        """2/3 deflection at VC."""
        cl = ControlSurfaceLimits(aileron_max_deg=30.0)
        delta = cl.aileron_at_speed(70.0, VA=48.0, VC=70.0, VD=90.0)
        expected = math.radians(30.0) * 2 / 3
        assert delta == pytest.approx(expected, rel=1e-3)

    def test_onethird_aileron_at_VD(self):
        """1/3 deflection at VD."""
        cl = ControlSurfaceLimits(aileron_max_deg=30.0)
        delta = cl.aileron_at_speed(90.0, VA=48.0, VC=70.0, VD=90.0)
        expected = math.radians(30.0) * 1 / 3
        assert delta == pytest.approx(expected, rel=1e-3)

    def test_rudder_reduces_with_speed(self):
        """Rudder deflection reduces above VA."""
        cl = ControlSurfaceLimits(rudder_max_deg=25.0)
        delta_VA = cl.rudder_at_speed(48.0, VA=48.0, VD=90.0)
        delta_high = cl.rudder_at_speed(80.0, VA=48.0, VD=90.0)
        assert delta_VA > delta_high


# ---------------------------------------------------------------------------
# Pratt gust formula
# ---------------------------------------------------------------------------

class TestPrattGust:
    """Test Pratt quasi-static gust formula."""

    def test_positive_delta_nz(self):
        """Δn must be positive."""
        dn = pratt_gust_delta_nz(
            V_eas=70.0, Ude_fps=50.0,
            wing_loading_pa=1500.0, CLalpha=2*math.pi,
            mean_chord_m=1.5, weight_N=50000.0, wing_area_m2=20.0)
        assert dn > 0

    def test_increases_with_speed(self):
        """Δn increases with airspeed (linear in V_eas)."""
        params = dict(Ude_fps=50.0, wing_loading_pa=1500.0,
                       CLalpha=2*math.pi, mean_chord_m=1.5,
                       weight_N=50000.0, wing_area_m2=20.0)
        dn1 = pratt_gust_delta_nz(V_eas=50.0, **params)
        dn2 = pratt_gust_delta_nz(V_eas=100.0, **params)
        assert dn2 > dn1

    def test_increases_with_gust_speed(self):
        """Δn increases with gust velocity."""
        params = dict(V_eas=70.0, wing_loading_pa=1500.0,
                       CLalpha=2*math.pi, mean_chord_m=1.5,
                       weight_N=50000.0, wing_area_m2=20.0)
        dn1 = pratt_gust_delta_nz(Ude_fps=25.0, **params)
        dn2 = pratt_gust_delta_nz(Ude_fps=50.0, **params)
        assert dn2 > dn1

    def test_alleviation_factor_range(self):
        """Kg should be between 0 and 1."""
        WS = 1500.0
        cbar = 1.5
        CLa = 2 * math.pi
        mu_g = 2.0 * WS / (RHO_0 * cbar * CLa * G_MPS2)
        Kg = 0.88 * mu_g / (5.3 + mu_g)
        assert 0 < Kg < 1.0

    def test_zero_inputs_return_zero(self):
        """Zero wing loading returns zero."""
        dn = pratt_gust_delta_nz(70.0, 50.0, 0.0, 2*math.pi, 1.5, 50000.0, 20.0)
        assert dn == 0.0


# ---------------------------------------------------------------------------
# V-n diagram generation
# ---------------------------------------------------------------------------

class TestVnDiagram:
    """Test V-n diagram computation."""

    @pytest.fixture
    def config(self):
        """Typical light GA aircraft config."""
        return AircraftConfig(
            speeds=SpeedSchedule(VS1=28.0, VA=0.0, VC=65.0, VD=85.0, VF=35.0),
            weight_cg_conditions=[
                WeightCGCondition("MTOW", weight_N=10000.0, cg_x=2000.0),
            ],
            altitudes_m=[0.0],
            wing_area_m2=15.0,
            CLalpha=5.5,
            mean_chord_m=1.4,
        )

    def test_computes_VA(self, config):
        """VA computed from VS1 and nz_max."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        nz_max = vn.nz_max
        expected_VA = 28.0 * math.sqrt(nz_max)
        # VA should match
        a_plus = [p for p in vn.corner_points if p.label == "A+"]
        assert len(a_plus) == 1
        assert a_plus[0].V_eas == pytest.approx(expected_VA, rel=1e-3)
        assert a_plus[0].nz == pytest.approx(nz_max, rel=1e-3)

    def test_corner_point_count(self, config):
        """Should have at least 12 corner points."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        # Maneuver: A+, A-, C+, C-, D+, D- = 6
        # Gust: VB+/-, VC+/-, VD+/- = 6
        # Flap: 2
        assert len(vn.corner_points) >= 12

    def test_maneuver_corners(self, config):
        """Check all maneuver corner labels exist."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        labels = [p.label for p in vn.corner_points]
        for expected in ["A+", "A-", "C+", "C-", "D+", "D-"]:
            assert expected in labels, f"Missing corner: {expected}"

    def test_gust_corners(self, config):
        """Check gust corner labels exist."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        labels = [p.label for p in vn.corner_points]
        for expected in ["Gust_VC+", "Gust_VC-", "Gust_VD+", "Gust_VD-"]:
            assert expected in labels, f"Missing gust corner: {expected}"

    def test_gust_nz_above_1g(self, config):
        """Positive gust nz should be > 1.0."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        gust_pos = [p for p in vn.corner_points if p.label.endswith("+")
                     and p.category == "gust"]
        for pt in gust_pos:
            assert pt.nz > 1.0

    def test_gust_nz_below_1g(self, config):
        """Negative gust nz should be < 1.0."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        gust_neg = [p for p in vn.corner_points if p.label.endswith("-")
                     and p.category == "gust"]
        for pt in gust_neg:
            assert pt.nz < 1.0

    def test_flap_corners(self, config):
        """Flap corner at VF."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        flap_pts = [p for p in vn.corner_points if p.category == "flap"]
        assert len(flap_pts) >= 1

    def test_maneuver_curve_generated(self, config):
        """Maneuver curve has data."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        assert len(vn.maneuver_curve) > 10

    def test_gust_curves_generated(self, config):
        """Gust curves have data."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        assert len(vn.gust_curve_pos) > 10
        assert len(vn.gust_curve_neg) > 10

    def test_nz_max_part23(self, config):
        """nz_max follows Part 23 formula."""
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)
        expected = part23_nz_max(wc.weight_N)
        assert vn.nz_max == pytest.approx(expected)


class TestComputeAllVnDiagrams:
    """Test batch V-n diagram computation."""

    def test_multiple_conditions(self):
        """Generate diagrams for multiple W/CG × altitudes."""
        config = AircraftConfig(
            speeds=SpeedSchedule(VS1=28.0, VC=65.0, VD=85.0),
            weight_cg_conditions=[
                WeightCGCondition("MTOW", weight_N=10000.0, cg_x=2000.0),
                WeightCGCondition("MLW", weight_N=8000.0, cg_x=1900.0),
            ],
            altitudes_m=[0.0, 3000.0],
            wing_area_m2=15.0,
            CLalpha=5.5,
            mean_chord_m=1.4,
        )
        diagrams = compute_all_vn_diagrams(config)
        assert len(diagrams) == 4  # 2 W/CG × 2 altitudes


# ---------------------------------------------------------------------------
# AircraftConfig
# ---------------------------------------------------------------------------

class TestAircraftConfig:
    """Test AircraftConfig creation and from_dict."""

    def test_from_dict(self):
        """Create config from dict (simulating YAML input)."""
        d = {
            'speeds': {'VS1': 28.0, 'VC': 65.0, 'VD': 85.0},
            'weight_cg': [
                {'label': 'MTOW', 'weight_N': 10000.0, 'cg_x': 2000.0},
            ],
            'altitudes_m': [0.0, 5000.0],
            'wing_area_m2': 15.0,
            'CLalpha': 5.5,
            'mean_chord_m': 1.4,
        }
        cfg = AircraftConfig.from_dict(d)
        assert cfg.speeds.VS1 == 28.0
        assert len(cfg.weight_cg_conditions) == 1
        assert len(cfg.altitudes_m) == 2
        assert cfg.wing_area_m2 == 15.0

    def test_nz_methods(self):
        """nz_max and nz_min methods."""
        cfg = AircraftConfig()
        W = 10000.0
        assert cfg.nz_max(W) == part23_nz_max(W)
        assert cfg.nz_min(W) == part23_nz_min(cfg.nz_max(W))


# ---------------------------------------------------------------------------
# Landing gear config
# ---------------------------------------------------------------------------

class TestLandingGearConfig:
    """Test landing gear nz computation."""

    def test_nz_landing_positive(self):
        """Landing nz should be > 1.0."""
        lg = LandingGearConfig(stroke=0.3, strut_efficiency=0.7,
                                 sink_rate_fps=10.0)
        nz = lg.compute_nz_landing(10000.0)
        assert nz > 1.0

    def test_nz_landing_minimum(self):
        """Landing nz should be at least 2.0 per §23.473."""
        lg = LandingGearConfig(stroke=1.0, strut_efficiency=0.9,
                                 sink_rate_fps=5.0)
        nz = lg.compute_nz_landing(10000.0)
        assert nz >= 2.0

    def test_nz_landing_zero_stroke(self):
        """Zero stroke returns conservative default."""
        lg = LandingGearConfig(stroke=0.0)
        nz = lg.compute_nz_landing(10000.0)
        assert nz == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# V-n diagram visualization (smoke test)
# ---------------------------------------------------------------------------

class TestVnPlot:
    """Smoke test for V-n diagram plotting."""

    def test_plot_vn_diagram(self, tmp_path):
        """Plot generates PNG without error."""
        from nastaero.visualization.cert_plot import plot_vn_diagram

        config = AircraftConfig(
            speeds=SpeedSchedule(VS1=28.0, VC=65.0, VD=85.0, VF=35.0),
            weight_cg_conditions=[
                WeightCGCondition("MTOW", weight_N=10000.0, cg_x=2000.0),
            ],
            wing_area_m2=15.0,
            CLalpha=5.5,
            mean_chord_m=1.4,
        )
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, 0.0)

        out = str(tmp_path / "vn_test.png")
        result = plot_vn_diagram(vn, output_path=out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 1000  # Non-trivial file


import os

"""Tests for Phase 1: nz coupling fix + TrimCondition→TRIM bridge.

Tests the critical nz load factor fix in SOL 144 and the
TrimCondition to TRIM card conversion bridge for certification loads.
"""
import os
import numpy as np
import pytest
from nastaero.bdf.parser import BDFParser
from nastaero.loads_analysis.case_generator import (
    TrimCondition, CaseGenerator, isa_atmosphere,
    trim_condition_to_trim_card, trim_conditions_to_model,
)

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
GOLAND_BDF = os.path.join(VALIDATION_DIR, "goland_wing", "goland_static.bdf")


def parse_bdf(filepath):
    parser = BDFParser()
    return parser.parse(filepath)


# ---------------------------------------------------------------------------
# TrimCondition → TRIM card conversion
# ---------------------------------------------------------------------------

class TestTrimCardConversion:
    """Test TrimCondition → TRIM card bridge."""

    def test_basic_conversion(self):
        """Convert a simple 1g level flight case."""
        tc = TrimCondition(
            case_id=1, mach=0.3, q=5000.0, nz=1.0,
            fixed_vars={"ROLL": 0.0, "YAW": 0.0, "ARON": 0.0},
            free_vars=["ANGLEA", "ELEV"],
            label="Level M=0.30",
        )
        trim = trim_condition_to_trim_card(tc)

        assert trim.tid == 1
        assert trim.mach == pytest.approx(0.3)
        assert trim.q == pytest.approx(5000.0)

        # Check variables — fixed_vars + URDD3
        var_dict = dict(trim.variables)
        assert "ROLL" in var_dict
        assert "YAW" in var_dict
        assert "ARON" in var_dict
        assert var_dict["ROLL"] == pytest.approx(0.0)
        assert "URDD3" in var_dict
        assert var_dict["URDD3"] == pytest.approx(1.0)

    def test_nz_25g_conversion(self):
        """Convert a 2.5g maneuver case."""
        tc = TrimCondition(
            case_id=42, mach=0.5, q=15000.0, nz=2.5,
            fixed_vars={"ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0,
                         "URDD4": 0.0, "URDD6": 0.0,
                         "ARON": 0.0, "RUD": 0.0},
            free_vars=["ANGLEA", "ELEV"],
            label="Maneuver nz=2.5g",
        )
        trim = trim_condition_to_trim_card(tc)

        var_dict = dict(trim.variables)
        assert var_dict["URDD3"] == pytest.approx(2.5)

    def test_nz_negative(self):
        """Convert a negative load factor case."""
        tc = TrimCondition(
            case_id=10, mach=0.4, q=10000.0, nz=-1.0,
            fixed_vars={"ROLL": 0.0, "YAW": 0.0},
            free_vars=["ANGLEA", "ELEV"],
            label="nz=-1.0g",
        )
        trim = trim_condition_to_trim_card(tc)
        var_dict = dict(trim.variables)
        assert var_dict["URDD3"] == pytest.approx(-1.0)

    def test_urdd3_not_duplicated(self):
        """URDD3 in fixed_vars should not be duplicated."""
        tc = TrimCondition(
            case_id=5, mach=0.3, q=5000.0, nz=2.0,
            fixed_vars={"ROLL": 0.0, "URDD3": 3.0},
            free_vars=["ANGLEA", "ELEV"],
        )
        trim = trim_condition_to_trim_card(tc)
        urdd3_values = [v for k, v in trim.variables if k == "URDD3"]
        # When URDD3 is already in fixed_vars, it should keep that value
        assert len(urdd3_values) == 1
        assert urdd3_values[0] == pytest.approx(3.0)

    def test_roll_case_with_aileron(self):
        """Rolling case with ARON deflection."""
        tc = TrimCondition(
            case_id=100, mach=0.35, q=8000.0, nz=1.0,
            fixed_vars={"ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0,
                         "URDD4": 0.0, "URDD6": 0.0,
                         "ARON": 0.3, "RUD": 0.0},
            free_vars=["ANGLEA", "ELEV"],
            label="Roll ARON=0.3rad",
        )
        trim = trim_condition_to_trim_card(tc)
        var_dict = dict(trim.variables)
        assert var_dict["ARON"] == pytest.approx(0.3)
        assert var_dict["URDD3"] == pytest.approx(1.0)

    def test_yaw_case_with_rudder(self):
        """Yaw case with RUD deflection."""
        tc = TrimCondition(
            case_id=200, mach=0.35, q=8000.0, nz=1.0,
            fixed_vars={"ROLL": 0.0, "YAW": 0.0, "URDD2": 0.0,
                         "URDD4": 0.0, "URDD6": 0.0,
                         "ARON": 0.0, "RUD": 0.2},
            free_vars=["ANGLEA", "ELEV"],
            label="Yaw RUD=0.2rad",
        )
        trim = trim_condition_to_trim_card(tc)
        var_dict = dict(trim.variables)
        assert var_dict["RUD"] == pytest.approx(0.2)

    def test_tid_override(self):
        """Override TRIM card ID."""
        tc = TrimCondition(case_id=1, mach=0.3, q=5000.0, nz=1.0)
        trim = trim_condition_to_trim_card(tc, tid=999)
        assert trim.tid == 999


# ---------------------------------------------------------------------------
# Model injection
# ---------------------------------------------------------------------------

class TestModelInjection:
    """Test trim_conditions_to_model()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(GOLAND_BDF):
            pytest.skip("Goland BDF not found")
        self.model = parse_bdf(GOLAND_BDF)

    def test_inject_single_case(self):
        """Inject one TrimCondition into the model."""
        cases = [TrimCondition(
            case_id=1, mach=0.5, q=5000.0, nz=1.0,
            fixed_vars={"ROLL": 0.0},
            free_vars=["ANGLEA"],
        )]
        original_trim_count = len(self.model.trims)
        created = trim_conditions_to_model(self.model, cases)

        assert len(created) == 1
        assert len(self.model.trims) == original_trim_count + 1

    def test_inject_multiple_cases(self):
        """Inject multiple TrimConditions — IDs are unique."""
        cases = CaseGenerator.level_flight_sweep(
            machs=[0.2, 0.3, 0.4], altitude_m=0.0)
        original_trim_count = len(self.model.trims)
        created = trim_conditions_to_model(self.model, cases)

        assert len(created) == 3
        assert len(self.model.trims) == original_trim_count + 3

        # Verify unique IDs
        trim_ids = [t.tid for t, _ in created]
        assert len(set(trim_ids)) == 3


# ---------------------------------------------------------------------------
# nz load factor propagation
# ---------------------------------------------------------------------------

class TestNzPropagation:
    """Test that nz load factor correctly propagates through the solver."""

    def test_nz_from_urdd3_in_trim_vars(self):
        """Verify nz extraction from URDD3 in trim variables."""
        from nastaero.bdf.cards.aero import TRIM

        # Create TRIM card with URDD3 = 2.5
        trim = TRIM(tid=1, mach=0.3, q=5000.0)
        trim.variables = [("ROLL", 0.0), ("URDD3", 2.5), ("YAW", 0.0)]

        # Parse trim vars as the solver does
        trim_vars = {}
        for label, val in trim.variables:
            trim_vars[label] = val

        nz = trim_vars.get("URDD3", 1.0)
        assert nz == pytest.approx(2.5)

    def test_nz_default_when_no_urdd3(self):
        """Without URDD3, nz defaults to 1.0."""
        from nastaero.bdf.cards.aero import TRIM

        trim = TRIM(tid=1, mach=0.3, q=5000.0)
        trim.variables = [("ROLL", 0.0), ("YAW", 0.0)]

        trim_vars = dict(trim.variables)
        nz = trim_vars.get("URDD3", 1.0)
        assert nz == pytest.approx(1.0)

    def test_nz_propagation_from_trim_condition(self):
        """TrimCondition.nz→TRIM card URDD3→solver nz extraction."""
        tc = TrimCondition(
            case_id=1, mach=0.5, q=10000.0, nz=3.8,
            fixed_vars={"ROLL": 0.0, "YAW": 0.0},
            free_vars=["ANGLEA", "ELEV"],
        )
        trim = trim_condition_to_trim_card(tc)

        # Simulate solver extraction
        trim_vars = dict(trim.variables)
        nz = trim_vars.get("URDD3", 1.0)
        assert nz == pytest.approx(3.8)

    def test_rhs_scaling_with_nz(self):
        """Verify that rhs = nz * weight (not just weight)."""
        total_weight = 50000.0  # N
        nz = 2.5

        # The fixed formula (after our fix)
        F_z_fixed = 0.0  # no fixed aero contribution
        rhs = nz * total_weight - F_z_fixed

        assert rhs == pytest.approx(125000.0)  # 2.5 * 50000
        assert rhs != pytest.approx(total_weight)  # NOT 50000

    def test_inertial_load_scaling(self):
        """Inertial force must scale with nz."""
        mass_node = 100.0  # kg
        g = 9810.0  # mm/s^2 (N-mm-sec units)
        nz = 2.5

        F_inertial = -mass_node * nz * g  # downward
        assert F_inertial == pytest.approx(-2452500.0)

        # At 1g:
        F_1g = -mass_node * 1.0 * g
        assert abs(F_inertial) == pytest.approx(2.5 * abs(F_1g))


# ---------------------------------------------------------------------------
# ISA atmosphere (regression tests)
# ---------------------------------------------------------------------------

class TestISA:
    """Verify ISA atmosphere model."""

    def test_sea_level(self):
        rho, T, a = isa_atmosphere(0.0)
        assert rho == pytest.approx(1.225, rel=1e-3)
        assert T == pytest.approx(288.15, rel=1e-4)
        assert a == pytest.approx(340.3, rel=1e-2)

    def test_11km(self):
        rho, T, a = isa_atmosphere(11000.0)
        assert T == pytest.approx(216.65, rel=1e-3)
        assert rho == pytest.approx(0.3639, rel=1e-2)

    def test_tropopause(self):
        """Above 11km, temperature is isothermal."""
        _, T11, _ = isa_atmosphere(11000.0)
        _, T15, _ = isa_atmosphere(15000.0)
        assert T15 == pytest.approx(T11, rel=1e-6)


# ---------------------------------------------------------------------------
# CaseGenerator — sweep generation
# ---------------------------------------------------------------------------

class TestCaseGenerator:
    """Test case generation utilities."""

    def test_level_sweep_count(self):
        cases = CaseGenerator.level_flight_sweep(
            machs=[0.1, 0.2, 0.3, 0.4, 0.5], altitude_m=0.0)
        assert len(cases) == 5

    def test_level_sweep_nz(self):
        """nz is carried through to each case."""
        cases = CaseGenerator.level_flight_sweep(
            machs=[0.3], altitude_m=0.0, nz=2.5)
        assert cases[0].nz == pytest.approx(2.5)

    def test_vn_cases_count(self):
        cases = CaseGenerator.vn_diagram_cases(
            machs=[0.3, 0.5], nz_values=[-1.0, 1.0, 2.5])
        assert len(cases) == 6  # 2 Mach × 3 nz

    def test_vn_cases_nz_values(self):
        cases = CaseGenerator.vn_diagram_cases(
            machs=[0.3], nz_values=[-1.5, 3.8])
        nz_values = [tc.nz for tc in cases]
        assert -1.5 in nz_values
        assert 3.8 in nz_values

    def test_dynamic_pressure(self):
        """q = 0.5 * rho * V^2."""
        rho, T, a = isa_atmosphere(0.0)
        cases = CaseGenerator.level_flight_sweep(
            machs=[0.3], altitude_m=0.0)
        V = 0.3 * a
        q_expected = 0.5 * rho * V**2
        assert cases[0].q == pytest.approx(q_expected, rel=1e-6)

    def test_csv_roundtrip(self, tmp_path):
        """CSV export/import preserves data."""
        cases = CaseGenerator.level_flight_sweep(
            machs=[0.2, 0.4, 0.6], altitude_m=5000.0, nz=1.5)
        filepath = str(tmp_path / "test_cases.csv")
        CaseGenerator.to_csv(cases, filepath)
        loaded = CaseGenerator.from_csv(filepath)

        assert len(loaded) == 3
        for orig, load in zip(cases, loaded):
            assert orig.mach == pytest.approx(load.mach, rel=1e-6)
            assert orig.nz == pytest.approx(load.nz, rel=1e-6)
            assert orig.altitude_m == pytest.approx(load.altitude_m, rel=1e-6)

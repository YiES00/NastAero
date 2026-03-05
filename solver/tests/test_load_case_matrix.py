"""Tests for Phase 3: Load case matrix generation.

Tests the complete certification load case matrix including symmetric
maneuver, gust, rolling, yaw, checked maneuver, flap, and landing cases
per FAA Part 23 §23.321-§23.511.
"""
import math
import os
import pytest

from nastaero.loads_analysis.certification.aircraft_config import (
    AircraftConfig, SpeedSchedule, WeightCGCondition,
    ControlSurfaceLimits, LandingGearConfig,
    part23_nz_max, part23_nz_min,
    eas_to_mach, dynamic_pressure_from_eas,
)
from nastaero.loads_analysis.certification.vn_diagram import (
    VnDiagram, VnPoint, compute_vn_diagram,
)
from nastaero.loads_analysis.certification.landing_loads import (
    LandingCondition, LandingConditionType,
    compute_gear_reactions, compute_landing_inertial_forces,
    combine_forces,
    generate_level_landing, generate_tail_down_landing,
    generate_one_wheel_landing, generate_side_load,
    generate_rebound, generate_ground_handling,
    generate_all_landing_conditions,
)
from nastaero.loads_analysis.certification.load_case_matrix import (
    CertLoadCase, LoadCaseMatrix,
)

import numpy as np


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_test_config(n_weights=2, n_altitudes=2):
    """Create a standard test aircraft configuration."""
    W1 = 12000.0 * 9.80665  # ~12000 kg → N
    W2 = 8000.0 * 9.80665   # ~8000 kg → N

    weights = [
        WeightCGCondition(label="MTOW Fwd", weight_N=W1, cg_x=5000.0),
        WeightCGCondition(label="MLW Aft", weight_N=W2, cg_x=5500.0),
    ][:n_weights]

    altitudes = [0.0, 5000.0][:n_altitudes]

    return AircraftConfig(
        speeds=SpeedSchedule(
            VS1=30.0, VA=60.0, VB=0.0, VC=80.0, VD=100.0, VF=40.0,
        ),
        weight_cg_conditions=weights,
        altitudes_m=altitudes,
        wing_area_m2=20.0,
        CLalpha=5.5,
        mean_chord_m=1.5,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0,
            rudder_max_deg=25.0,
            elevator_max_deg=25.0,
        ),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[100, 101],
            nose_gear_node_ids=[102],
            main_gear_x=5200.0,
            nose_gear_x=2000.0,
            strut_efficiency=0.7,
            stroke=0.3,
            sink_rate_fps=10.0,
        ),
        gust_Ude_VC_fps=50.0,
        gust_Ude_VD_fps=25.0,
    )


# ---------------------------------------------------------------------------
# Landing loads tests
# ---------------------------------------------------------------------------

class TestGearReactions:
    """Test gear reaction force computation."""

    def test_level_landing_vertical(self):
        """Level landing produces upward gear reactions."""
        wc = WeightCGCondition(label="test", weight_N=50000.0, cg_x=5000.0)
        gear = LandingGearConfig(
            main_gear_node_ids=[100, 101],
            nose_gear_node_ids=[102],
            main_gear_x=5200.0,
            nose_gear_x=2000.0,
        )

        cond = LandingCondition(
            nz_cg=2.5,
            weight_cg=wc,
            main_gear_vertical_frac=0.8,
            nose_gear_vertical_frac=0.2,
        )

        forces = compute_gear_reactions(cond, gear)
        assert len(forces) == 3  # 2 main + 1 nose

        # Total vertical force should equal nz × W
        total_fz = sum(f[2] for f in forces.values())
        assert total_fz == pytest.approx(2.5 * 50000.0, rel=1e-6)

    def test_one_wheel_landing(self):
        """One-wheel landing applies 0.75 factor to one gear."""
        wc = WeightCGCondition(label="test", weight_N=50000.0, cg_x=5000.0)
        gear = LandingGearConfig(
            main_gear_node_ids=[100, 101],
            nose_gear_node_ids=[],
        )

        cond = LandingCondition(
            condition_type=LandingConditionType.ONE_WHEEL,
            nz_cg=2.5,
            weight_cg=wc,
            main_gear_vertical_frac=1.0,
            one_wheel_side="left",
        )

        forces = compute_gear_reactions(cond, gear)
        # Only left gear (node 100) should have load
        assert 100 in forces
        assert forces[100][2] == pytest.approx(2.5 * 50000.0 * 0.75, rel=1e-6)

    def test_side_load_has_lateral(self):
        """Side load condition produces lateral forces."""
        wc = WeightCGCondition(label="test", weight_N=50000.0, cg_x=5000.0)
        gear = LandingGearConfig(
            main_gear_node_ids=[100],
            nose_gear_node_ids=[],
        )

        cond = LandingCondition(
            nz_cg=2.0,
            weight_cg=wc,
            lateral_factor=0.8,
            main_gear_vertical_frac=1.0,
        )

        forces = compute_gear_reactions(cond, gear)
        assert forces[100][1] == pytest.approx(2.0 * 50000.0 * 0.8, rel=1e-6)

    def test_braking_drag(self):
        """Braking condition produces rearward drag forces."""
        wc = WeightCGCondition(label="test", weight_N=50000.0, cg_x=5000.0)
        gear = LandingGearConfig(
            main_gear_node_ids=[100],
            nose_gear_node_ids=[],
        )

        cond = LandingCondition(
            nz_cg=1.0,
            weight_cg=wc,
            drag_factor=0.8,
            main_gear_vertical_frac=1.0,
        )

        forces = compute_gear_reactions(cond, gear)
        # Drag is negative Fx
        assert forces[100][0] < 0
        assert forces[100][0] == pytest.approx(
            -1.0 * 50000.0 * 0.8, rel=1e-6)


class TestLandingInertialForces:
    """Test inertial force computation for landing."""

    def test_inertial_downward(self):
        """Inertial forces are downward (opposing gear reaction)."""
        cond = LandingCondition(nz_cg=2.5)
        node_masses = {1: 100.0, 2: 200.0}

        forces = compute_landing_inertial_forces(cond, node_masses)
        assert forces[1][2] < 0  # Downward
        assert forces[2][2] < 0

    def test_inertial_scales_with_nz(self):
        """Inertial forces scale linearly with nz."""
        node_masses = {1: 100.0}

        f1g = compute_landing_inertial_forces(
            LandingCondition(nz_cg=1.0), node_masses)
        f25g = compute_landing_inertial_forces(
            LandingCondition(nz_cg=2.5), node_masses)

        assert abs(f25g[1][2]) == pytest.approx(2.5 * abs(f1g[1][2]), rel=1e-6)


class TestCombineForces:
    """Test force combination."""

    def test_combine_overlapping_nodes(self):
        """Forces at same node are summed."""
        f1 = {1: np.array([100, 0, 0, 0, 0, 0], dtype=float)}
        f2 = {1: np.array([0, 200, 0, 0, 0, 0], dtype=float)}

        combined = combine_forces(f1, f2)
        np.testing.assert_allclose(
            combined[1], [100, 200, 0, 0, 0, 0])

    def test_combine_disjoint_nodes(self):
        """Forces at different nodes are preserved."""
        f1 = {1: np.array([100, 0, 0, 0, 0, 0], dtype=float)}
        f2 = {2: np.array([0, 200, 0, 0, 0, 0], dtype=float)}

        combined = combine_forces(f1, f2)
        assert 1 in combined
        assert 2 in combined


class TestLandingConditionGenerators:
    """Test landing condition generators."""

    def test_level_landing(self):
        """Level landing generates 2 cases (with/without drag)."""
        config = make_test_config()
        wc = config.weight_cg_conditions[0]
        cases = generate_level_landing(1, config, wc)
        assert len(cases) == 2
        assert all(c.condition_type == LandingConditionType.LEVEL_LANDING
                    for c in cases)
        assert cases[0].nz_cg >= 2.0

    def test_tail_down_landing(self):
        """Tail-down generates 1 case with main gear only."""
        config = make_test_config()
        wc = config.weight_cg_conditions[0]
        cases = generate_tail_down_landing(10, config, wc)
        assert len(cases) == 1
        assert cases[0].nose_gear_vertical_frac == 0.0
        assert cases[0].main_gear_vertical_frac == 1.0

    def test_one_wheel_landing_both_sides(self):
        """One-wheel generates 2 cases (left + right)."""
        config = make_test_config()
        wc = config.weight_cg_conditions[0]
        cases = generate_one_wheel_landing(20, config, wc)
        assert len(cases) == 2
        sides = [c.one_wheel_side for c in cases]
        assert "left" in sides
        assert "right" in sides

    def test_side_load_has_lateral_factor(self):
        """Side load has non-zero lateral factor."""
        config = make_test_config()
        wc = config.weight_cg_conditions[0]
        cases = generate_side_load(30, config, wc)
        assert len(cases) == 1
        assert cases[0].lateral_factor > 0

    def test_rebound(self):
        """Rebound generates 1 case."""
        config = make_test_config()
        wc = config.weight_cg_conditions[0]
        cases = generate_rebound(40, config, wc)
        assert len(cases) == 1
        assert cases[0].far_section == "§23.487"

    def test_ground_handling(self):
        """Ground handling generates taxi + braking + turning + nose yaw."""
        config = make_test_config()
        wc = config.weight_cg_conditions[0]
        cases = generate_ground_handling(50, config, wc)
        assert len(cases) >= 3  # taxi, braked, turning (+ maybe nose yaw)
        far_sections = {c.far_section for c in cases}
        assert "§23.491" in far_sections
        assert "§23.493" in far_sections
        assert "§23.497" in far_sections

    def test_all_landing_conditions(self):
        """Generate all landing conditions for complete config."""
        config = make_test_config()
        cases = generate_all_landing_conditions(config)
        # 2 weight conditions ×
        #   (2 level + 1 tail + 2 one-wheel + 1 side + 1 rebound + ~4 ground)
        # = 2 × 11 = 22 (approximately)
        assert len(cases) >= 18  # At least 9 per weight condition
        assert len(cases) <= 30  # Sanity upper bound

        # Unique IDs
        ids = [c.case_id for c in cases]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Load case matrix tests
# ---------------------------------------------------------------------------

class TestLoadCaseMatrix:
    """Test complete load case matrix generation."""

    @pytest.fixture
    def matrix(self):
        config = make_test_config()
        m = LoadCaseMatrix(config)
        m.generate_all()
        return m

    def test_has_all_categories(self, matrix):
        """All required categories are present."""
        summary = matrix.summary()
        assert "symmetric" in summary
        assert "gust" in summary
        assert "rolling" in summary
        assert "yaw" in summary
        assert "checked" in summary
        assert "landing" in summary

    def test_symmetric_cases_exist(self, matrix):
        """Symmetric cases are generated from V-n corners."""
        sym = matrix.cases_by_category("symmetric")
        assert len(sym) > 0
        # Each V-n diagram has ~6 maneuver corners × 2 weights × 2 altitudes
        assert len(sym) >= 12

    def test_gust_cases_exist(self, matrix):
        """Gust cases from Pratt formula are generated."""
        gust = matrix.cases_by_category("gust")
        assert len(gust) > 0
        # ~6 gust corners × 2 weights × 2 altitudes
        assert len(gust) >= 12

    def test_rolling_cases(self, matrix):
        """Rolling cases have ARON deflection in fixed vars."""
        rolling = matrix.cases_by_category("rolling")
        assert len(rolling) > 0

        for c in rolling:
            tc = c.trim_condition
            assert "ARON" in tc.fixed_vars
            assert tc.fixed_vars["ARON"] != 0.0  # Non-zero aileron
            assert tc.nz == pytest.approx(1.0)  # nz=1.0 for rolling

    def test_rolling_aileron_schedule(self, matrix):
        """Rolling cases follow §23.349 aileron schedule."""
        rolling = matrix.cases_by_category("rolling")

        # Group by speed label from the label string
        va_cases = [c for c in rolling if "VA" in c.trim_condition.label]
        vc_cases = [c for c in rolling if "VC" in c.trim_condition.label]
        vd_cases = [c for c in rolling if "VD" in c.trim_condition.label]

        if va_cases and vc_cases:
            # VA should have larger aileron deflection than VC
            aron_va = abs(va_cases[0].trim_condition.fixed_vars["ARON"])
            aron_vc = abs(vc_cases[0].trim_condition.fixed_vars["ARON"])
            assert aron_va > aron_vc

        if vc_cases and vd_cases:
            aron_vc = abs(vc_cases[0].trim_condition.fixed_vars["ARON"])
            aron_vd = abs(vd_cases[0].trim_condition.fixed_vars["ARON"])
            assert aron_vc > aron_vd

    def test_rolling_left_right(self, matrix):
        """Rolling cases include both left and right turns."""
        rolling = matrix.cases_by_category("rolling")
        aron_signs = [
            c.trim_condition.fixed_vars["ARON"] > 0 for c in rolling]
        assert True in aron_signs   # At least one positive
        assert False in aron_signs  # At least one negative

    def test_yaw_cases(self, matrix):
        """Yaw cases have RUD deflection."""
        yaw = matrix.cases_by_category("yaw")
        assert len(yaw) > 0

        for c in yaw:
            tc = c.trim_condition
            assert "RUD" in tc.fixed_vars
            assert tc.fixed_vars["RUD"] != 0.0

    def test_yaw_has_overswing(self, matrix):
        """Yaw cases include overswing sideslip conditions."""
        yaw = matrix.cases_by_category("yaw")
        overswing = [c for c in yaw if "overswing" in c.trim_condition.label]
        assert len(overswing) > 0

        # Overswing cases should have SIDES in fixed_vars
        for c in overswing:
            assert "SIDES" in c.trim_condition.fixed_vars

    def test_checked_maneuver_cases(self, matrix):
        """Checked maneuver includes nz_max, 0g, and nz_min."""
        checked = matrix.cases_by_category("checked")
        assert len(checked) > 0

        nz_values = [c.trim_condition.nz for c in checked]
        # Should include positive nz_max, 0.0, and negative nz_min
        assert any(nz > 2.0 for nz in nz_values)
        assert any(nz == pytest.approx(0.0) for nz in nz_values)
        assert any(nz < 0 for nz in nz_values)

    def test_flap_cases(self, matrix):
        """Flap cases are generated at VF."""
        flap = matrix.cases_by_category("flap")
        assert len(flap) > 0
        assert all("Flap" in c.trim_condition.label for c in flap)

    def test_landing_cases_generated(self, matrix):
        """Landing cases are populated."""
        assert len(matrix.landing_cases) > 0

    def test_unique_case_ids(self, matrix):
        """All case IDs are unique."""
        ids = matrix.all_case_ids()
        assert len(set(ids)) == len(ids)

    def test_total_cases_reasonable(self, matrix):
        """Total case count is in expected range."""
        # 2 weights × 2 altitudes × (~12 sym + ~12 gust + ~12 roll
        #   + ~12 yaw + ~6 checked + ~3 flap) ≈ 228 flight
        # + ~22 landing ≈ 250 total
        total = matrix.total_cases
        assert total >= 100  # Minimum sanity
        assert total <= 1000  # Maximum sanity

    def test_far_section_coverage(self, matrix):
        """Key FAR sections are covered."""
        sections = matrix.far_sections_covered()
        required = ["§23.337", "§23.341", "§23.349", "§23.351"]
        for sec in required:
            assert sec in sections, f"Missing FAR section: {sec}"


class TestLoadCaseMatrixSingleCondition:
    """Test matrix with single weight/altitude for count verification."""

    @pytest.fixture
    def matrix_1x1(self):
        config = make_test_config(n_weights=1, n_altitudes=1)
        m = LoadCaseMatrix(config)
        m.generate_all()
        return m

    def test_symmetric_count(self, matrix_1x1):
        """6 maneuver corners → 6 symmetric cases."""
        sym = matrix_1x1.cases_by_category("symmetric")
        assert len(sym) == 6  # A+, A-, C+, C-, D+, D-

    def test_gust_count(self, matrix_1x1):
        """6 gust corners → 6 gust cases."""
        gust = matrix_1x1.cases_by_category("gust")
        assert len(gust) == 6  # VB±, VC±, VD±

    def test_rolling_count(self, matrix_1x1):
        """3 speeds × 2 directions = 6 rolling cases."""
        rolling = matrix_1x1.cases_by_category("rolling")
        assert len(rolling) == 6

    def test_yaw_count(self, matrix_1x1):
        """3 speeds × 2 directions × 2 snapshots = 12 yaw cases."""
        yaw = matrix_1x1.cases_by_category("yaw")
        assert len(yaw) == 12

    def test_checked_count(self, matrix_1x1):
        """2 speeds × 3 nz = 6 checked cases."""
        checked = matrix_1x1.cases_by_category("checked")
        assert len(checked) == 6

    def test_flap_count(self, matrix_1x1):
        """3 nz values at VF = 3 flap cases."""
        flap = matrix_1x1.cases_by_category("flap")
        assert len(flap) == 3


class TestLoadCaseMatrixCSV:
    """Test CSV roundtrip for load case matrix."""

    def test_csv_roundtrip(self, tmp_path):
        """CSV export/import preserves data."""
        config = make_test_config(n_weights=1, n_altitudes=1)
        m = LoadCaseMatrix(config)
        m.generate_all()

        filepath = str(tmp_path / "cert_cases.csv")
        m.to_csv(filepath)

        # Verify file exists and has data
        import csv as csv_mod
        with open(filepath, 'r') as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(m.flight_cases)

        # Reimport
        m2 = LoadCaseMatrix.from_csv(filepath, config)
        assert len(m2.flight_cases) == len(m.flight_cases)

    def test_csv_categories_preserved(self, tmp_path):
        """Categories are preserved through CSV roundtrip."""
        config = make_test_config(n_weights=1, n_altitudes=1)
        m = LoadCaseMatrix(config)
        m.generate_all()

        filepath = str(tmp_path / "cert_cases.csv")
        m.to_csv(filepath)
        m2 = LoadCaseMatrix.from_csv(filepath, config)

        orig_cats = set(c.category for c in m.flight_cases)
        loaded_cats = set(c.category for c in m2.flight_cases)
        assert orig_cats == loaded_cats


class TestCertLoadCase:
    """Test CertLoadCase dataclass."""

    def test_case_id_from_trim(self):
        """case_id property reads from TrimCondition."""
        from nastaero.loads_analysis.case_generator import TrimCondition
        tc = TrimCondition(case_id=42, mach=0.3, q=5000.0, nz=1.0)
        clc = CertLoadCase(trim_condition=tc)
        assert clc.case_id == 42

    def test_label_from_trim(self):
        """label property reads from TrimCondition."""
        from nastaero.loads_analysis.case_generator import TrimCondition
        tc = TrimCondition(case_id=1, mach=0.3, q=5000.0, nz=1.0,
                           label="test label")
        clc = CertLoadCase(trim_condition=tc)
        assert clc.label == "test label"

    def test_default_solve_type(self):
        """Default solve type is 'trim'."""
        clc = CertLoadCase()
        assert clc.solve_type == "trim"

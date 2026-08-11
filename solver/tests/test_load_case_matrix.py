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
            main_gear_vertical_frac=0.5,
            one_wheel_side="left",
        )

        forces = compute_gear_reactions(cond, gear)
        # §23.483: 편측이 수평착륙의 그 쪽 반력(전체의 절반)을 전담
        assert 100 in forces
        assert forces[100][2] == pytest.approx(2.5 * 50000.0 * 0.5, rel=1e-6)
        assert forces[101][2] == 0.0

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
        # 자세(3점/노즈클리어) × 드래그(무/스핀업 K) = 4
        assert len(cases) == 4
        assert all(c.condition_type == LandingConditionType.LEVEL_LANDING
                    for c in cases)
        assert cases[0].nz_cg >= 2.67          # §23.473(g)
        assert all(c.lift_factor == pytest.approx(2 / 3) for c in cases)

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
        # §23.485: 좌/우 미끄럼 2방향, 0.5W/0.33W 비대칭, nz=1.33
        assert len(cases) == 2
        assert cases[0].side_frac_per_main == (0.5, 0.33)
        assert cases[0].nz_cg == pytest.approx(1.33)

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
        # 중량조건 2 × (레벨 4 + 테일다운 1 + 원휠 2 + 사이드 2 +
        #   리바운드 1 + 지상 3 + 노즈 보조 3) = 2 × 16 = 32 내외
        assert len(cases) >= 26
        assert len(cases) <= 36

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


class TestManualLandingLoadFactor:
    """FAR 23 LOADS Manual Ch.20 — LGFACTOR.BAS 예제 검증.

    6-place GA: W=3,230 lb, S=184.125 ft², 스트로크 7 in, 타이어
    처짐 (19−7)/6=2 in, 오레오(η=0.75), L=0.667 →
    V=9.0048 ft/s, N=3.0951, N_gear=2.4281.
    """

    def test_lgfactor_example(self):
        gear = LandingGearConfig(
            strut_efficiency=0.75,
            stroke=7.0 * 0.0254,
            tire_deflection=2.0 * 0.0254,
            tire_efficiency=0.3,
            sink_rate_fps=0.0,      # §23.473(d) 규정식 사용
        )
        W_N = 3230.0 * 4.44822
        S_m2 = 184.125 * 0.09290304
        n, n_gear, v_fps = gear.landing_load_factors(W_N, S_m2,
                                                     lift_factor=0.667)
        assert v_fps == pytest.approx(9.0048, rel=1e-3)
        assert n == pytest.approx(3.0951, rel=2e-3)
        assert n_gear == pytest.approx(2.4281, rel=2e-3)

    def test_sink_speed_clamp(self):
        gear = LandingGearConfig(stroke=0.2, sink_rate_fps=0.0)
        # 매우 가벼운 익면하중 → 7 fps 하한
        _, _, v = gear.landing_load_factors(1000.0, 100.0)
        assert v == pytest.approx(7.0)
        # 매우 무거운 익면하중 → 10 fps 상한
        _, _, v = gear.landing_load_factors(1e6, 1.0)
        assert v == pytest.approx(10.0)

    def test_minimums_2367(self):
        """§23.473(g): N ≥ 2.67, N_gear ≥ 2.0."""
        gear = LandingGearConfig(strut_efficiency=0.9, stroke=1.0,
                                 sink_rate_fps=7.0)   # 매우 부드러운 기어
        n, n_gear, _ = gear.landing_load_factors(50000.0, 20.0)
        assert n >= 2.67 - 1e-9
        assert n_gear >= 2.0 - 1e-9


class TestManualCh20CaseFactors:
    """매뉴얼 Ch.20 케이스 계수 — 지면반력·좌우 비대칭 측력·양력비."""

    def _wc(self):
        return WeightCGCondition(label="t", weight_N=20000.0, cg_x=4600.0)

    def _gear(self):
        return LandingGearConfig(
            main_gear_node_ids=[1, 2], nose_gear_node_ids=[3],
            main_gear_x=4700.0, nose_gear_x=1500.0,
            strut_efficiency=0.75, stroke=0.25, sink_rate_fps=10.0)

    def test_level_landing_ground_reaction_uses_n_minus_lift(self):
        """지면반력 = (nz − 2/3)·W (§23.473(e))."""
        cond = LandingCondition(nz_cg=3.0, weight_cg=self._wc(),
                                lift_factor=2 / 3,
                                main_gear_vertical_frac=1.0)
        forces = compute_gear_reactions(cond, self._gear())
        total_fz = sum(f[2] for f in forces.values())
        assert total_fz == pytest.approx((3.0 - 2 / 3) * 20000.0, rel=1e-9)

    def test_one_wheel_carries_level_side_reaction(self):
        """§23.483: 편측 반력 = 수평착륙의 그 쪽 반력 (0.75 계수 아님)."""
        cond = LandingCondition(
            condition_type=LandingConditionType.ONE_WHEEL,
            nz_cg=3.0, weight_cg=self._wc(), lift_factor=2 / 3,
            main_gear_vertical_frac=0.5, one_wheel_side="left")
        forces = compute_gear_reactions(cond, self._gear())
        assert forces[1][2] == pytest.approx(
            (3.0 - 2 / 3) * 20000.0 * 0.5, rel=1e-9)
        assert forces[2][2] == 0.0

    def test_side_load_asymmetric(self):
        """§23.485: 측력 0.5W / 0.33W 비대칭, 수직 1.33W 균등."""
        cond = LandingCondition(
            condition_type=LandingConditionType.SIDE_LOAD,
            nz_cg=1.33, weight_cg=self._wc(),
            side_frac_per_main=(0.5, 0.33),
            main_gear_vertical_frac=1.0)
        forces = compute_gear_reactions(cond, self._gear())
        assert forces[1][1] == pytest.approx(0.5 * 20000.0, rel=1e-9)
        assert forces[2][1] == pytest.approx(0.33 * 20000.0, rel=1e-9)
        assert forces[1][2] == pytest.approx(1.33 * 20000.0 / 2, rel=1e-9)

    def test_braked_roll_no_nose_drag(self):
        """§23.493: 드래그 0.8은 브레이크 휠(메인)에만."""
        cond = LandingCondition(
            condition_type=LandingConditionType.BRAKED_ROLL,
            nz_cg=1.33, weight_cg=self._wc(),
            drag_factor=0.8, nose_drag_factor=0.0,
            main_gear_vertical_frac=0.8, nose_gear_vertical_frac=0.2)
        forces = compute_gear_reactions(cond, self._gear())
        assert forces[1][0] < 0            # 메인 드래그 (후방)
        assert forces[3][0] == 0.0         # 노즈 드래그 없음

    def test_nose_supplementary(self):
        """§23.499: 수직 2.25×정적 + aft 0.8V / fwd 0.4V / side 0.7V."""
        wc = self._wc()
        v_static = wc.weight_N * 0.2
        for sup, comp, sign in (("aft", 0, -0.8), ("fwd", 0, +0.4),
                                ("side", 1, 0.7)):
            cond = LandingCondition(
                condition_type=LandingConditionType.NOSE_WHEEL_YAW,
                nz_cg=1.0, weight_cg=wc, nose_supp_dir=sup,
                nose_gear_vertical_frac=0.2)
            forces = compute_gear_reactions(cond, self._gear())
            assert forces[3][2] == pytest.approx(2.25 * v_static, rel=1e-9)
            assert forces[3][comp] == pytest.approx(
                sign * 2.25 * v_static, rel=1e-9)

    def test_landing_case_self_equilibrium_fz(self):
        """기어 + 관성 + 양력의 ΣFz = 0 (릴리프 이전에도 수직은 닫힘)."""
        from nastaero.loads_analysis.certification.landing_loads import (
            compute_landing_lift_forces,
        )

        wc = self._wc()
        cond = LandingCondition(nz_cg=3.0, weight_cg=wc,
                                lift_factor=2 / 3,
                                main_gear_vertical_frac=1.0)
        g = 9810.0
        masses = {10: wc.weight_N / g / 2, 11: wc.weight_N / g / 2}
        gear_f = compute_gear_reactions(cond, self._gear())
        inertial = compute_landing_inertial_forces(cond, masses, g=g)
        lift = compute_landing_lift_forces(cond, masses, g=g)
        total = combine_forces(gear_f, inertial, lift)
        fz = sum(f[2] for f in total.values())
        assert fz == pytest.approx(0.0, abs=1e-6)

    def test_spinup_drag_factor_interpolation(self):
        from nastaero.loads_analysis.certification.landing_loads import (
            spinup_drag_factor,
        )

        assert spinup_drag_factor(3000 * 4.44822) == pytest.approx(0.25)
        assert spinup_drag_factor(6000 * 4.44822) == pytest.approx(0.33)
        assert spinup_drag_factor(4500 * 4.44822) == pytest.approx(0.29)

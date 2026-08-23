"""End-to-end certification pipeline tests using GACOMP aircraft model.

Validates the entire FAA Part 23 certification loads analysis pipeline
against a real aircraft FEM model (제작사 GACOMP, P400C configuration):

  BDF Parse → AircraftConfig → V-n Diagram → Load Case Matrix
  → BatchRunner (SOL 144 trim) → VMT internal loads
  → Envelope Processing → Critical Case Identification → Report

GACOMP model specifications
----------------------------
- 22,640 GRID points, 27,016 elements (CQUAD4/CBAR/CTRIA3/RBE)
- 18,406 CONM2 mass entries, total ≈ 1,289 kg (≈ 2,842 lb)
- 44 CAERO1 panels → 783 aero boxes (VLM horseshoe vortices)
- 44 SPLINE1 entries for structure–aero coupling
- Control surfaces: ARON (aileron), ELEV (elevator), RUD (rudder)
- Unit system: N-mm-sec (mass in Mg = tonne)

These tests are marked ``@pytest.mark.slow`` because a single
SOL 144 subcase takes ~17 s on this model.
"""
import math
import os
import pytest
import numpy as np

from nastaero.bdf.parser import parse_bdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
GACOMP_BDF = os.path.join(VALIDATION_DIR, "GACOMP", "p400r3-free-trim.bdf")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(VALIDATION_DIR, "GACOMP")),
    reason="comparison-model data not present in this archive")

# Skip the entire module if BULK data is missing
_bulk_ok = os.path.exists(
    os.path.join(VALIDATION_DIR, "GACOMP", "BULK",
                 "p400r3-struct-asym_flex.dat"))
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _bulk_ok,
                       reason="GACOMP BULK data not available"),
]

# ---------------------------------------------------------------------------
# GACOMP physical constants (N-mm-sec)
# ---------------------------------------------------------------------------
GACOMP_TOTAL_MASS_MG = 1.2889        # Mg (tonne) — from CONM2 sum
GACOMP_TOTAL_MASS_KG = 1288.9        # kg
GACOMP_WEIGHT_N = GACOMP_TOTAL_MASS_KG * 9.80665   # ≈ 12,640 N
GACOMP_WEIGHT_LB = GACOMP_TOTAL_MASS_KG * 2.20462  # ≈ 2,842 lb
GACOMP_CG_X_MM = 3882.0              # mm — mass-weighted CG
GACOMP_HALF_SPAN_MM = 5616.6         # mm — from CAERO1 Y range


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gacomp_model():
    """Parse GACOMP BDF once for the entire module."""
    return parse_bdf(GACOMP_BDF)


def _make_gacomp_config():
    """Create AircraftConfig matching the GACOMP model.

    Speed schedule is based on GACOMP public performance data:
      VS1 ≈ 33 m/s (64 kt), VA ≈ 62 m/s, VC ≈ 80 m/s, VD ≈ 100 m/s
    Wing area ≈ 17.0 m² (from 제작사 GACOMP specs).
    """
    from nastaero.loads_analysis.certification.aircraft_config import (
        AircraftConfig, SpeedSchedule, WeightCGCondition,
        ControlSurfaceLimits, LandingGearConfig,
    )
    return AircraftConfig(
        speeds=SpeedSchedule(
            VS1=33.0,    # m/s — stall speed (clean)
            VA=62.0,     # m/s — manoeuvring speed
            VB=0.0,      # auto-compute from gust intersection
            VC=80.0,     # m/s — cruise speed
            VD=100.0,    # m/s — dive speed
            VF=40.0,     # m/s — flap speed
        ),
        weight_cg_conditions=[
            WeightCGCondition(
                label="MTOW",
                weight_N=GACOMP_WEIGHT_N,
                cg_x=GACOMP_CG_X_MM,
            ),
        ],
        altitudes_m=[0.0],     # sea level only for speed
        wing_area_m2=17.0,     # GACOMP wing area
        CLalpha=5.5,           # typical for GA wing
        mean_chord_m=1.6,      # MAC from wing geometry
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0,
            rudder_max_deg=25.0,
            elevator_max_deg=25.0,
        ),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[100, 101],
            nose_gear_node_ids=[102],
            main_gear_x=4200.0,    # mm — approx main gear position
            nose_gear_x=1500.0,    # mm — approx nose gear position
            strut_efficiency=0.7,
            stroke=0.25,           # m — typical GA shock strut
            sink_rate_fps=10.0,    # ft/s per §23.473(d)
        ),
        gust_Ude_VC_fps=50.0,     # §23.333(c)
        gust_Ude_VD_fps=25.0,     # §23.333(c)
    )


# ===================================================================
# 1. BDF Parsing Validation
# ===================================================================

class TestGACOMPParsing:
    """Verify that the full GACOMP model is parsed correctly."""

    def test_node_count(self, gacomp_model):
        assert len(gacomp_model.nodes) == 22640

    def test_element_count(self, gacomp_model):
        # 27016 -> 27020: 대필드 CBAR*의 소필드 오프셋 연속행을 행별 형식
        # 규칙으로 파싱하게 되면서 누락되던 CBAR 4개가 복원됨
        assert len(gacomp_model.elements) == 27020

    def test_mass_count(self, gacomp_model):
        assert len(gacomp_model.masses) > 18000

    def test_aero_panels(self, gacomp_model):
        assert len(gacomp_model.caero_panels) == 44

    def test_splines(self, gacomp_model):
        assert len(gacomp_model.splines) == 44

    def test_trim_cards(self, gacomp_model):
        assert len(gacomp_model.trims) == 7

    def test_subcases(self, gacomp_model):
        assert len(gacomp_model.subcases) == 7

    def test_total_mass(self, gacomp_model):
        """Total CONM2 mass ≈ 1.289 Mg (N-mm-sec)."""
        total = sum(m.mass for m in gacomp_model.masses.values())
        assert abs(total - GACOMP_TOTAL_MASS_MG) < 0.01

    def test_control_surfaces(self, gacomp_model):
        """Model has ARON, ELEV, RUD control surfaces."""
        labels = {a.label for a in gacomp_model.aesurfs.values()}
        assert 'ARON' in labels
        assert 'ELEV' in labels
        assert 'RUD' in labels


# ===================================================================
# 2. V-n Diagram with GACOMP Parameters
# ===================================================================

class TestGACOMPVnDiagram:
    """V-n diagram generation with real GACOMP weight."""

    def test_part23_nz(self):
        """nz_max = min(3.8, 2.1 + 24000/W_lb) for GACOMP."""
        from nastaero.loads_analysis.certification.aircraft_config import (
            part23_nz_max, part23_nz_min,
        )
        nz_max = part23_nz_max(GACOMP_WEIGHT_N)
        assert nz_max == pytest.approx(3.8, abs=0.01)
        assert part23_nz_min(nz_max) == pytest.approx(-1.52, abs=0.01)

    def test_compute_vn_diagram(self):
        """V-n diagram produces corner points with GACOMP data."""
        from nastaero.loads_analysis.certification.vn_diagram import (
            compute_vn_diagram,
        )
        config = _make_gacomp_config()
        wc = config.weight_cg_conditions[0]
        vn = compute_vn_diagram(config, wc, altitude_m=0.0)

        assert len(vn.corner_points) >= 8
        assert vn.nz_max == pytest.approx(3.8, abs=0.01)
        assert vn.nz_min == pytest.approx(-1.52, abs=0.01)

        labels = {p.label for p in vn.corner_points}
        # Must contain positive/negative maneuver and gust points
        assert any('A' in lbl for lbl in labels)
        assert any('D' in lbl for lbl in labels)


# ===================================================================
# 3. Load Case Matrix Generation
# ===================================================================

class TestGACOMPLoadCaseMatrix:
    """Load case matrix with GACOMP parameters."""

    def test_matrix_generation(self):
        """Generate full load case matrix."""
        from nastaero.loads_analysis.certification.load_case_matrix import (
            LoadCaseMatrix,
        )
        config = _make_gacomp_config()
        matrix = LoadCaseMatrix(config)
        matrix.generate_all()

        assert matrix.total_cases > 30
        assert len(matrix.flight_cases) > 20
        assert len(matrix.landing_cases) > 5

    def test_far_coverage(self):
        """Matrix covers key FAR sections."""
        from nastaero.loads_analysis.certification.load_case_matrix import (
            LoadCaseMatrix,
        )
        config = _make_gacomp_config()
        matrix = LoadCaseMatrix(config)
        matrix.generate_all()

        # Collect FAR sections from all cases
        far_sections = set()
        for c in matrix.flight_cases:
            far_sections.add(c.far_section)
        for c in matrix.landing_cases:
            far_sections.add(c.far_section)
        # Must cover maneuver, gust, landing sections
        assert any('23.337' in s for s in far_sections)  # maneuvering
        assert any('23.341' in s or '23.333' in s for s in far_sections)  # gust
        assert any('23.479' in s or '23.481' in s for s in far_sections)  # landing


# ===================================================================
# 4. SOL 144 Trim — Single Subcase
# ===================================================================

class TestGACOMPTrim:
    """Run real SOL 144 trim on GACOMP (single subcase)."""

    def test_trim_1g_level_flight(self, gacomp_model):
        """Subcase 1 (M=0.1): trim converges with physical results."""
        from nastaero.solvers.sol144 import solve_trim

        # Keep only subcase 1 for speed
        model = gacomp_model
        orig_subcases = list(model.subcases)
        orig_trims = dict(model.trims)
        model.subcases = [orig_subcases[0]]
        model.trims = {1: orig_trims[1]}

        try:
            result = solve_trim(model)
            assert len(result.subcases) == 1

            sc = result.subcases[0]
            assert sc.subcase_id == 1

            # Trim variables must exist
            assert 'ANGLEA' in sc.trim_variables
            assert 'ELEV' in sc.trim_variables

            # Alpha should be positive (nose up for lift)
            alpha = sc.trim_variables['ANGLEA']
            assert alpha > 0, f"Alpha should be positive, got {alpha}"

            # Check aero/inertial force balance
            aero_fz = sum(f[2] for f in sc.nodal_aero_forces.values())
            inertial_fz = sum(f[2] for f in sc.nodal_inertial_forces.values())

            # Aero Fz should be positive (upward lift)
            assert aero_fz > 0, f"Aero lift should be positive, got {aero_fz}"
            # Inertial Fz should be negative (weight downward)
            assert inertial_fz < 0, f"Inertial should be negative, got {inertial_fz}"

            # Lift ≈ Weight (within ~20% for trim)
            ratio = abs(aero_fz / inertial_fz)
            assert 0.7 < ratio < 1.3, \
                f"Lift/Weight ratio out of range: {ratio:.2f}"

        finally:
            # Restore full model
            model.subcases = orig_subcases
            model.trims = orig_trims


# ===================================================================
# 5. Full Pipeline (placeholder solver)
# ===================================================================

class TestGACOMPPipelinePlaceholder:
    """End-to-end pipeline without real solver (placeholder mode).

    Tests that all pipeline stages connect correctly with GACOMP config.
    """

    def test_config_to_matrix(self):
        """Config → V-n → Matrix → Batch (placeholder) → Envelope → Report."""
        from nastaero.loads_analysis.certification.vn_diagram import (
            compute_vn_diagram,
        )
        from nastaero.loads_analysis.certification.load_case_matrix import (
            LoadCaseMatrix,
        )
        from nastaero.loads_analysis.certification.batch_runner import (
            BatchRunner,
        )
        from nastaero.loads_analysis.certification.envelope import (
            EnvelopeProcessor,
        )
        from nastaero.loads_analysis.certification.report import (
            CertificationReport,
        )

        config = _make_gacomp_config()
        wc = config.weight_cg_conditions[0]

        # V-n diagram
        vn = compute_vn_diagram(config, wc, altitude_m=0.0)
        assert len(vn.corner_points) >= 8

        # Load case matrix
        matrix = LoadCaseMatrix(config)
        matrix.generate_all()
        assert matrix.total_cases > 30

        # Batch runner (no BDF model → placeholder results)
        runner = BatchRunner(matrix, bdf_model=None)
        batch_result = runner.run()
        assert batch_result.n_total == matrix.total_cases

        # Envelope processing
        proc = EnvelopeProcessor(batch_result)
        proc.compute_envelopes()
        proc.identify_critical_cases()

        env_summary = proc.summary()
        assert env_summary['n_critical'] >= 0

        # Report generation
        report = CertificationReport(matrix, batch_result, proc)
        rep_summary = report.summary()

        assert rep_summary['total_cases'] == matrix.total_cases
        assert rep_summary['far_sections_covered'] > 0
        assert 0.0 <= rep_summary['compliance_rate'] <= 1.0

    def test_csv_roundtrip(self, tmp_path):
        """Matrix CSV export/import and critical loads CSV."""
        from nastaero.loads_analysis.certification.load_case_matrix import (
            LoadCaseMatrix,
        )
        from nastaero.loads_analysis.certification.batch_runner import (
            BatchRunner,
        )
        from nastaero.loads_analysis.certification.envelope import (
            EnvelopeProcessor,
        )
        from nastaero.loads_analysis.certification.report import (
            CertificationReport,
        )

        config = _make_gacomp_config()
        matrix = LoadCaseMatrix(config)
        matrix.generate_all()

        # Matrix CSV
        csv_path = str(tmp_path / "matrix.csv")
        matrix.to_csv(csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        # CSV header + flight_cases rows (landing cases may not be in CSV)
        assert len(lines) >= len(matrix.flight_cases) + 1  # header + data

        # Critical loads CSV
        runner = BatchRunner(matrix, bdf_model=None)
        batch_result = runner.run()
        proc = EnvelopeProcessor(batch_result)
        proc.compute_envelopes()
        proc.identify_critical_cases()
        report = CertificationReport(matrix, batch_result, proc)

        crit_csv = str(tmp_path / "critical.csv")
        report.to_csv(crit_csv)
        assert os.path.exists(crit_csv)

        comp_csv = str(tmp_path / "compliance.csv")
        report.compliance_to_csv(comp_csv)
        assert os.path.exists(comp_csv)


# ===================================================================
# 6. Full E2E with Real Solver (1 subcase)
# ===================================================================

class TestGACOMPE2ERealSolver:
    """End-to-end with real SOL 144 solver on GACOMP (1 subcase).

    This is the ultimate integration test: real BDF model → real solver
    → real VMT loads → envelope → report.
    """

    def test_single_subcase_full_pipeline(self, gacomp_model, tmp_path):
        """Full pipeline with 1 real trim subcase on GACOMP."""
        from nastaero.solvers.sol144 import solve_trim
        from nastaero.loads_analysis.certification.aircraft_config import (
            WeightCGCondition,
        )
        from nastaero.loads_analysis.certification.vn_diagram import (
            compute_vn_diagram,
        )
        from nastaero.loads_analysis.certification.load_case_matrix import (
            LoadCaseMatrix,
        )
        from nastaero.loads_analysis.certification.batch_runner import (
            BatchRunner, BatchResult, CaseResult,
        )
        from nastaero.loads_analysis.certification.envelope import (
            EnvelopeProcessor,
        )
        from nastaero.loads_analysis.certification.report import (
            CertificationReport,
        )

        config = _make_gacomp_config()
        wc = config.weight_cg_conditions[0]

        # ---- Step 1: V-n diagram ----
        vn = compute_vn_diagram(config, wc, altitude_m=0.0)
        assert len(vn.corner_points) >= 8

        # ---- Step 2: Solve 1 real trim subcase ----
        model = gacomp_model
        orig_subcases = list(model.subcases)
        orig_trims = dict(model.trims)
        model.subcases = [orig_subcases[0]]
        model.trims = {1: orig_trims[1]}

        try:
            result = solve_trim(model)
        finally:
            model.subcases = orig_subcases
            model.trims = orig_trims

        sc = result.subcases[0]

        # ---- Step 3: Build synthetic BatchResult from real solver result ----
        # Use the real nodal forces from the trim solution
        case_results = []

        # Real 1G level flight case
        case_results.append(CaseResult(
            case_id=1,
            converged=True,
            nodal_forces=dict(sc.nodal_combined_forces),
            category="symmetric",
            far_section="§23.337",
            nz=1.0,
            label="GACOMP_1G_M0.1",
            weight_label=wc.label,
        ))

        # Create scaled versions for nz_max and nz_min
        for nz_val, lbl, cid in [
            (3.8, "GACOMP_nzmax_M0.1", 2),
            (-1.52, "GACOMP_nzmin_M0.1", 3),
        ]:
            scaled_forces = {}
            for nid, f in sc.nodal_combined_forces.items():
                scaled_forces[nid] = f * nz_val
            case_results.append(CaseResult(
                case_id=cid,
                converged=True,
                nodal_forces=scaled_forces,
                category="symmetric",
                far_section="§23.337",
                nz=nz_val,
                label=lbl,
                weight_label=wc.label,
            ))

        batch_result = BatchResult(
            case_results=case_results,
            wall_time_s=17.0,
        )

        # ---- Step 4: VMT bridge ----
        from nastaero.loads_analysis.certification.vmt_bridge import (
            compute_vmt_for_batch,
        )
        vmt_data = compute_vmt_for_batch(model, batch_result)

        # Must produce VMT data for all 3 converged cases
        assert len(vmt_data) == 3, (
            f"Expected VMT data for 3 cases, got {len(vmt_data)}")

        # Each case should have at least 1 structural component
        for cid, components in vmt_data.items():
            assert len(components) > 0, (
                f"Case {cid}: no VMT components computed")
            for comp_name, data in components.items():
                assert "stations" in data
                assert "shear" in data
                assert "bending" in data
                assert "torsion" in data
                assert len(data["stations"]) > 0

        # ---- Step 5: Envelope processing with real VMT ----
        proc = EnvelopeProcessor(batch_result, vmt_data)
        proc.compute_envelopes()
        proc.identify_critical_cases()

        env_summary = proc.summary()
        # With real forces + VMT, should have envelopes
        assert len(env_summary['components']) > 0, \
            "Should have envelope components with real VMT data"

        # ---- Step 6: Report (with matrix from placeholder) ----
        matrix = LoadCaseMatrix(config)
        matrix.generate_all()

        report = CertificationReport(matrix, batch_result, proc)
        rep_summary = report.summary()
        assert rep_summary['total_cases'] > 0

        # ---- Step 7: Save outputs ----
        crit_csv = str(tmp_path / "critical_loads.csv")
        report.to_csv(crit_csv)
        assert os.path.exists(crit_csv)

        # ---- Verify physical reasonableness ----
        # nz=3.8 케이스의 하중 크기가 nz=1.0보다 커야 한다.
        # 합산 하중은 관성해제로 정확히 자기평형이므로 성분 합은
        # 두 경우 모두 0이다(그 합을 비교하면 수치 잡음을 비교하게
        # 된다). 크기 척도로 비교한다.
        f1g = case_results[0].nodal_forces
        f3g = case_results[1].nodal_forces
        mag_fz_1g = sum(abs(f[2]) for f in f1g.values())
        mag_fz_3g = sum(abs(f[2]) for f in f3g.values())
        assert mag_fz_3g > mag_fz_1g, \
            "3.8g forces should exceed 1g forces"

        # 자기평형 확인: 두 경우 모두 합력이 0이어야 한다
        for tag, ff in (("1g", f1g), ("3.8g", f3g)):
            scale = max(sum(abs(f[2]) for f in ff.values()), 1.0)
            assert abs(sum(f[2] for f in ff.values())) / scale < 1e-9, \
                f"{tag} 합산 하중이 자기평형이 아니다"

        # ---- Verify VMT physical reasonableness ----
        # For any wing-like component, nz=3.8 should produce
        # larger shear/bending than nz=1.0
        if 1 in vmt_data and 2 in vmt_data:
            for comp_name in vmt_data[1]:
                if comp_name in vmt_data[2]:
                    s1 = np.max(np.abs(vmt_data[1][comp_name]["shear"]))
                    s38 = np.max(np.abs(vmt_data[2][comp_name]["shear"]))
                    if s1 > 0:
                        assert s38 > s1, (
                            f"{comp_name}: 3.8g shear ({s38:.0f}) should "
                            f"exceed 1g shear ({s1:.0f})")

    def test_critical_design_load_selection(self, gacomp_model, tmp_path):
        """Final loads step: potato-plot critical selection + FORCE cards.

        Solves one real GACOMP trim subcase, fans it into a small batch of
        scaled cases, then runs ``select_critical_design_loads`` to reduce
        the batch to a compact design-load set and export FORCE-card BDFs
        applyable to the full-vehicle mesh.
        """
        from nastaero.solvers.sol144 import solve_trim
        from nastaero.loads_analysis.certification.batch_runner import (
            BatchResult, CaseResult,
        )
        from nastaero.loads_analysis.certification.envelope import (
            select_critical_design_loads, DesignLoadCase,
        )

        model = gacomp_model
        orig_subcases = list(model.subcases)
        orig_trims = dict(model.trims)
        model.subcases = [orig_subcases[0]]
        model.trims = {1: orig_trims[1]}
        try:
            result = solve_trim(model)
        finally:
            model.subcases = orig_subcases
            model.trims = orig_trims
        sc = result.subcases[0]

        # Fan the real 1-g forces into a batch of scaled maneuver/gust cases
        base = dict(sc.nodal_combined_forces)
        specs = [
            (1.0, "symmetric", "§23.337", "1g"),
            (3.8, "symmetric", "§23.337", "nz_max"),
            (-1.52, "symmetric", "§23.337", "nz_min"),
            (2.5, "gust", "§23.341", "gust_up"),
            (-1.0, "gust", "§23.341", "gust_dn"),
            (2.0, "rolling", "§23.349", "roll"),
        ]
        case_results = []
        for i, (nz, cat, far, tag) in enumerate(specs, start=1):
            case_results.append(CaseResult(
                case_id=i, converged=True,
                nodal_forces={nid: f * nz for nid, f in base.items()},
                category=cat, far_section=far, nz=nz, label=f"GACOMP_{tag}",
            ))
        batch_result = BatchResult(case_results=case_results, wall_time_s=1.0)

        out = str(tmp_path / "design_loads")
        summary = select_critical_design_loads(
            model, batch_result, output_dir=out,
            n_stations=20, fuselage_cg_x=GACOMP_CG_X_MM,
        )

        # Multiple structural components enveloped (wing + tails + fuselage)
        assert len(summary["components"]) >= 2

        # Reduction produced a compact, deduplicated design-load set
        design = summary["design_cases"]
        assert 0 < summary["n_design_cases"] <= summary["n_cases_in"]
        assert all(isinstance(d, DesignLoadCase) for d in design)
        ids = [d.case_id for d in design]
        assert len(ids) == len(set(ids))
        # ranked by governed extremes, descending
        ng = [d.n_govern for d in design]
        assert ng == sorted(ng, reverse=True)

        # FORCE-card BDFs written, one per design case, + master + summary
        exp = summary["export"]
        assert exp["n_cases"] == summary["n_design_cases"]
        assert exp["n_force_cards"] > 0
        assert os.path.exists(exp["master_bdf"])
        assert os.path.exists(exp["summary_csv"])
        for fname in exp["case_files"].values():
            assert os.path.exists(os.path.join(out, fname))
        # a real FORCE card is present in an exported deck
        any_bdf = os.path.join(out, next(iter(exp["case_files"].values())))
        with open(any_bdf) as fh:
            assert "FORCE" in fh.read()

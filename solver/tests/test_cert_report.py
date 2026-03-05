"""Tests for Phase 6: Certification report and visualization.

Tests report generation, compliance matrix, CSV export, and
visualization functions for the certification loads framework.
"""
import os
import pytest
import numpy as np

from nastaero.loads_analysis.certification.aircraft_config import (
    AircraftConfig, SpeedSchedule, WeightCGCondition,
    ControlSurfaceLimits, LandingGearConfig,
)
from nastaero.loads_analysis.certification.load_case_matrix import (
    LoadCaseMatrix,
)
from nastaero.loads_analysis.certification.batch_runner import (
    BatchRunner, BatchResult, CaseResult,
)
from nastaero.loads_analysis.certification.envelope import (
    EnvelopeProcessor, ComponentEnvelope, StationEnvelope,
)
from nastaero.loads_analysis.certification.report import (
    CertificationReport, CriticalLoadsRow, ComplianceEntry,
    FAR_SECTIONS,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_test_config():
    """Create test aircraft configuration."""
    W1 = 12000.0 * 9.80665

    return AircraftConfig(
        speeds=SpeedSchedule(
            VS1=30.0, VA=60.0, VB=0.0, VC=80.0, VD=100.0, VF=40.0,
        ),
        weight_cg_conditions=[
            WeightCGCondition(label="MTOW Fwd", weight_N=W1, cg_x=5000.0),
        ],
        altitudes_m=[0.0],
        wing_area_m2=20.0,
        CLalpha=5.5,
        mean_chord_m=1.5,
        ctrl_limits=ControlSurfaceLimits(),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[100, 101],
            nose_gear_node_ids=[102],
            main_gear_x=5200.0,
            nose_gear_x=2000.0,
            strut_efficiency=0.7,
            stroke=0.3,
            sink_rate_fps=10.0,
        ),
    )


def make_full_setup():
    """Create matrix + batch results + envelope for testing."""
    config = make_test_config()
    matrix = LoadCaseMatrix(config)
    matrix.generate_all()

    runner = BatchRunner(matrix, bdf_model=None)
    batch_result = runner.run()

    # Create synthetic VMT data
    stations = np.array([0.0, 2500.0, 5000.0, 7500.0, 10000.0])
    vmt_data = {}

    for r in batch_result.case_results:
        nz = r.nz if r.nz != 0 else 1.0
        vmt_data[r.case_id] = {
            "Wing": {
                "stations": stations,
                "shear": np.array([50000, 40000, 30000, 15000, 0]) * abs(nz),
                "bending": np.array([0, 5e7, 8e7, 9e7, 1e8]) * abs(nz),
                "torsion": np.array([1e6, 8e5, 5e5, 2e5, 0]) * abs(nz),
            },
        }

    proc = EnvelopeProcessor(batch_result, vmt_data)
    proc.compute_envelopes()
    proc.identify_critical_cases()

    return matrix, batch_result, proc


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestCertificationReport:
    """Test CertificationReport class."""

    @pytest.fixture
    def report_setup(self):
        matrix, batch_result, proc = make_full_setup()
        return CertificationReport(matrix, batch_result, proc)

    def test_summary(self, report_setup):
        """Summary contains all expected keys."""
        summary = report_setup.summary()
        assert "total_cases" in summary
        assert "flight_cases" in summary
        assert "landing_cases" in summary
        assert "converged" in summary
        assert "far_sections_covered" in summary
        assert "compliance_rate" in summary
        assert "lateral_maneuvers" in summary
        assert "landing_loads" in summary

    def test_total_cases(self, report_setup):
        """Total cases count is correct."""
        summary = report_setup.summary()
        assert summary["total_cases"] > 0
        assert summary["flight_cases"] > 0
        assert summary["landing_cases"] > 0

    def test_compliance_rate(self, report_setup):
        """Compliance rate is between 0 and 1."""
        summary = report_setup.summary()
        assert 0.0 <= summary["compliance_rate"] <= 1.0

    def test_far_sections_covered(self, report_setup):
        """At least some FAR sections are covered."""
        summary = report_setup.summary()
        assert summary["far_sections_covered"] > 0


class TestRegulatoryCompliance:
    """Test regulatory compliance matrix."""

    @pytest.fixture
    def report(self):
        matrix, batch_result, proc = make_full_setup()
        return CertificationReport(matrix, batch_result, proc)

    def test_compliance_entries(self, report):
        """Compliance matrix has entries for all FAR sections."""
        compliance = report.regulatory_compliance_matrix()
        assert len(compliance) == len(FAR_SECTIONS)

    def test_key_sections_covered(self, report):
        """Key flight load sections are covered."""
        compliance = report.regulatory_compliance_matrix()
        covered = {e.section for e in compliance if e.status == "covered"}
        # These must be covered by any complete matrix
        assert "§23.337" in covered  # Maneuver
        assert "§23.341" in covered  # Gust
        assert "§23.349" in covered  # Rolling
        assert "§23.351" in covered  # Yaw

    def test_landing_sections_covered(self, report):
        """Key landing sections are covered."""
        compliance = report.regulatory_compliance_matrix()
        covered = {e.section for e in compliance if e.status == "covered"}
        assert "§23.479" in covered  # Level landing
        assert "§23.481" in covered  # Tail-down

    def test_section_case_counts(self, report):
        """Covered sections have positive case counts."""
        compliance = report.regulatory_compliance_matrix()
        for e in compliance:
            if e.status == "covered":
                assert e.n_cases > 0


class TestCriticalLoadsTable:
    """Test critical loads table generation."""

    @pytest.fixture
    def report(self):
        matrix, batch_result, proc = make_full_setup()
        return CertificationReport(matrix, batch_result, proc)

    def test_has_rows(self, report):
        """Critical loads table is populated."""
        rows = report.critical_loads_table()
        assert len(rows) > 0

    def test_row_structure(self, report):
        """Each row has component, station, quantity."""
        rows = report.critical_loads_table()
        for row in rows:
            assert row.component != ""
            assert row.quantity in ["V", "M", "T"]

    def test_filter_by_component(self, report):
        """Can filter to specific components."""
        all_rows = report.critical_loads_table()
        wing_rows = report.critical_loads_table(components=["Wing"])
        assert len(wing_rows) <= len(all_rows)
        assert all(r.component == "Wing" for r in wing_rows)

    def test_max_greater_than_min(self, report):
        """Max values are >= min values."""
        rows = report.critical_loads_table()
        for row in rows:
            assert row.max_value >= row.min_value


class TestLateralManeuverSummary:
    """Test lateral maneuver summary."""

    @pytest.fixture
    def report(self):
        matrix, batch_result, proc = make_full_setup()
        return CertificationReport(matrix, batch_result, proc)

    def test_has_rolling_and_yaw(self, report):
        """Summary has both rolling and yaw sections."""
        lat = report.lateral_maneuver_summary()
        assert "rolling" in lat
        assert "yaw" in lat

    def test_rolling_cases_counted(self, report):
        """Rolling cases are counted."""
        lat = report.lateral_maneuver_summary()
        assert lat["rolling"]["n_cases"] > 0

    def test_yaw_cases_counted(self, report):
        """Yaw cases are counted."""
        lat = report.lateral_maneuver_summary()
        assert lat["yaw"]["n_cases"] > 0


class TestLandingLoadsSummary:
    """Test landing loads summary."""

    @pytest.fixture
    def report(self):
        matrix, batch_result, proc = make_full_setup()
        return CertificationReport(matrix, batch_result, proc)

    def test_has_landing_cases(self, report):
        """Landing summary counts cases."""
        landing = report.landing_loads_summary()
        assert landing["n_cases"] > 0

    def test_landing_far_sections(self, report):
        """Landing summary lists FAR sections."""
        landing = report.landing_loads_summary()
        assert len(landing["far_sections"]) > 0


# ---------------------------------------------------------------------------
# CSV export tests
# ---------------------------------------------------------------------------

class TestCSVExport:
    """Test CSV export functions."""

    def test_critical_loads_csv(self, tmp_path):
        """Export critical loads table to CSV."""
        matrix, batch_result, proc = make_full_setup()
        report = CertificationReport(matrix, batch_result, proc)

        filepath = str(tmp_path / "critical_loads.csv")
        report.to_csv(filepath)
        assert os.path.exists(filepath)

        # Read back and verify
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0
        assert "Component" in rows[0]
        assert "Max Value" in rows[0]

    def test_compliance_csv(self, tmp_path):
        """Export compliance matrix to CSV."""
        matrix, batch_result, proc = make_full_setup()
        report = CertificationReport(matrix, batch_result, proc)

        filepath = str(tmp_path / "compliance.csv")
        report.compliance_to_csv(filepath)
        assert os.path.exists(filepath)

        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(FAR_SECTIONS)


# ---------------------------------------------------------------------------
# Visualization tests (smoke tests)
# ---------------------------------------------------------------------------

class TestCertVisualization:
    """Smoke tests for cert visualization functions."""

    def test_plot_potato(self, tmp_path):
        """Potato plot generates without error."""
        from nastaero.visualization.cert_plot import plot_potato
        from nastaero.loads_analysis.certification.envelope import PotatoData

        potato = PotatoData(
            station=5000.0,
            component="Wing",
            x_values=[100, 200, -50, 150, -100],
            y_values=[1e6, 2e6, -5e5, 1.5e6, -1e6],
            categories=["symmetric", "gust", "symmetric", "rolling", "landing"],
            x_label="Shear V (N)",
            y_label="Bending M (N-mm)",
        )

        path = plot_potato(potato, output_path=str(tmp_path / "potato.png"))
        assert os.path.exists(path)

    def test_plot_vmt_envelope(self, tmp_path):
        """VMT envelope plot generates without error."""
        from nastaero.visualization.cert_plot import plot_vmt_envelope

        env = ComponentEnvelope(
            component="Wing",
            stations=[0, 2500, 5000, 7500, 10000],
            envelopes=[
                StationEnvelope(station=s,
                                 V_max=50000 * (1 - s/10000),
                                 V_min=-20000 * (1 - s/10000),
                                 M_max=1e8 * s/10000,
                                 M_min=-5e7 * s/10000,
                                 T_max=2e6 * (1 - s/10000),
                                 T_min=-1e6 * (1 - s/10000))
                for s in [0, 2500, 5000, 7500, 10000]
            ],
            n_cases=50,
        )

        path = plot_vmt_envelope(env, output_path=str(tmp_path / "vmt.png"))
        assert os.path.exists(path)

    def test_plot_critical_frequency(self, tmp_path):
        """Critical frequency plot generates without error."""
        from nastaero.visualization.cert_plot import plot_critical_frequency

        freq = {1: 12, 2: 8, 3: 15, 4: 5, 5: 3}
        path = plot_critical_frequency(
            freq, output_path=str(tmp_path / "freq.png"))
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# FAR sections reference tests
# ---------------------------------------------------------------------------

class TestFARSections:
    """Test FAR sections reference data."""

    def test_sections_exist(self):
        """FAR sections dict is populated."""
        assert len(FAR_SECTIONS) >= 15

    def test_key_sections(self):
        """All key sections are defined."""
        for sec in ["§23.337", "§23.341", "§23.349", "§23.351",
                     "§23.479", "§23.481", "§23.483"]:
            assert sec in FAR_SECTIONS

    def test_sections_have_titles(self):
        """All sections have non-empty titles."""
        for sec, title in FAR_SECTIONS.items():
            assert title != "", f"{sec} has empty title"

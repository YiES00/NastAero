"""Tests for Phase 4: BatchRunner with checkpointing.

Tests the batch execution engine for certification load cases,
including flight case grouping, landing case execution, checkpointing,
and result aggregation.
"""
import json
import os
import pytest
import numpy as np

from nastaero.loads_analysis.certification.aircraft_config import (
    AircraftConfig, SpeedSchedule, WeightCGCondition,
    ControlSurfaceLimits, LandingGearConfig,
)
from nastaero.loads_analysis.certification.load_case_matrix import (
    LoadCaseMatrix, CertLoadCase,
)
from nastaero.loads_analysis.certification.batch_runner import (
    BatchRunner, BatchResult, CaseResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_test_config():
    """Create test aircraft configuration."""
    W1 = 12000.0 * 9.80665
    W2 = 8000.0 * 9.80665

    return AircraftConfig(
        speeds=SpeedSchedule(
            VS1=30.0, VA=60.0, VB=0.0, VC=80.0, VD=100.0, VF=40.0,
        ),
        weight_cg_conditions=[
            WeightCGCondition(label="MTOW Fwd", weight_N=W1, cg_x=5000.0),
            WeightCGCondition(label="MLW Aft", weight_N=W2, cg_x=5500.0),
        ],
        altitudes_m=[0.0],
        wing_area_m2=20.0,
        CLalpha=5.5,
        mean_chord_m=1.5,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0,
            rudder_max_deg=25.0,
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


def make_matrix():
    """Create and populate a test load case matrix."""
    config = make_test_config()
    m = LoadCaseMatrix(config)
    m.generate_all()
    return m


# ---------------------------------------------------------------------------
# CaseResult tests
# ---------------------------------------------------------------------------

class TestCaseResult:
    """Test CaseResult dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        r = CaseResult()
        assert r.case_id == 0
        assert r.converged is False
        assert r.nodal_forces is None

    def test_with_nodal_forces(self):
        """Can store nodal forces."""
        forces = {1: np.array([100, 0, -500, 0, 0, 0], dtype=float)}
        r = CaseResult(case_id=42, converged=True, nodal_forces=forces)
        assert r.case_id == 42
        assert r.converged is True
        assert 1 in r.nodal_forces


# ---------------------------------------------------------------------------
# BatchResult tests
# ---------------------------------------------------------------------------

class TestBatchResult:
    """Test BatchResult container."""

    def test_n_converged(self):
        """Count converged cases."""
        br = BatchResult()
        br.case_results = [
            CaseResult(case_id=1, converged=True),
            CaseResult(case_id=2, converged=False),
            CaseResult(case_id=3, converged=True),
        ]
        assert br.n_converged == 2
        assert br.n_total == 3

    def test_get_result(self):
        """Retrieve result by case ID."""
        br = BatchResult()
        br.case_results = [
            CaseResult(case_id=42, converged=True, nz=2.5),
        ]
        r = br.get_result(42)
        assert r is not None
        assert r.nz == pytest.approx(2.5)

    def test_get_missing_result(self):
        """Missing case ID returns None."""
        br = BatchResult()
        assert br.get_result(999) is None

    def test_results_by_category(self):
        """Filter results by category."""
        br = BatchResult()
        br.case_results = [
            CaseResult(case_id=1, category="symmetric"),
            CaseResult(case_id=2, category="gust"),
            CaseResult(case_id=3, category="symmetric"),
        ]
        sym = br.results_by_category("symmetric")
        assert len(sym) == 2

    def test_summary(self):
        """Summary returns correct structure."""
        br = BatchResult()
        br.case_results = [
            CaseResult(case_id=1, category="symmetric", converged=True),
            CaseResult(case_id=2, category="gust", converged=True),
            CaseResult(case_id=3, category="symmetric", converged=False),
        ]
        br.wall_time_s = 5.0
        s = br.summary()
        assert s["total"] == 3
        assert s["converged"] == 2
        assert s["wall_time_s"] == 5.0
        assert "symmetric" in s["by_category"]


# ---------------------------------------------------------------------------
# BatchRunner tests (no solver)
# ---------------------------------------------------------------------------

class TestBatchRunnerNoSolver:
    """Test BatchRunner without actual solver (placeholder mode)."""

    def test_run_creates_results(self):
        """Running without model creates placeholder results."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None, n_workers=0)
        results = runner.run()

        assert results.n_total == matrix.total_cases
        assert results.wall_time_s > 0

    def test_all_flight_cases_processed(self):
        """All flight cases produce results."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        flight_ids = {c.case_id for c in matrix.flight_cases}
        result_ids = {r.case_id for r in results.case_results
                       if r.category != "landing"}
        assert flight_ids == result_ids

    def test_all_landing_cases_processed(self):
        """All landing cases produce results."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        landing_results = results.results_by_category("landing")
        assert len(landing_results) == len(matrix.landing_cases)

    def test_landing_gear_forces(self):
        """Landing cases have gear reaction forces."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        landing_results = results.results_by_category("landing")
        # At least some landing cases should have nodal forces
        with_forces = [r for r in landing_results if r.nodal_forces]
        assert len(with_forces) > 0

    def test_landing_vertical_equilibrium(self):
        """Landing gear reactions are positive (upward)."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        landing_results = results.results_by_category("landing")
        for r in landing_results:
            if r.nodal_forces:
                total_fz = sum(f[2] for f in r.nodal_forces.values())
                assert total_fz > 0  # Net upward reaction

    def test_categories_preserved(self):
        """Category metadata is preserved in results."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        cats = {r.category for r in results.case_results}
        assert "symmetric" in cats
        assert "gust" in cats
        assert "rolling" in cats
        assert "yaw" in cats
        assert "landing" in cats

    def test_nz_preserved(self):
        """nz values are preserved in results."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        # Find a symmetric case with high nz
        sym = results.results_by_category("symmetric")
        nz_values = [r.nz for r in sym]
        assert any(nz > 2.0 for nz in nz_values)
        assert any(nz < 0 for nz in nz_values)

    def test_batch_size_parameter(self):
        """Different batch sizes produce same result count."""
        matrix = make_matrix()

        r1 = BatchRunner(matrix, bdf_model=None, batch_size=10).run()
        r2 = BatchRunner(matrix, bdf_model=None, batch_size=100).run()

        assert r1.n_total == r2.n_total


# ---------------------------------------------------------------------------
# Checkpointing tests
# ---------------------------------------------------------------------------

class TestCheckpointing:
    """Test checkpoint save/resume."""

    def test_save_checkpoint(self, tmp_path):
        """Checkpoint file is created."""
        matrix = make_matrix()
        cp_dir = str(tmp_path / "checkpoints")

        runner = BatchRunner(matrix, bdf_model=None,
                              checkpoint_dir=cp_dir)
        runner.run()

        cp_file = os.path.join(cp_dir, "batch_checkpoint.json")
        assert os.path.exists(cp_file)

    def test_checkpoint_content(self, tmp_path):
        """Checkpoint contains correct metadata."""
        matrix = make_matrix()
        cp_dir = str(tmp_path / "checkpoints")

        runner = BatchRunner(matrix, bdf_model=None,
                              checkpoint_dir=cp_dir)
        results = runner.run()

        cp_file = os.path.join(cp_dir, "batch_checkpoint.json")
        with open(cp_file, 'r') as f:
            data = json.load(f)

        assert len(data["completed_ids"]) == results.n_total
        assert len(data["results_metadata"]) == results.n_total

    def test_resume_from_checkpoint(self, tmp_path):
        """Resume produces same result count."""
        matrix = make_matrix()
        cp_dir = str(tmp_path / "checkpoints")

        # First run
        runner1 = BatchRunner(matrix, bdf_model=None,
                               checkpoint_dir=cp_dir)
        r1 = runner1.run()

        # Resume run — all cases already done, should be fast
        runner2 = BatchRunner(matrix, bdf_model=None,
                               checkpoint_dir=cp_dir)
        r2 = runner2.run(resume=True)

        assert r2.n_total == r1.n_total

    def test_resume_skips_completed(self, tmp_path):
        """Resumed run doesn't re-process completed cases."""
        matrix = make_matrix()
        cp_dir = str(tmp_path / "checkpoints")

        # First run
        runner1 = BatchRunner(matrix, bdf_model=None,
                               checkpoint_dir=cp_dir)
        r1 = runner1.run()

        # Resume — should load checkpoint and skip all
        runner2 = BatchRunner(matrix, bdf_model=None,
                               checkpoint_dir=cp_dir)
        r2 = runner2.run(resume=True)

        # Total should be same but second run should be very fast
        assert r2.n_total == r1.n_total
        assert r2.wall_time_s < r1.wall_time_s + 1.0  # Allow some slack

    def test_no_checkpoint_dir(self):
        """Running without checkpoint_dir works fine."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None,
                              checkpoint_dir=None)
        results = runner.run()
        assert results.n_total > 0


# ---------------------------------------------------------------------------
# Summary and reporting tests
# ---------------------------------------------------------------------------

class TestBatchSummary:
    """Test batch result summary and reporting."""

    def test_summary_categories(self):
        """Summary includes all categories."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        summary = results.summary()
        cats = summary["by_category"]
        assert "symmetric" in cats
        assert "gust" in cats
        assert "rolling" in cats

    def test_far_coverage_from_results(self):
        """FAR sections are tracked in results."""
        matrix = make_matrix()
        runner = BatchRunner(matrix, bdf_model=None)
        results = runner.run()

        far_sections = {r.far_section for r in results.case_results
                         if r.far_section}
        assert "§23.337" in far_sections
        assert "§23.341" in far_sections
        assert "§23.349" in far_sections

"""Tests for Phase 5: Envelope processing and critical case identification.

Tests VMT envelope computation, critical case identification,
potato plot data generation, and summary statistics.
"""
import math
import pytest
import numpy as np

from nastaero.loads_analysis.certification.batch_runner import (
    BatchResult, CaseResult,
)
from nastaero.loads_analysis.certification.envelope import (
    CriticalCase, StationEnvelope, ComponentEnvelope,
    PotatoData, EnvelopeProcessor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch_result():
    """Create a BatchResult with test case metadata."""
    br = BatchResult()
    br.case_results = [
        CaseResult(case_id=1, category="symmetric", far_section="§23.337",
                    converged=True, nz=3.8, label="Symmetric nz=3.8"),
        CaseResult(case_id=2, category="symmetric", far_section="§23.337",
                    converged=True, nz=-1.52, label="Symmetric nz=-1.52"),
        CaseResult(case_id=3, category="gust", far_section="§23.341",
                    converged=True, nz=2.5, label="Gust VC+"),
        CaseResult(case_id=4, category="rolling", far_section="§23.349",
                    converged=True, nz=1.0, label="Roll Right VA"),
        CaseResult(case_id=5, category="landing", far_section="§23.479",
                    converged=True, nz=2.67, label="Level landing"),
    ]
    br.completed_ids = {1, 2, 3, 4, 5}
    return br


def make_vmt_data():
    """Create synthetic VMT data for testing.

    Returns {case_id: {component: {stations, shear, bending, torsion}}}
    """
    stations = np.array([0.0, 2500.0, 5000.0, 7500.0, 10000.0])

    vmt = {}

    # Case 1: High positive (symmetric nz=3.8)
    vmt[1] = {
        "Wing": {
            "stations": stations,
            "shear": np.array([50000, 40000, 30000, 15000, 0]),
            "bending": np.array([0, 5e7, 8e7, 9e7, 1e8]),
            "torsion": np.array([1e6, 8e5, 5e5, 2e5, 0]),
        }
    }

    # Case 2: High negative (symmetric nz=-1.52)
    vmt[2] = {
        "Wing": {
            "stations": stations,
            "shear": np.array([-20000, -16000, -12000, -6000, 0]),
            "bending": np.array([0, -2e7, -3.5e7, -4e7, -4.5e7]),
            "torsion": np.array([-5e5, -4e5, -3e5, -1e5, 0]),
        }
    }

    # Case 3: Gust (moderate positive)
    vmt[3] = {
        "Wing": {
            "stations": stations,
            "shear": np.array([35000, 28000, 21000, 10000, 0]),
            "bending": np.array([0, 3.5e7, 6e7, 7e7, 7.5e7]),
            "torsion": np.array([1.2e6, 9e5, 6e5, 3e5, 0]),
        }
    }

    # Case 4: Rolling (asymmetric — higher torsion)
    vmt[4] = {
        "Wing": {
            "stations": stations,
            "shear": np.array([30000, 25000, 18000, 8000, 0]),
            "bending": np.array([0, 3e7, 5e7, 6e7, 6.5e7]),
            "torsion": np.array([2e6, 1.5e6, 1e6, 5e5, 0]),
        }
    }

    # Case 5: Landing (different distribution)
    vmt[5] = {
        "Wing": {
            "stations": stations,
            "shear": np.array([60000, 45000, 25000, 10000, 0]),
            "bending": np.array([0, 6e7, 9e7, 9.5e7, 1.05e8]),
            "torsion": np.array([8e5, 6e5, 4e5, 2e5, 0]),
        }
    }

    return vmt


# ---------------------------------------------------------------------------
# StationEnvelope tests
# ---------------------------------------------------------------------------

class TestStationEnvelope:
    """Test StationEnvelope dataclass."""

    def test_default_extremes(self):
        """Default values are ±inf."""
        se = StationEnvelope()
        assert se.V_max == -math.inf
        assert se.V_min == math.inf


# ---------------------------------------------------------------------------
# ComponentEnvelope tests
# ---------------------------------------------------------------------------

class TestComponentEnvelope:
    """Test ComponentEnvelope dataclass."""

    def test_array_properties(self):
        """Array properties return correct shapes."""
        env = ComponentEnvelope(
            component="Wing",
            stations=[0, 1000, 2000],
            envelopes=[
                StationEnvelope(station=0, V_max=100, V_min=-50,
                                 M_max=1e6, M_min=-5e5,
                                 T_max=1e5, T_min=-1e4),
                StationEnvelope(station=1000, V_max=80, V_min=-40,
                                 M_max=8e5, M_min=-4e5,
                                 T_max=8e4, T_min=-8e3),
                StationEnvelope(station=2000, V_max=0, V_min=0,
                                 M_max=0, M_min=0,
                                 T_max=0, T_min=0),
            ],
        )

        assert len(env.V_max_array) == 3
        assert env.V_max_array[0] == 100
        assert env.M_max_array[0] == 1e6


# ---------------------------------------------------------------------------
# EnvelopeProcessor tests
# ---------------------------------------------------------------------------

class TestEnvelopeProcessor:
    """Test the main envelope processor."""

    @pytest.fixture
    def processor(self):
        br = make_batch_result()
        vmt = make_vmt_data()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        return proc

    def test_compute_envelopes(self, processor):
        """Envelopes are computed for each component."""
        env = processor.get_envelope("Wing")
        assert env is not None
        assert env.component == "Wing"
        assert len(env.stations) == 5
        assert env.n_cases == 5

    def test_max_shear_at_root(self, processor):
        """Maximum shear is at root station."""
        env = processor.get_envelope("Wing")
        # Root station (index 0) should have highest shear
        root_env = env.envelopes[0]
        # Landing case (60000) has the highest root shear
        assert root_env.V_max == pytest.approx(60000)
        assert root_env.V_max_case_id == 5  # Landing

    def test_min_shear_at_root(self, processor):
        """Minimum shear is from negative case."""
        env = processor.get_envelope("Wing")
        root_env = env.envelopes[0]
        assert root_env.V_min == pytest.approx(-20000)
        assert root_env.V_min_case_id == 2  # Negative symmetric

    def test_max_bending_at_tip(self, processor):
        """Maximum bending is at tip station."""
        env = processor.get_envelope("Wing")
        tip_env = env.envelopes[4]  # station 10000
        # Landing case has highest tip bending (1.05e8)
        assert tip_env.M_max == pytest.approx(1.05e8)
        assert tip_env.M_max_case_id == 5  # Landing

    def test_max_torsion_from_rolling(self, processor):
        """Maximum torsion is from rolling case (asymmetric)."""
        env = processor.get_envelope("Wing")
        root_env = env.envelopes[0]
        # Rolling case has highest root torsion (2e6)
        assert root_env.T_max == pytest.approx(2e6)
        assert root_env.T_max_case_id == 4  # Rolling

    def test_tip_values_zero(self, processor):
        """Tip station values are zero (free end)."""
        env = processor.get_envelope("Wing")
        tip = env.envelopes[4]
        assert tip.V_max == pytest.approx(0)
        assert tip.V_min == pytest.approx(0)

    def test_add_vmt_curve(self):
        """add_vmt_curve adds data correctly."""
        br = make_batch_result()
        proc = EnvelopeProcessor(br)

        stations = np.array([0, 100, 200])
        proc.add_vmt_curve(
            case_id=1, component="HTP",
            stations=stations,
            shear=np.array([1000, 500, 0]),
            bending=np.array([0, 5e4, 1e5]),
            torsion=np.array([100, 50, 0]),
        )

        assert 1 in proc.vmt_data
        assert "HTP" in proc.vmt_data[1]


# ---------------------------------------------------------------------------
# Critical case identification tests
# ---------------------------------------------------------------------------

class TestCriticalCases:
    """Test critical case identification."""

    @pytest.fixture
    def processor(self):
        br = make_batch_result()
        vmt = make_vmt_data()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        proc.identify_critical_cases()
        return proc

    def test_critical_cases_exist(self, processor):
        """Critical cases are identified."""
        cc = processor.get_critical_cases()
        assert len(cc) > 0

    def test_critical_per_station(self, processor):
        """Each station has 6 critical cases (V/M/T × max/min)."""
        cc_wing = processor.get_critical_cases(component="Wing")
        # 5 stations × 6 quantities = 30
        assert len(cc_wing) == 30

    def test_critical_max_shear_category(self, processor):
        """Max shear at root is from landing category."""
        cc = processor.get_critical_cases(component="Wing", quantity="V")
        root_max = [c for c in cc
                     if c.station == 0.0 and c.extreme == "max"]
        assert len(root_max) == 1
        assert root_max[0].category == "landing"
        assert root_max[0].case_id == 5

    def test_critical_max_torsion_category(self, processor):
        """Max torsion at root is from rolling category."""
        cc = processor.get_critical_cases(component="Wing", quantity="T")
        root_max = [c for c in cc
                     if c.station == 0.0 and c.extreme == "max"]
        assert len(root_max) == 1
        assert root_max[0].category == "rolling"
        assert root_max[0].case_id == 4

    def test_critical_case_frequency(self, processor):
        """Case frequency counts are correct."""
        freq = processor.critical_case_frequency()
        assert isinstance(freq, dict)
        # At least some cases should be critical
        assert sum(freq.values()) > 0

    def test_category_distribution(self, processor):
        """Category distribution includes diverse categories."""
        dist = processor.critical_category_distribution()
        # Should have at least symmetric and rolling
        assert len(dist) >= 2

    def test_critical_has_far_section(self, processor):
        """Critical cases have FAR section metadata."""
        cc = processor.get_critical_cases()
        for c in cc:
            assert c.far_section != "" or c.case_id == 0


# ---------------------------------------------------------------------------
# Potato plot tests
# ---------------------------------------------------------------------------

class TestPotatoPlot:
    """Test potato plot data generation."""

    @pytest.fixture
    def processor(self):
        br = make_batch_result()
        vmt = make_vmt_data()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        return proc

    def test_potato_vm(self, processor):
        """V-M potato plot has correct number of points."""
        potato = processor.compute_potato(
            "Wing", station=5000.0, x_quantity="V", y_quantity="M")
        assert potato.n_points == 5  # 5 cases
        assert potato.component == "Wing"
        assert "Shear" in potato.x_label
        assert "Bending" in potato.y_label

    def test_potato_mt(self, processor):
        """M-T potato plot works."""
        potato = processor.compute_potato(
            "Wing", station=2500.0, x_quantity="M", y_quantity="T")
        assert potato.n_points == 5

    def test_potato_categories(self, processor):
        """Potato data includes category labels."""
        potato = processor.compute_potato(
            "Wing", station=5000.0)
        assert len(potato.categories) == 5
        assert "symmetric" in potato.categories
        assert "rolling" in potato.categories

    def test_potato_case_ids(self, processor):
        """Potato data includes case IDs."""
        potato = processor.compute_potato(
            "Wing", station=5000.0)
        assert 1 in potato.case_ids
        assert 4 in potato.case_ids

    def test_potato_convex_hull(self, processor):
        """Convex hull is computed when scipy available."""
        potato = processor.compute_potato(
            "Wing", station=5000.0)
        # Hull may or may not be available (depends on scipy)
        # Just check it doesn't crash
        if potato.hull_x is not None:
            assert len(potato.hull_x) >= 4  # At least a triangle + close

    def test_potato_nearest_station(self, processor):
        """Potato uses nearest station when exact match unavailable."""
        potato = processor.compute_potato(
            "Wing", station=4999.0)  # Close to 5000
        assert potato.n_points == 5  # Should still work


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestEnvelopeSummary:
    """Test envelope summary statistics."""

    def test_summary_structure(self):
        """Summary returns expected structure."""
        br = make_batch_result()
        vmt = make_vmt_data()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        proc.identify_critical_cases()

        summary = proc.summary()
        assert "components" in summary
        assert "n_critical" in summary
        assert "category_distribution" in summary
        assert "case_frequency_top10" in summary

    def test_summary_component_list(self):
        """Summary lists all components."""
        br = make_batch_result()
        vmt = make_vmt_data()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()

        summary = proc.summary()
        assert "Wing" in summary["components"]


class TestCriticalCaseDataclass:
    """Test CriticalCase dataclass."""

    def test_default_values(self):
        cc = CriticalCase()
        assert cc.station == 0.0
        assert cc.value == 0.0
        assert cc.quantity == ""

    def test_full_init(self):
        cc = CriticalCase(
            station=5000.0,
            component="Wing",
            quantity="M",
            extreme="max",
            value=1e8,
            case_id=42,
            category="symmetric",
        )
        assert cc.station == 5000.0
        assert cc.value == 1e8


# ---------------------------------------------------------------------------
# Interaction-diagram (potato-plot) critical selection + design-load set
# ---------------------------------------------------------------------------

from nastaero.loads_analysis.certification.envelope import (  # noqa: E402
    DesignLoadCase, _hull_vertex_indices,
)


def make_combined_load_vmt():
    """VMT data where a combined-load case is critical on the hull but is
    not the axis max/min of any single quantity at the tip station.

    Station index -1 (tip). Cases:
      1: max shear, mid moment      (V axis extreme)
      2: mid shear, max moment      (M axis extreme)
      3: high shear AND high moment (interior to neither axis, but a hull
         corner of the V-M plane -> combined-load critical)
      4: small loads                (interior point, never critical)
    """
    stations = np.array([0.0, 5000.0])
    vmt = {
        1: {"Wing": {"stations": stations,
                     "shear":   np.array([0.0, 100.0]),
                     "bending": np.array([0.0, 40.0]),
                     "torsion": np.array([0.0, 10.0])}},
        2: {"Wing": {"stations": stations,
                     "shear":   np.array([0.0, 40.0]),
                     "bending": np.array([0.0, 100.0]),
                     "torsion": np.array([0.0, 10.0])}},
        3: {"Wing": {"stations": stations,
                     "shear":   np.array([0.0, 90.0]),
                     "bending": np.array([0.0, 90.0]),
                     "torsion": np.array([0.0, 80.0])}},
        # centroid of cases 1-3 -> strictly interior in every plane
        4: {"Wing": {"stations": stations,
                     "shear":   np.array([0.0, 76.7]),
                     "bending": np.array([0.0, 76.7]),
                     "torsion": np.array([0.0, 33.3])}},
    }
    br = BatchResult()
    br.case_results = [
        CaseResult(case_id=1, category="symmetric", converged=True, nz=3.0, label="V-max"),
        CaseResult(case_id=2, category="gust", converged=True, nz=2.5, label="M-max"),
        CaseResult(case_id=3, category="rolling", converged=True, nz=2.0, label="combined"),
        CaseResult(case_id=4, category="taxi", converged=True, nz=1.0, label="mild"),
    ]
    br.completed_ids = {1, 2, 3, 4}
    return br, vmt


class TestHullVertexHelper:
    def test_axis_extremes_always_included(self):
        # square + interior point; interior (index 4) must be excluded
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
        v = _hull_vertex_indices(pts)
        assert 4 not in v
        assert set(v) == {0, 1, 2, 3}

    def test_degenerate_few_points(self):
        assert _hull_vertex_indices(np.array([[1.0, 2.0]])) == [0]
        assert set(_hull_vertex_indices(np.array([[0, 0], [1, 1]]))) == {0, 1}

    def test_collinear_falls_back_to_axis_extremes(self):
        pts = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        v = _hull_vertex_indices(pts)
        assert 0 in v and 3 in v  # the two ends


class TestInteractionCriticalSelection:
    def test_combined_load_case_is_selected(self):
        br, vmt = make_combined_load_vmt()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        proc.add_interaction_critical_cases()

        cids = {cc.case_id for cc in proc.get_critical_cases()}
        # The combined-load case (3) sits on the V-M / V-T / M-T hull corner
        assert 3 in cids
        # Axis extremes are on the hull too
        assert 1 in cids and 2 in cids
        # The mild interior case is never a hull vertex
        assert 4 not in cids

    def test_hull_records_use_plane_label(self):
        br, vmt = make_combined_load_vmt()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        proc.add_interaction_critical_cases(planes=(("V", "M"),))
        hull = [c for c in proc.get_critical_cases() if c.extreme == "hull"]
        assert hull
        assert all(c.quantity == "V-M" for c in hull)


class TestDesignCaseReduction:
    def test_reduces_and_ranks(self):
        br, vmt = make_combined_load_vmt()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        proc.identify_critical_cases()
        proc.add_interaction_critical_cases()
        design = proc.select_design_cases()

        # compact: one DesignLoadCase per distinct case, mild case dropped
        assert all(isinstance(d, DesignLoadCase) for d in design)
        ids = [d.case_id for d in design]
        assert len(ids) == len(set(ids))         # deduplicated
        assert 4 not in ids                       # interior case not a design load
        # ranked by number of governed extremes, descending
        ngov = [d.n_govern for d in design]
        assert ngov == sorted(ngov, reverse=True)
        assert design[0].n_govern >= 1
        assert design[0].components == ["Wing"]


class TestDesignLoadTable:
    def test_why_and_basis(self):
        from nastaero.loads_analysis.certification.envelope import (
            DesignLoadCase, design_load_table,
        )
        d = DesignLoadCase(
            case_id=7, label="gust", category="gust", nz=2.5,
            governs=[("Right Wing", 3000.0, "M", "max"),
                     ("Right Wing", 3000.0, "V-M", "hull"),
                     ("VTP", 0.0, "V-T", "hull")],
            components=["Right Wing", "VTP"], n_govern=3)
        row = design_load_table([d])[0]
        assert row["basis"] == "both"
        assert row["envelope_quantities"] == ["M"]
        assert set(row["interaction_planes"]) == {"V-M", "V-T"}
        assert "Right Wing" in row["why"] and "potato" in row["why"]

    def test_pure_interaction_basis(self):
        from nastaero.loads_analysis.certification.envelope import DesignLoadCase
        d = DesignLoadCase(case_id=1, governs=[("Wing", 1.0, "V-M", "hull")],
                           components=["Wing"], n_govern=1)
        assert d.basis == "interaction"
        assert d.by_envelope == []

    def test_full_selection_produces_table(self):
        br, vmt = make_combined_load_vmt()
        proc = EnvelopeProcessor(br, vmt)
        proc.compute_envelopes()
        proc.identify_critical_cases()
        proc.add_interaction_critical_cases()
        from nastaero.loads_analysis.certification.envelope import design_load_table
        table = design_load_table(proc.select_design_cases())
        assert table and table[0]["rank"] == 1
        assert all("basis" in r and "why" in r for r in table)


class TestHull3d:
    """3D (V,M,T) 볼록 헐 선정 — 2D 투영이 놓치는 삼중 조합 임계."""

    def _octahedron_batch(self):
        # 케이스 1~6: (V,M,T) 축 극값(정팔면체 꼭짓점), 케이스 7: 세
        # 투영 모두에서 내부점이지만 3D에서는 몸체 밖(x+y+z=1.35>1)
        pts = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1), (0.45, 0.45, 0.45),
        ]
        br = BatchResult()
        vmt = {}
        for cid, (v, m, t) in enumerate(pts, 1):
            br.case_results.append(CaseResult(
                case_id=cid, category="test", converged=True,
                label=f"case{cid}"))
            br.completed_ids.add(cid)
            vmt[cid] = {"Wing": {
                "stations": np.array([0.0]),
                "shear": np.array([float(v)]),
                "bending": np.array([float(m)]),
                "torsion": np.array([float(t)]),
            }}
        return br, vmt

    def test_2d_planes_miss_interior_projection_case(self):
        br, vmt = self._octahedron_batch()
        proc = EnvelopeProcessor(br, vmt)
        proc.add_interaction_critical_cases()
        assert 7 not in {cc.case_id for cc in proc.get_critical_cases()}

    def test_3d_hull_catches_it(self):
        br, vmt = self._octahedron_batch()
        proc = EnvelopeProcessor(br, vmt)
        added = proc.add_interaction_critical_cases_3d()
        ids = {cc.case_id for cc in added}
        assert 7 in ids                      # 삼중 조합 임계 포착
        assert {1, 2, 3, 4, 5, 6} <= ids     # 2D 꼭짓점은 부분집합으로 유지
        cc7 = next(cc for cc in added if cc.case_id == 7)
        assert cc7.extreme == "hull3d" and cc7.quantity == "V-M-T"

    def test_design_case_basis_includes_hull3d(self):
        br, vmt = self._octahedron_batch()
        proc = EnvelopeProcessor(br, vmt)
        proc.add_interaction_critical_cases_3d()
        dcs = proc.select_design_cases()
        dc7 = next(d for d in dcs if d.case_id == 7)
        assert dc7.basis == "interaction"
        assert "potato" in dc7.why()

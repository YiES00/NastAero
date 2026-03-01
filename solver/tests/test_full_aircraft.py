"""Tests for full aircraft static aeroelastic trim (SOL 144).

Tests a wing + horizontal tail + fuselage beam model with elevator
control surface. Verifies:
- BDF parsing of multiple CAERO1, per-spline mapping, AELIST, AESURF
- Two-constraint trim: Fz balance (lift = weight) + My balance (moment = 0)
- Free trim variables: ANGLEA and ELEV
- Physically correct displacement directions
- Wing and tail contribute separately to total lift
"""
import os
import numpy as np
import pytest
from nastaero.bdf.parser import BDFParser

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
AIRCRAFT_BDF = os.path.join(VALIDATION_DIR, "full_aircraft", "full_aircraft_trim.bdf")


def parse_bdf(filepath):
    parser = BDFParser()
    return parser.parse(filepath)


class TestFullAircraftParsing:
    """Test that the full aircraft BDF is parsed correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = parse_bdf(AIRCRAFT_BDF)

    def test_sol(self):
        assert self.model.sol == 144

    def test_nodes(self):
        assert len(self.model.nodes) == 23  # 11 wing + 7 tail + 5 fuselage

    def test_elements(self):
        assert len(self.model.elements) == 20  # 10 wing + 6 tail + 4 fuselage

    def test_rigids(self):
        assert len(self.model.rigids) == 2  # wing-fuse + tail-fuse RBE2

    def test_masses(self):
        assert len(self.model.masses) == 5  # fuselage concentrated masses

    def test_two_caero_panels(self):
        """Should have separate wing and tail CAERO1 panels."""
        assert len(self.model.caero_panels) == 2
        assert 2001 in self.model.caero_panels  # wing
        assert 3001 in self.model.caero_panels  # tail

    def test_wing_caero(self):
        c = self.model.caero_panels[2001]
        assert c.nspan == 10
        assert c.nchord == 2
        assert c.p1[0] == pytest.approx(0.0)  # Wing LE at x=0
        assert c.chord1 == pytest.approx(2.0)  # 2m chord

    def test_tail_caero(self):
        c = self.model.caero_panels[3001]
        assert c.nspan == 6
        assert c.nchord == 2
        assert c.p1[0] == pytest.approx(8.0)  # Tail LE at x=8
        assert c.chord1 == pytest.approx(1.0)  # 1m chord

    def test_two_splines(self):
        """Should have separate wing and tail splines."""
        assert len(self.model.splines) == 2
        assert 10 in self.model.splines  # wing spline
        assert 20 in self.model.splines  # tail spline

    def test_wing_spline(self):
        s = self.model.splines[10]
        assert s.caero == 2001
        assert s.setg == 10

    def test_tail_spline(self):
        s = self.model.splines[20]
        assert s.caero == 3001
        assert s.setg == 20

    def test_structural_sets(self):
        """Wing and tail should have separate SET1 definitions."""
        assert 10 in self.model.sets
        assert 20 in self.model.sets
        assert len(self.model.sets[10].ids) == 11  # wing nodes
        assert len(self.model.sets[20].ids) == 7   # tail nodes

    def test_aesurf(self):
        """Elevator control surface should be defined."""
        assert len(self.model.aesurfs) == 1
        elev = next(iter(self.model.aesurfs.values()))
        assert elev.label == "ELEV"
        assert elev.alid1 == 610

    def test_aelist(self):
        """AELIST should define elevator boxes (TE column of tail)."""
        assert 610 in self.model.aelists
        al = self.model.aelists[610]
        assert len(al.elements) == 6  # 6 TE boxes
        assert 3002 in al.elements
        assert 3012 in al.elements

    def test_trim(self):
        assert 1 in self.model.trims
        t = self.model.trims[1]
        assert t.mach == pytest.approx(0.3)
        assert t.q == pytest.approx(6125.0)
        assert len(t.variables) == 1
        assert t.variables[0][0] == "URDD3"


class TestFullAircraftTrim:
    """Test SOL 144 trim solution for the full aircraft model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from nastaero.solvers.sol144 import solve_trim
        self.model = parse_bdf(AIRCRAFT_BDF)
        self.results = solve_trim(self.model)
        self.sc = self.results.subcases[0]

    def test_has_results(self):
        assert len(self.results.subcases) > 0

    def test_displacements_exist(self):
        assert len(self.sc.displacements) > 0

    def test_trim_variables_exist(self):
        assert self.sc.trim_variables is not None
        assert "ANGLEA" in self.sc.trim_variables
        assert "ELEV" in self.sc.trim_variables

    def test_anglea_reasonable(self):
        """ANGLEA should be a small positive angle (1-10 degrees)."""
        alpha = self.sc.trim_variables["ANGLEA"]
        alpha_deg = np.degrees(alpha)
        assert np.isfinite(alpha), "ANGLEA should be finite"
        assert 0 < alpha_deg < 15, f"ANGLEA should be 0-15 deg, got {alpha_deg:.2f}"

    def test_elevator_reasonable(self):
        """Elevator deflection should be finite and reasonable."""
        elev = self.sc.trim_variables["ELEV"]
        elev_deg = np.degrees(elev)
        assert np.isfinite(elev), "ELEV should be finite"
        assert abs(elev_deg) < 30, f"ELEV magnitude too large: {elev_deg:.2f} deg"

    def test_lift_equals_weight(self):
        """Total aerodynamic lift should balance structural weight."""
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        assert total_fz > 0, f"Total Fz should be positive, got {total_fz:.2f}"
        # Weight from the model
        total_mass = 3450.0  # Expected from BDF properties
        weight = total_mass * 9.81
        assert abs(total_fz - weight) / weight < 0.01, (
            f"Lift should equal weight: Fz={total_fz:.1f}, W={weight:.1f}"
        )

    def test_pitch_moment_balanced(self):
        """Pitch moment about CG should be approximately zero."""
        # Compute CG
        from nastaero.solvers.sol144 import _compute_cg_x
        cg_x = _compute_cg_x(self.model)

        # Compute pitch moment
        n_boxes = len(self.sc.aero_boxes)
        my = sum(
            self.sc.aero_forces[i, 2] *
            (self.sc.aero_boxes[i].control_point[0] - cg_x)
            for i in range(n_boxes)
        )
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        # Moment should be small relative to force * reference length
        ref_moment = total_fz * 10.0  # force * approx distance
        assert abs(my) / ref_moment < 0.01, (
            f"Pitch moment should be ~0: My={my:.2f} N*m"
        )

    def test_wing_tip_deflects_upward(self):
        """Wing tip should bend upward under positive lift."""
        tip_z = self.sc.displacements[11][2]
        assert tip_z > 0, f"Wing tip should bend UP, got T3={tip_z:.6e}"

    def test_wing_displacement_increases_spanwise(self):
        """Wing displacement should increase from root to tip."""
        prev_z = -1e10
        for nid in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            z = self.sc.displacements[nid][2]
            assert z >= prev_z - 1e-12, (
                f"Displacement should increase: node {nid} T3={z:.6e} < prev {prev_z:.6e}"
            )
            prev_z = z

    def test_wing_and_tail_both_produce_lift(self):
        """Both wing and tail should generate positive lift."""
        n = len(self.sc.aero_boxes)
        wing_fz = sum(
            self.sc.aero_forces[i, 2]
            for i in range(n) if 2001 <= self.sc.aero_boxes[i].box_id <= 2020
        )
        tail_fz = sum(
            self.sc.aero_forces[i, 2]
            for i in range(n) if 3001 <= self.sc.aero_boxes[i].box_id <= 3012
        )
        assert wing_fz > 0, f"Wing should produce lift, got Fz={wing_fz:.2f}"
        assert tail_fz > 0, f"Tail should produce lift, got Fz={tail_fz:.2f}"

    def test_wing_carries_more_lift_than_tail(self):
        """Wing should carry more lift than horizontal tail."""
        n = len(self.sc.aero_boxes)
        wing_fz = sum(
            self.sc.aero_forces[i, 2]
            for i in range(n) if 2001 <= self.sc.aero_boxes[i].box_id <= 2020
        )
        tail_fz = sum(
            self.sc.aero_forces[i, 2]
            for i in range(n) if 3001 <= self.sc.aero_boxes[i].box_id <= 3012
        )
        assert wing_fz > tail_fz, (
            f"Wing Fz={wing_fz:.1f} should exceed tail Fz={tail_fz:.1f}"
        )

    def test_aero_boxes_count(self):
        """Should have 20 wing + 12 tail = 32 total boxes."""
        assert len(self.sc.aero_boxes) == 32
        assert self.sc.aero_forces.shape == (32, 3)

    def test_control_surface_effectiveness(self):
        """Elevator deflection should produce significant tail lift."""
        # With ELEV > 0 and only applied to tail TE boxes,
        # the tail TE boxes should have higher lift than LE boxes
        n = len(self.sc.aero_boxes)
        tail_le_fz = sum(
            self.sc.aero_forces[i, 2]
            for i in range(n) if self.sc.aero_boxes[i].box_id in [3001,3003,3005,3007,3009,3011]
        )
        tail_te_fz = sum(
            self.sc.aero_forces[i, 2]
            for i in range(n) if self.sc.aero_boxes[i].box_id in [3002,3004,3006,3008,3010,3012]
        )
        # TE boxes (elevator) should have higher force per box
        avg_le = tail_le_fz / 6
        avg_te = tail_te_fz / 6
        assert avg_te > avg_le, (
            f"Elevator TE lift/box={avg_te:.1f} should exceed LE={avg_le:.1f}"
        )

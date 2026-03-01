"""Tests for SOL 144 - Static Aeroelastic Trim."""
import os
import numpy as np
import pytest
from nastaero.bdf.parser import BDFParser

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
GOLAND_BDF = os.path.join(VALIDATION_DIR, "goland_wing", "goland_static.bdf")


def parse_bdf(filepath):
    parser = BDFParser()
    return parser.parse(filepath)


class TestGolandParsing:
    """Test that the Goland wing BDF is parsed correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = parse_bdf(GOLAND_BDF)

    def test_sol(self):
        assert self.model.sol == 144

    def test_nodes(self):
        assert len(self.model.nodes) == 11

    def test_elements(self):
        assert len(self.model.elements) == 10

    def test_aeros(self):
        assert self.model.aeros is not None
        assert self.model.aeros.refc == pytest.approx(1.8288)
        assert self.model.aeros.refb == pytest.approx(12.192)

    def test_aero(self):
        assert self.model.aero is not None
        assert self.model.aero.velocity == pytest.approx(50.0)

    def test_caero(self):
        assert 1001 in self.model.caero_panels
        c = self.model.caero_panels[1001]
        assert c.nspan == 8
        assert c.nchord == 2

    def test_spline(self):
        assert 100 in self.model.splines

    def test_set1(self):
        assert 10 in self.model.sets
        assert len(self.model.sets[10].ids) == 11

    def test_aestat(self):
        assert 501 in self.model.aestats
        assert self.model.aestats[501].label == "ANGLEA"

    def test_trim(self):
        assert 1 in self.model.trims
        t = self.model.trims[1]
        assert t.mach == pytest.approx(0.3)
        assert t.q == pytest.approx(1531.25)
        assert len(t.variables) == 1
        assert t.variables[0][0] == "URDD3"


class TestGolandTrim:
    """Test SOL 144 trim solution for Goland wing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from nastaero.solvers.sol144 import solve_trim
        self.model = parse_bdf(GOLAND_BDF)
        self.results = solve_trim(self.model)

    def test_has_results(self):
        assert len(self.results.subcases) > 0

    def test_displacements_exist(self):
        sc = self.results.subcases[0]
        assert len(sc.displacements) > 0

    def test_trim_variables_exist(self):
        sc = self.results.subcases[0]
        assert sc.trim_variables is not None
        assert "ANGLEA" in sc.trim_variables

    def test_angle_of_attack_reasonable(self):
        """ANGLEA should be a finite, small angle."""
        sc = self.results.subcases[0]
        alpha = sc.trim_variables["ANGLEA"]
        alpha_deg = np.degrees(alpha)
        # Check that ANGLEA is finite and within a reasonable range
        assert np.isfinite(alpha), "ANGLEA should be finite"
        assert abs(alpha_deg) < 30, f"ANGLEA magnitude too large: {alpha_deg:.4f} deg"

    def test_aero_forces(self):
        """Aero forces should exist and produce lift."""
        sc = self.results.subcases[0]
        assert sc.aero_forces is not None
        total_fz = np.sum(sc.aero_forces[:, 2])
        # Should produce positive lift (upward)
        assert total_fz > 0, f"Total Fz should be positive, got {total_fz:.2f}"

    def test_tip_displacement(self):
        """Tip should deflect (node 11)."""
        sc = self.results.subcases[0]
        if 11 in sc.displacements:
            tip_z = sc.displacements[11][2]  # z-displacement
            # Should be non-trivial
            assert abs(tip_z) > 1e-10, "Tip should have non-zero displacement"

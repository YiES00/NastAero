"""SOL 101 validation tests."""
import os
import numpy as np
import pytest

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
CANTILEVER_BDF = os.path.join(VALIDATION_DIR, "cantilever_beam", "cantilever.bdf")


def run_sol101(bdf_path):
    from nastaero.bdf.parser import BDFParser
    from nastaero.solvers.sol101 import solve_static
    parser = BDFParser()
    model = parser.parse(bdf_path)
    return solve_static(model)


class TestCantileverBeam:
    """Cantilever beam with tip load: delta = PL^3/(3EI)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.results = run_sol101(CANTILEVER_BDF)

    def test_single_subcase(self):
        assert len(self.results.subcases) == 1

    def test_tip_deflection(self):
        # Analytical: delta = PL^3 / (3*E*I)
        P = 100.0; L = 1.0; E = 7.0e10; I = 8.333e-10
        delta_analytical = P * L**3 / (3 * E * I)
        sc = self.results.subcases[0]
        # Tip node is 11, Z-displacement is index 2
        tip_z = sc.displacements[11][2]
        rel_error = abs(tip_z - delta_analytical) / delta_analytical
        assert rel_error < 0.002, (
            f"Tip deflection {tip_z:.6e} vs analytical {delta_analytical:.6e}, "
            f"relative error = {rel_error:.4%}"
        )

    def test_spc_reaction_z(self):
        """Sum of Z reactions should equal applied load."""
        sc = self.results.subcases[0]
        total_rz = sc.spc_forces[1][2]  # Z-reaction at fixed node
        assert abs(total_rz + 100.0) < 1.0, f"Z-reaction = {total_rz}, expected -100"

    def test_zero_x_displacement(self):
        """All X-displacements should be zero (no axial load)."""
        sc = self.results.subcases[0]
        for nid in sorted(sc.displacements.keys()):
            assert abs(sc.displacements[nid][0]) < 1e-10

    def test_displacement_increases_along_beam(self):
        """Z-displacement magnitude should increase from root to tip."""
        sc = self.results.subcases[0]
        prev_abs = 0.0
        for nid in range(2, 12):
            curr_abs = abs(sc.displacements[nid][2])
            assert curr_abs >= prev_abs, f"Node {nid}: |uz|={curr_abs} < prev {prev_abs}"
            prev_abs = curr_abs

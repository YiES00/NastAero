"""Comprehensive Nastran Verification Manual Tests.

Validates NastAero against well-known analytical solutions from
structural mechanics textbooks and MSC Nastran Verification Manual.
"""
import os
import numpy as np
import pytest
from nastaero.bdf.parser import BDFParser
from nastaero.solvers.sol101 import solve_static
from nastaero.solvers.sol103 import solve_modes

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
VM_DIR = os.path.join(VALIDATION_DIR, "nastran_vm")


def parse_and_solve_101(bdf_path):
    parser = BDFParser()
    model = parser.parse(bdf_path)
    return solve_static(model), model


def parse_and_solve_103(bdf_path):
    parser = BDFParser()
    model = parser.parse(bdf_path)
    return solve_modes(model), model


# ============================================================================
# VM1: Axial Rod Under Tension
# Ref: Nastran Verification Manual, Timoshenko
# ============================================================================
class TestVM1_AxialRod:
    """Axial rod: delta = PL/(AE).
    L=10 in, A=1.0 in^2, E=30e6 psi, P=1000 lb.
    Analytical: delta = 1000*10/(1*30e6) = 3.333e-4 in.
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm1_rod_axial.bdf"))
        self.sc = self.results.subcases[0]

    def test_axial_displacement(self):
        P = 1000.0; L = 10.0; A = 1.0; E = 30.0e6
        delta_analytical = P * L / (A * E)
        tip_x = self.sc.displacements[2][0]
        rel_error = abs(tip_x - delta_analytical) / delta_analytical
        assert rel_error < 0.001, (
            f"Axial disp {tip_x:.6e} vs analytical {delta_analytical:.6e}, "
            f"error = {rel_error:.4%}")

    def test_zero_transverse(self):
        """No transverse displacement for pure axial load."""
        for nid in self.sc.displacements:
            assert abs(self.sc.displacements[nid][1]) < 1e-15
            assert abs(self.sc.displacements[nid][2]) < 1e-15

    def test_reaction_force(self):
        """Reaction at fixed end should equal applied load."""
        rx = self.sc.spc_forces[1][0]
        assert abs(rx + 1000.0) < 0.1, f"X-reaction = {rx}, expected -1000"


# ============================================================================
# VM2: Cantilever Beam Under End Moment
# Ref: Timoshenko - Strength of Materials
# ============================================================================
class TestVM2_CantileverMoment:
    """Cantilever with tip moment M about Y-axis.
    L=10m, E=200GPa, I=1e-4 m^4, M=10000 N-m.
    Analytical: theta_tip = ML/(EI) = 10000*10/(200e9*1e-4) = 5.0e-3 rad
    delta_tip = ML^2/(2*EI) = 10000*100/(2*200e9*1e-4) = 2.5e-2 m

    Note: In CBAR convention, positive My moment at tip causes
    negative uz displacement (beam curves downward in z when moment
    is about +y axis with beam along +x).
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm2_cantilever_moment.bdf"))
        self.sc = self.results.subcases[0]

    def test_tip_deflection(self):
        M = 10000.0; L = 10.0; E = 200.0e9; I = 1.0e-4
        delta_analytical = M * L**2 / (2 * E * I)
        tip_z = self.sc.displacements[11][2]
        # Magnitude check (sign depends on CBAR convention)
        rel_error = abs(abs(tip_z) - delta_analytical) / delta_analytical
        assert rel_error < 0.01, (
            f"Tip Z-disp |{tip_z:.6e}| vs analytical {delta_analytical:.6e}, "
            f"error = {rel_error:.4%}")

    def test_tip_rotation(self):
        M = 10000.0; L = 10.0; E = 200.0e9; I = 1.0e-4
        theta_analytical = M * L / (E * I)
        # Rotation about Y at tip node (component index 4 = RY)
        tip_ry = self.sc.displacements[11][4]
        rel_error = abs(abs(tip_ry) - theta_analytical) / theta_analytical
        assert rel_error < 0.01, (
            f"Tip rotation |{tip_ry:.6e}| vs analytical {theta_analytical:.6e}, "
            f"error = {rel_error:.4%}")

    def test_midspan_deflection(self):
        """Midspan deflection magnitude = M*x^2/(2EI) at x=5."""
        M = 10000.0; E = 200.0e9; I = 1.0e-4
        x = 5.0
        delta_mid = M * x**2 / (2 * E * I)
        mid_z = self.sc.displacements[6][2]
        rel_error = abs(abs(mid_z) - delta_mid) / delta_mid
        assert rel_error < 0.01, (
            f"Mid-span Z-disp |{mid_z:.6e}| vs analytical {delta_mid:.6e}, "
            f"error = {rel_error:.4%}")

    def test_constant_curvature(self):
        """Pure moment -> constant curvature -> linear rotation along span."""
        M = 10000.0; E = 200.0e9; I = 1.0e-4
        kappa = M / (E * I)  # 1/R
        for nid in range(2, 12):
            x = (nid - 1) * 1.0
            theta_expected = kappa * x
            theta_computed = abs(self.sc.displacements[nid][4])
            if abs(theta_expected) > 1e-10:
                rel_error = abs(theta_computed - theta_expected) / abs(theta_expected)
                assert rel_error < 0.02, (
                    f"Node {nid}: |theta|={theta_computed:.6e} vs expected "
                    f"{theta_expected:.6e}, error={rel_error:.4%}")


# ============================================================================
# VM3: Simply-Supported Plate Under Uniform Pressure
# Ref: Timoshenko - Theory of Plates and Shells
# ============================================================================
class TestVM3_PlateUniformPressure:
    """SS plate under uniform pressure.
    a=b=1.0m, t=0.01m, E=200GPa, nu=0.3, p=1000 Pa.
    D = E*t^3/(12*(1-nu^2)) = 200e9*1e-6/10.92 = 18315.02 N-m.
    w_max = alpha*p*a^4/D, alpha=0.00406 for a/b=1 (Timoshenko Table 8)
    w_max = 0.00406 * 1000 * 1.0 / 18315.02 = 2.216e-4 m
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm3_plate_pressure.bdf"))
        self.sc = self.results.subcases[0]

    def test_center_deflection(self):
        """Center node (41) deflection vs Timoshenko analytical."""
        E = 200.0e9; nu = 0.3; t = 0.01; p = 1000.0; a = 1.0
        D = E * t**3 / (12 * (1 - nu**2))
        alpha = 0.00406  # Timoshenko, Table 8, a/b=1
        w_analytical = alpha * p * a**4 / D
        w_center = abs(self.sc.displacements[41][2])
        rel_error = abs(w_center - w_analytical) / w_analytical
        # 8x8 mesh with selective reduced integration -> ~5% accuracy
        assert rel_error < 0.10, (
            f"Center deflection {w_center:.6e} vs Timoshenko {w_analytical:.6e}, "
            f"error = {rel_error:.2%}")

    def test_symmetry(self):
        """Deflection should be symmetric about both axes."""
        sc = self.sc
        w39 = abs(sc.displacements[39][2])
        w43 = abs(sc.displacements[43][2])
        assert abs(w39 - w43) / max(w39, 1e-15) < 0.01, \
            f"X-symmetry broken: w(39)={w39:.6e} vs w(43)={w43:.6e}"

        w23 = abs(sc.displacements[23][2])
        w59 = abs(sc.displacements[59][2])
        assert abs(w23 - w59) / max(w23, 1e-15) < 0.01, \
            f"Y-symmetry broken: w(23)={w23:.6e} vs w(59)={w59:.6e}"

    def test_edge_zero(self):
        """Edges should have zero deflection (simply supported)."""
        for nid in range(1, 10):
            assert abs(self.sc.displacements[nid][2]) < 1e-12


# ============================================================================
# VM4: Cantilever Beam Natural Frequencies (20 element mesh)
# Ref: Euler-Bernoulli beam theory
# ============================================================================
class TestVM4_BeamModes20:
    """Cantilever beam: 20 CBAR elements, L=10m.
    f_n = beta_n^2/(2*pi*L^2) * sqrt(EI/(rho*A))
    E=200GPa, A=1e-4 m^2, I=8.333e-10 m^4, rho=7850 kg/m^3
    C = sqrt(EI/(rho*A)) = sqrt(200e9*8.333e-10/(7850*1e-4)) = 461.24 m^2/s
    f1 = 1.8751^2 * 461.24 / (2*pi*100) = 2.582 Hz
    f2 = 4.6941^2 * 461.24 / (2*pi*100) = 16.18 Hz
    f3 = 7.8548^2 * 461.24 / (2*pi*100) = 45.31 Hz

    Note: CBAR elements have 6 DOFs, so bending occurs in both XY and XZ
    planes, giving each mode twice (double frequencies). We extract unique
    frequencies by removing duplicates.
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_103(
            os.path.join(VM_DIR, "vm4_beam_modes_20elem.bdf"))
        self.sc = self.results.subcases[0]
        # Remove duplicate frequencies (XY and XZ bending give same frequency)
        all_freqs = self.sc.frequencies
        unique_freqs = []
        for f in all_freqs:
            if not unique_freqs or abs(f - unique_freqs[-1]) / max(f, 1e-10) > 0.01:
                unique_freqs.append(f)
        self.unique_freqs = unique_freqs

    def _analytical_frequency(self, beta_n):
        E = 200.0e9; I = 8.333e-10; rho = 7850.0; A = 1.0e-4; L = 10.0
        C = np.sqrt(E * I / (rho * A))
        return beta_n**2 * C / (2 * np.pi * L**2)

    def test_first_frequency(self):
        f1_analytical = self._analytical_frequency(1.8751)
        f1_computed = self.unique_freqs[0]
        rel_error = abs(f1_computed - f1_analytical) / f1_analytical
        assert rel_error < 0.02, (
            f"f1 = {f1_computed:.4f} Hz vs analytical {f1_analytical:.4f} Hz, "
            f"error = {rel_error:.2%}")

    def test_second_frequency(self):
        f2_analytical = self._analytical_frequency(4.6941)
        f2_computed = self.unique_freqs[1]
        rel_error = abs(f2_computed - f2_analytical) / f2_analytical
        assert rel_error < 0.05, (
            f"f2 = {f2_computed:.4f} Hz vs analytical {f2_analytical:.4f} Hz, "
            f"error = {rel_error:.2%}")

    def test_third_frequency(self):
        f3_analytical = self._analytical_frequency(7.8548)
        f3_computed = self.unique_freqs[2]
        rel_error = abs(f3_computed - f3_analytical) / f3_analytical
        assert rel_error < 0.10, (
            f"f3 = {f3_computed:.4f} Hz vs analytical {f3_analytical:.4f} Hz, "
            f"error = {rel_error:.2%}")

    def test_frequency_ratios(self):
        """Frequency ratios should match analytical ratios."""
        f1 = self.unique_freqs[0]
        f2 = self.unique_freqs[1]
        ratio_21_analytical = 4.6941**2 / 1.8751**2  # = 6.267
        ratio_21 = f2 / f1
        assert abs(ratio_21 - ratio_21_analytical) / ratio_21_analytical < 0.05, (
            f"f2/f1 = {ratio_21:.4f} vs analytical {ratio_21_analytical:.4f}")


# ============================================================================
# VM5: Three-Bar Truss
# Ref: Direct Stiffness Method textbook
# ============================================================================
class TestVM5_ThreeBarTruss:
    """Three-bar planar truss under combined loading (CROD elements).
    Nodes 1(0,40), 2(40,40), 3(40,0) pinned; Node 4(0,0) free.
    Rod 1: 1->4 (vertical, L=40), Rod 2: 2->4 (diagonal, L=56.57),
    Rod 3: 3->4 (horizontal, L=40).
    A=1 in^2, E=10e6 psi.
    Fx=1000, Fy=-1000 at node 4.

    Analytical: ux = 0.004 in, uy = -0.004 in
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm5_3bar_truss.bdf"))
        self.sc = self.results.subcases[0]

    def test_x_displacement(self):
        """Node 4 X-displacement."""
        ux = self.sc.displacements[4][0]
        ux_analytical = 0.004
        rel_error = abs(ux - ux_analytical) / abs(ux_analytical)
        assert rel_error < 0.001, (
            f"ux = {ux:.6e} vs analytical {ux_analytical:.6e}, "
            f"error = {rel_error:.2%}")

    def test_y_displacement(self):
        """Node 4 Y-displacement."""
        uy = self.sc.displacements[4][1]
        uy_analytical = -0.004
        rel_error = abs(uy - uy_analytical) / abs(uy_analytical)
        assert rel_error < 0.001, (
            f"uy = {uy:.6e} vs analytical {uy_analytical:.6e}, "
            f"error = {rel_error:.2%}")

    def test_equilibrium(self):
        """Sum of all reactions should equal applied load."""
        total_rx = sum(self.sc.spc_forces[nid][0] for nid in [1, 2, 3])
        total_ry = sum(self.sc.spc_forces[nid][1] for nid in [1, 2, 3])
        assert abs(total_rx + 1000.0) < 1.0, f"Sum Rx = {total_rx}, expected -1000"
        assert abs(total_ry + (-1000.0)) < 1.0, f"Sum Ry = {total_ry}, expected 1000"


# ============================================================================
# Existing Cantilever Beam Tests (enhanced precision)
# ============================================================================
class TestExistingCantilever_Enhanced:
    """Enhanced validation of existing cantilever beam.
    L=1.0m, E=7e10, A=1e-4, I=8.333e-10, P=100N (in +Z direction).
    delta = PL^3/(3EI) = 100/(3*7e10*8.333e-10) = 5.714e-1 m.

    CBAR convention: positive Z-force -> positive Z-displacement,
    but negative Y-rotation (θy < 0).
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VALIDATION_DIR, "cantilever_beam", "cantilever.bdf"))
        self.sc = self.results.subcases[0]

    def test_tip_deflection_precise(self):
        P = 100.0; L = 1.0; E = 7.0e10; I = 8.333e-10
        delta = P * L**3 / (3 * E * I)
        tip_z = self.sc.displacements[11][2]
        rel_error = abs(tip_z - delta) / delta
        assert rel_error < 0.002, f"Error = {rel_error:.4%}"

    def test_tip_rotation(self):
        """Tip rotation magnitude = PL^2/(2EI)."""
        P = 100.0; L = 1.0; E = 7.0e10; I = 8.333e-10
        theta = P * L**2 / (2 * E * I)
        tip_ry = self.sc.displacements[11][4]
        # In CBAR convention, theta_y is negative for positive Z-force
        rel_error = abs(abs(tip_ry) - theta) / theta
        assert rel_error < 0.002, (
            f"Tip rotation |{tip_ry:.6e}| vs analytical {theta:.6e}, "
            f"error = {rel_error:.4%}")

    def test_deflection_curve(self):
        """Full deflection curve: w(x) = Px^2(3L-x)/(6EI)."""
        P = 100.0; L = 1.0; E = 7.0e10; I = 8.333e-10
        max_error = 0.0
        for nid in range(2, 12):
            x = (nid - 1) * 0.1
            w_exact = P * x**2 * (3*L - x) / (6 * E * I)
            w_computed = self.sc.displacements[nid][2]
            if w_exact > 1e-12:
                error = abs(w_computed - w_exact) / w_exact
                max_error = max(max_error, error)
        assert max_error < 0.003, f"Max deflection curve error = {max_error:.4%}"

    def test_moment_reaction(self):
        """Moment reaction at root: |My| = P*L = 100*1 = 100 N-m."""
        my = self.sc.spc_forces[1][4]
        assert abs(abs(my) - 100.0) < 2.0, f"Moment reaction = {my}, expected ±100"


# ============================================================================
# SOL 103: Enhanced Plate Frequency Tests
# ============================================================================
class TestPlateFrequencies_Enhanced:
    """Simply-supported square plate natural frequencies.
    a=b=1.0m, t=0.01m, Steel (E=210GPa, rho=7850)
    D = E*t^3/(12*(1-nu^2)) = 2.1e11*1e-6/10.92 = 19230.8 N-m
    f_mn = (pi/2) * sqrt(D/(rho*t)) * (m^2/a^2 + n^2/b^2)
    f_11 = pi/2 * sqrt(19230.8/78.5) * 2 = 49.14 Hz

    Note: With coarse 4x4 mesh, expect ~10-20% error due to
    Mindlin plate discretization.
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_103(
            os.path.join(VALIDATION_DIR, "plate_modes", "plate_modes.bdf"))
        self.sc = self.results.subcases[0]

    def test_first_mode_order_magnitude(self):
        """First plate mode should be near f_11 analytical."""
        E = 2.1e11; nu = 0.3; t = 0.01; rho = 7850.0; a = 1.0
        D = E * t**3 / (12 * (1 - nu**2))
        f_11 = (np.pi / 2) * np.sqrt(D / (rho * t)) * (1/a**2 + 1/a**2)
        positive_freqs = [f for f in self.sc.frequencies if f > 1.0]
        if positive_freqs:
            f1 = positive_freqs[0]
            ratio = f1 / f_11
            # With coarse mesh and selective reduced integration, ~20% error
            assert 0.7 < ratio < 1.3, (
                f"f1 = {f1:.2f} Hz, analytical f_11 = {f_11:.2f} Hz, ratio = {ratio:.2f}")


# ============================================================================
# SOL 144: Goland Wing Trim Verification (enhanced)
# ============================================================================
class TestGolandTrim_Enhanced:
    """Enhanced Goland wing trim validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from nastaero.solvers.sol144 import solve_trim
        parser = BDFParser()
        model = parser.parse(os.path.join(VALIDATION_DIR,
                                           "goland_wing", "goland_static.bdf"))
        self.results = solve_trim(model)
        self.model = model
        self.sc = self.results.subcases[0]

    def test_trim_converged(self):
        """Trim should produce finite results."""
        assert len(self.sc.displacements) > 0
        assert self.sc.trim_variables is not None
        for label, val in self.sc.trim_variables.items():
            assert np.isfinite(val), f"{label} = {val} is not finite"

    def test_positive_lift(self):
        """Aero forces should produce positive lift (Fz > 0)."""
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        assert total_fz > 0, f"Total Fz = {total_fz:.2f}, should be positive"

    def test_aero_pressure_distribution(self):
        """Pressure coefficients should vary across span."""
        cp = self.sc.aero_pressures
        assert cp is not None
        assert len(cp) > 0
        cp_range = np.max(cp) - np.min(cp)
        assert cp_range > 1e-10, "All Cp values are identical"

    def test_structural_deformation(self):
        """Wing should deform under aero loads."""
        max_disp = max(np.max(np.abs(d)) for d in self.sc.displacements.values())
        assert max_disp > 1e-10, "No structural deformation"

    def test_root_fixed(self):
        """Root node should have zero displacement (clamped)."""
        root_disp = self.sc.displacements[1]
        assert np.max(np.abs(root_disp)) < 1e-10, \
            f"Root displacement = {root_disp}, should be zero"


# ============================================================================
# VM6: Fixed-Fixed Beam with Center Point Load
# Ref: Roark's Formulas for Stress and Strain
# ============================================================================
class TestVM6_FixedFixedBeam:
    """Fixed-fixed beam with center point load.
    P=10000N at midspan, L=10m, E=200GPa, I=8.333e-6 m^4.
    Analytical: delta_center = PL^3/(192EI) = 3.125e-2 m.
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm6_fixed_fixed_beam.bdf"))
        self.sc = self.results.subcases[0]

    def test_center_deflection(self):
        """Center node deflection vs analytical."""
        P = 10000; L = 10; E = 200e9; I = 8.333e-6
        delta_analytical = P * L**3 / (192 * E * I)
        delta_center = abs(self.sc.displacements[6][1])  # Y-direction
        rel_error = abs(delta_center - delta_analytical) / delta_analytical
        assert rel_error < 0.01, (
            f"Center |uy| = {delta_center:.6e} vs analytical {delta_analytical:.6e}, "
            f"error = {rel_error:.2%}")

    def test_symmetric_deflection(self):
        """Deflection should be symmetric about midspan."""
        for i in range(1, 6):
            d_left = abs(self.sc.displacements[1 + i][1])
            d_right = abs(self.sc.displacements[11 - i][1])
            assert abs(d_left - d_right) / max(d_left, 1e-15) < 0.001, (
                f"Asymmetry: node {1+i} = {d_left:.6e}, node {11-i} = {d_right:.6e}")

    def test_reactions(self):
        """Each end reaction = P/2 = 5000 N."""
        ry1 = abs(self.sc.spc_forces[1][1])
        ry11 = abs(self.sc.spc_forces[11][1])
        assert abs(ry1 - 5000.0) < 1.0, f"Ry at node 1 = {ry1:.2f}"
        assert abs(ry11 - 5000.0) < 1.0, f"Ry at node 11 = {ry11:.2f}"

    def test_fixed_end_moments(self):
        """Fixed-end moments = PL/8 = 12500 N-m."""
        mz1 = abs(self.sc.spc_forces[1][5])
        mz11 = abs(self.sc.spc_forces[11][5])
        assert abs(mz1 - 12500.0) < 50.0, f"Mz at node 1 = {mz1:.2f}"
        assert abs(mz11 - 12500.0) < 50.0, f"Mz at node 11 = {mz11:.2f}"

    def test_zero_rotation_at_ends(self):
        """Both ends should have zero rotation (fixed-fixed)."""
        for nid in [1, 11]:
            for dof in range(3, 6):
                assert abs(self.sc.displacements[nid][dof]) < 1e-12


# ============================================================================
# VM7: Cantilever Plate Strip (CQUAD4)
# Ref: Beam-equivalent approximation
# ============================================================================
class TestVM7_CantileverPlateStrip:
    """Cantilever plate strip: 8 CQUAD4 elements, 1 wide.
    L=1.0m, b=0.1m, t=0.01m, E=200GPa, P=100N.
    Analytical beam-equivalent: delta_tip ≈ PL^3/(3*D*b).
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm7_cantilever_plate_strip.bdf"))
        self.sc = self.results.subcases[0]

    def test_tip_deflection(self):
        """Tip deflection should match beam-equivalent."""
        E = 200e9; nu = 0.3; t = 0.01; b = 0.1; L = 1.0; P = 100
        D = E * t**3 / (12 * (1 - nu**2))
        delta_beam = P * L**3 / (3 * D * b)
        # Average tip deflection (2 tip nodes)
        d9 = abs(self.sc.displacements[9][2])
        d18 = abs(self.sc.displacements[18][2])
        avg_tip = (d9 + d18) / 2
        rel_error = abs(avg_tip - delta_beam) / delta_beam
        # ~10% error expected (plate vs beam, Poisson effect)
        assert rel_error < 0.15, (
            f"Avg tip = {avg_tip:.6e} vs beam approx {delta_beam:.6e}, "
            f"error = {rel_error:.2%}")

    def test_consistent_width(self):
        """Both tip nodes should have same deflection (uniform width)."""
        d9 = abs(self.sc.displacements[9][2])
        d18 = abs(self.sc.displacements[18][2])
        assert abs(d9 - d18) / max(d9, 1e-15) < 0.01

    def test_fixed_end(self):
        """Root nodes should have zero displacement."""
        for nid in [1, 10]:
            assert np.max(np.abs(self.sc.displacements[nid])) < 1e-12

    def test_increasing_deflection(self):
        """Deflection should increase monotonically from root to tip."""
        # Bottom row: nodes 1,2,...,9
        prev = 0.0
        for nid in range(1, 10):
            curr = abs(self.sc.displacements[nid][2])
            assert curr >= prev - 1e-12, f"Non-monotonic at node {nid}"
            prev = curr


# ============================================================================
# VM8: Portal Frame
# ============================================================================
class TestVM8_PortalFrame:
    """Portal frame: 2 columns + 1 beam, horizontal load at corner.
    Verifies multi-element frame with combined bending.
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm8_portal_frame.bdf"))
        self.sc = self.results.subcases[0]

    def test_horizontal_equilibrium(self):
        """Sum of horizontal reactions = applied horizontal force (5000N)."""
        # Base nodes are 1 and 6
        fx_sum = self.sc.spc_forces[1][0] + self.sc.spc_forces[6][0]
        assert abs(fx_sum + 5000.0) < 5.0, f"Sum Fx = {fx_sum}, expected -5000"

    def test_sway_direction(self):
        """Top nodes should move in +X (same direction as load)."""
        # Node 5 (top-left, loaded) and node 10 (top-right)
        d5_x = self.sc.displacements[5][0]
        d10_x = self.sc.displacements[10][0]
        assert d5_x > 0, f"Node 5 X-disp = {d5_x}, expected positive"
        assert d10_x > 0, f"Node 10 X-disp = {d10_x}, expected positive"

    def test_nearly_equal_sway(self):
        """Both top nodes should have nearly equal X-displacement (rigid beam)."""
        d5_x = self.sc.displacements[5][0]
        d10_x = self.sc.displacements[10][0]
        ratio = d5_x / d10_x if d10_x != 0 else float('inf')
        assert 0.99 < ratio < 1.01, f"Sway ratio = {ratio}"

    def test_base_fixed(self):
        """Base nodes should have zero displacement."""
        for nid in [1, 6]:
            assert np.max(np.abs(self.sc.displacements[nid])) < 1e-12


# ============================================================================
# VM9: Propped Cantilever with Uniform Load
# Ref: Roark's Formulas for Stress and Strain
# ============================================================================
class TestVM9_ProppedCantilever:
    """Propped cantilever: fixed at left, roller at right, uniform load.
    w=1000 N/m, L=10m, EI=200e9*8.333e-6.
    Analytical: R_right=3wL/8=3750, R_left=5wL/8=6250.
    Max deflection: wL^4/(185EI) at x≈0.4215L.
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_101(
            os.path.join(VM_DIR, "vm9_propped_cantilever.bdf"))
        self.sc = self.results.subcases[0]

    def test_total_reaction(self):
        """Total vertical reaction = total applied load = wL = 10000 N."""
        total_ry = self.sc.spc_forces[1][1] + self.sc.spc_forces[11][1]
        assert abs(total_ry - 10000.0) < 10.0, (
            f"Total Ry = {total_ry:.2f}, expected 10000")

    def test_reaction_distribution(self):
        """Roller reaction ≈ 3wL/8, Fixed reaction ≈ 5wL/8."""
        ry_fixed = self.sc.spc_forces[1][1]
        ry_roller = self.sc.spc_forces[11][1]
        # Allow ~2% error due to lumped load approximation
        assert abs(ry_fixed - 6250) / 6250 < 0.03, (
            f"Ry_fixed = {ry_fixed:.2f}, expected 6250")
        assert abs(ry_roller - 3750) / 3750 < 0.03, (
            f"Ry_roller = {ry_roller:.2f}, expected 3750")

    def test_max_deflection(self):
        """Max deflection near analytical value."""
        w = 1000; L = 10; E = 200e9; I = 8.333e-6
        delta_analytical = w * L**4 / (185 * E * I)
        max_defl = max(abs(self.sc.displacements[nid][1]) for nid in range(2, 11))
        rel_error = abs(max_defl - delta_analytical) / delta_analytical
        # ~5% error due to lumped load and discrete mesh
        assert rel_error < 0.10, (
            f"Max |uy| = {max_defl:.6e} vs analytical {delta_analytical:.6e}, "
            f"error = {rel_error:.2%}")

    def test_zero_displacement_at_supports(self):
        """Both supports should have zero Y-displacement."""
        assert abs(self.sc.displacements[1][1]) < 1e-12
        assert abs(self.sc.displacements[11][1]) < 1e-12

    def test_asymmetric_deflection(self):
        """Deflection should be asymmetric (max closer to fixed end)."""
        # Max should be near x=0.4215L ≈ node 5 (x=4m) or 6 (x=5m)
        max_node = max(range(2, 11), key=lambda n: abs(self.sc.displacements[n][1]))
        # Should be between nodes 4 and 8
        assert 4 <= max_node <= 8, (
            f"Max deflection at node {max_node}, expected near node 5-7")


# ============================================================================
# VM10: 8x8 Plate Modal Analysis
# Ref: Kirchhoff plate theory
# ============================================================================
class TestVM10_PlateModes8x8:
    """8x8 SS plate modes: better accuracy than 4x4 mesh.
    a=b=1.0m, t=0.01m, E=210GPa, nu=0.3, rho=7850.
    f_mn = (pi/2)*sqrt(D/(rho*t))*(m^2+n^2)/a^2
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = parse_and_solve_103(
            os.path.join(VM_DIR, "vm10_plate_modes_8x8.bdf"))
        self.sc = self.results.subcases[0]

    def _analytical_freq(self, m, n):
        E = 210e9; nu = 0.3; t = 0.01; rho = 7850; a = 1.0
        D = E * t**3 / (12 * (1 - nu**2))
        return (np.pi / 2) * np.sqrt(D / (rho * t)) * (m**2/a**2 + n**2/a**2)

    def test_first_mode(self):
        """f_11 should be close to analytical."""
        f11_analytical = self._analytical_freq(1, 1)
        positive_freqs = [f for f in self.sc.frequencies if f > 1.0]
        f1 = positive_freqs[0]
        rel_error = abs(f1 - f11_analytical) / f11_analytical
        assert rel_error < 0.05, (
            f"f1 = {f1:.4f} Hz vs analytical f_11 = {f11_analytical:.4f} Hz, "
            f"error = {rel_error:.2%}")

    def test_second_mode(self):
        """f_12 = f_21 (degenerate) should be close to analytical."""
        f12_analytical = self._analytical_freq(1, 2)
        positive_freqs = [f for f in self.sc.frequencies if f > 1.0]
        # Modes 2 and 3 should be degenerate f_12 = f_21
        if len(positive_freqs) >= 3:
            f2 = positive_freqs[1]
            rel_error = abs(f2 - f12_analytical) / f12_analytical
            assert rel_error < 0.05, (
                f"f2 = {f2:.4f} Hz vs analytical f_12 = {f12_analytical:.4f} Hz, "
                f"error = {rel_error:.2%}")

    def test_degenerate_modes(self):
        """Modes 2 and 3 should be degenerate (f_12 = f_21)."""
        positive_freqs = [f for f in self.sc.frequencies if f > 1.0]
        if len(positive_freqs) >= 3:
            f2 = positive_freqs[1]
            f3 = positive_freqs[2]
            assert abs(f2 - f3) / f2 < 0.01, (
                f"Modes 2,3 not degenerate: {f2:.4f} vs {f3:.4f}")

    def test_frequency_ordering(self):
        """Frequencies should be in ascending order."""
        positive_freqs = [f for f in self.sc.frequencies if f > 1.0]
        for i in range(1, len(positive_freqs)):
            assert positive_freqs[i] >= positive_freqs[i-1] - 1e-6

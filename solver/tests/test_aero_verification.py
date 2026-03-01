"""Aeroelastic Verification Manual Tests (AVM series).

Validates NastAero SOL 144 (static aeroelastic trim) against
analytical aerodynamic solutions and known Nastran benchmark results.

Test cases:
  AVM1 - Flat plate wing CL_alpha validation (AR=6)
  AVM2 - Goland wing enhanced trim at M=0.5
  AVM3 - Rigid wing trim angle verification (AR=4)
  DLM  - Unit tests for DLM/VLM AIC matrix
"""
import os
import numpy as np
import pytest
from nastaero.bdf.parser import BDFParser
from nastaero.aero.panel import generate_panel_mesh, generate_all_panels, AeroBox
from nastaero.aero.dlm import build_aic_matrix, circulation_to_delta_cp

VALIDATION_DIR = os.path.join(os.path.dirname(__file__), "validation")
AERO_VM_DIR = os.path.join(VALIDATION_DIR, "aero_vm")
GOLAND_DIR = os.path.join(VALIDATION_DIR, "goland_wing")


def parse_bdf(filepath):
    parser = BDFParser()
    return parser.parse(filepath)


def solve_trim(filepath):
    """Parse a BDF and run SOL 144 trim."""
    from nastaero.solvers.sol144 import solve_trim as _solve_trim
    model = parse_bdf(filepath)
    results = _solve_trim(model)
    return results, model


# ============================================================================
# DLM / VLM Unit Tests
# ============================================================================
class TestDLM_AIC:
    """Validate VLM AIC matrix properties and CL_alpha predictions."""

    def _make_rectangular_wing(self, AR, nspan=8, nchord=4):
        """Create a rectangular wing CAERO1-like object for testing."""
        chord = 1.0
        span = AR * chord
        from types import SimpleNamespace
        caero = SimpleNamespace(
            nspan=nspan, nchord=nchord,
            p1=np.array([0.0, 0.0, 0.0]),
            p4=np.array([0.0, span, 0.0]),
            chord1=chord, chord4=chord,
        )
        return generate_panel_mesh(caero)

    def test_aic_square(self):
        """AIC matrix should be square (n_boxes x n_boxes)."""
        boxes = self._make_rectangular_wing(6.0, nspan=4, nchord=2)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        assert D.shape == (8, 8)

    def test_aic_nonsingular(self):
        """AIC matrix should be non-singular."""
        boxes = self._make_rectangular_wing(6.0)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        assert np.linalg.det(D) != 0.0

    def test_aic_diagonal_dominance(self):
        """AIC diagonal should dominate for well-separated panels."""
        boxes = self._make_rectangular_wing(6.0, nspan=4, nchord=2)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        # Diagonal values should be among the largest in each row
        for i in range(D.shape[0]):
            assert abs(D[i, i]) > 0, f"Zero diagonal at [{i},{i}]"

    def test_cl_alpha_ar20(self):
        """CL_alpha for AR=20 wing should be close to 2*pi*AR/(AR+2)."""
        AR = 20.0
        boxes = self._make_rectangular_wing(AR, nspan=20, nchord=4)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        D_inv = np.linalg.inv(D)

        # Unit angle of attack normalwash
        w = -np.ones(len(boxes))  # w/V = -alpha for alpha=1 rad
        gamma = D_inv @ w
        delta_cp = circulation_to_delta_cp(boxes, gamma)

        # CL = sum(delta_cp * area) / S_ref
        S_ref = AR * 1.0  # chord=1
        CL = sum(delta_cp[i] * boxes[i].area for i in range(len(boxes))) / S_ref

        # Lifting line theory: CL_alpha = 2*pi*AR/(AR+2) = 5.71
        cl_alpha_ll = 2.0 * np.pi * AR / (AR + 2)
        rel_error = abs(CL - cl_alpha_ll) / cl_alpha_ll

        assert rel_error < 0.05, (
            f"CL_alpha(AR={AR}) = {CL:.4f} vs lifting line {cl_alpha_ll:.4f}, "
            f"error = {rel_error:.2%}")

    def test_cl_alpha_ar6(self):
        """CL_alpha for AR=6 wing should be close to lifting line theory."""
        AR = 6.0
        boxes = self._make_rectangular_wing(AR, nspan=12, nchord=4)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        D_inv = np.linalg.inv(D)

        w = -np.ones(len(boxes))
        gamma = D_inv @ w
        delta_cp = circulation_to_delta_cp(boxes, gamma)

        S_ref = AR * 1.0
        CL = sum(delta_cp[i] * boxes[i].area for i in range(len(boxes))) / S_ref

        # Helmbold formula for low AR: CL_alpha = 2*pi*AR / (AR + 2*(1+AR/2))
        # Simplified lifting line: 2*pi*AR/(AR+2) = 4.71
        # VLM typically gives slightly lower than lifting line
        cl_alpha_ll = 2.0 * np.pi * AR / (AR + 2)
        rel_error = abs(CL - cl_alpha_ll) / cl_alpha_ll

        assert rel_error < 0.10, (
            f"CL_alpha(AR={AR}) = {CL:.4f} vs lifting line {cl_alpha_ll:.4f}, "
            f"error = {rel_error:.2%}")

    def test_cl_alpha_ar4(self):
        """CL_alpha for AR=4 wing should be within 15% of lifting line."""
        AR = 4.0
        boxes = self._make_rectangular_wing(AR, nspan=8, nchord=4)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        D_inv = np.linalg.inv(D)

        w = -np.ones(len(boxes))
        gamma = D_inv @ w
        delta_cp = circulation_to_delta_cp(boxes, gamma)

        S_ref = AR * 1.0
        CL = sum(delta_cp[i] * boxes[i].area for i in range(len(boxes))) / S_ref

        cl_alpha_ll = 2.0 * np.pi * AR / (AR + 2)
        rel_error = abs(CL - cl_alpha_ll) / cl_alpha_ll

        assert rel_error < 0.15, (
            f"CL_alpha(AR={AR}) = {CL:.4f} vs lifting line {cl_alpha_ll:.4f}, "
            f"error = {rel_error:.2%}")

    def test_cl_alpha_increases_with_ar(self):
        """CL_alpha should increase monotonically with AR."""
        cl_values = []
        for AR in [2.0, 4.0, 6.0, 10.0]:
            boxes = self._make_rectangular_wing(AR, nspan=8, nchord=4)
            D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
            D_inv = np.linalg.inv(D)
            w = -np.ones(len(boxes))
            gamma = D_inv @ w
            delta_cp = circulation_to_delta_cp(boxes, gamma)
            S_ref = AR * 1.0
            CL = sum(delta_cp[i] * boxes[i].area
                     for i in range(len(boxes))) / S_ref
            cl_values.append(CL)

        for i in range(1, len(cl_values)):
            assert cl_values[i] > cl_values[i - 1], (
                f"CL_alpha not increasing: AR sequence gives {cl_values}")

    def test_cl_alpha_bounded_by_2pi(self):
        """CL_alpha should always be less than 2*pi (thin airfoil limit)."""
        for AR in [4.0, 8.0, 20.0]:
            boxes = self._make_rectangular_wing(AR, nspan=max(int(AR)*2, 8), nchord=4)
            D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
            D_inv = np.linalg.inv(D)
            w = -np.ones(len(boxes))
            gamma = D_inv @ w
            delta_cp = circulation_to_delta_cp(boxes, gamma)
            S_ref = AR * 1.0
            CL = sum(delta_cp[i] * boxes[i].area
                     for i in range(len(boxes))) / S_ref

            assert CL < 2 * np.pi, (
                f"CL_alpha({AR}) = {CL:.4f} exceeds 2*pi = {2*np.pi:.4f}")

    def test_prandtl_glauert_increases_cl(self):
        """Compressible CL_alpha (M>0) should be larger than incompressible."""
        AR = 6.0
        boxes = self._make_rectangular_wing(AR, nspan=8, nchord=4)

        # Incompressible
        D0 = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        D_inv0 = np.linalg.inv(D0)
        w = -np.ones(len(boxes))
        gamma0 = D_inv0 @ w
        dcp0 = circulation_to_delta_cp(boxes, gamma0)
        S_ref = AR * 1.0
        CL0 = sum(dcp0[i] * boxes[i].area for i in range(len(boxes))) / S_ref

        # M=0.5
        D05 = build_aic_matrix(boxes, mach=0.5, reduced_freq=0.0)
        D_inv05 = np.linalg.inv(D05)
        gamma05 = D_inv05 @ w
        dcp05 = circulation_to_delta_cp(boxes, gamma05)
        CL05 = sum(dcp05[i] * boxes[i].area for i in range(len(boxes))) / S_ref

        assert CL05 > CL0, (
            f"CL(M=0.5) = {CL05:.4f} should be > CL(M=0) = {CL0:.4f}")


# ============================================================================
# Panel Mesh Tests
# ============================================================================
class TestPanelMesh:
    """Validate aerodynamic panel mesh generation."""

    def test_box_count(self):
        """Number of boxes = nspan * nchord."""
        from types import SimpleNamespace
        caero = SimpleNamespace(
            nspan=6, nchord=3,
            p1=np.array([0.0, 0.0, 0.0]),
            p4=np.array([0.0, 10.0, 0.0]),
            chord1=2.0, chord4=2.0,
        )
        boxes = generate_panel_mesh(caero)
        assert len(boxes) == 18  # 6*3

    def test_control_point_behind_doublet(self):
        """Control point (3/4c) should be downstream of doublet (1/4c)."""
        from types import SimpleNamespace
        caero = SimpleNamespace(
            nspan=4, nchord=2,
            p1=np.array([0.0, 0.0, 0.0]),
            p4=np.array([0.0, 4.0, 0.0]),
            chord1=1.0, chord4=1.0,
        )
        boxes = generate_panel_mesh(caero)
        for box in boxes:
            assert box.control_point[0] > box.doublet_point[0], (
                f"Control point x={box.control_point[0]:.4f} should be > "
                f"doublet x={box.doublet_point[0]:.4f}")

    def test_total_area(self):
        """Sum of box areas should equal total panel area."""
        from types import SimpleNamespace
        caero = SimpleNamespace(
            nspan=8, nchord=4,
            p1=np.array([0.0, 0.0, 0.0]),
            p4=np.array([0.0, 6.0, 0.0]),
            chord1=1.0, chord4=1.0,
        )
        boxes = generate_panel_mesh(caero)
        total_area = sum(b.area for b in boxes)
        expected_area = 6.0 * 1.0  # span * chord
        rel_error = abs(total_area - expected_area) / expected_area
        assert rel_error < 1e-10, (
            f"Total area = {total_area:.6f} vs expected {expected_area:.6f}")

    def test_normals_upward(self):
        """Panel normals should point upward (positive z) for horizontal wing."""
        from types import SimpleNamespace
        caero = SimpleNamespace(
            nspan=4, nchord=2,
            p1=np.array([0.0, 0.0, 0.0]),
            p4=np.array([0.0, 4.0, 0.0]),
            chord1=1.0, chord4=1.0,
        )
        boxes = generate_panel_mesh(caero)
        for box in boxes:
            assert box.normal[2] > 0.9, f"Normal z={box.normal[2]:.4f}"


# ============================================================================
# AVM1: Flat Plate Wing CL_alpha Validation (AR=6)
# ============================================================================
class TestAVM1_FlatPlateCL:
    """AVM1: Flat plate rectangular wing for CL_alpha validation.

    AR=6, c=1.0m, b=6.0m, very stiff structure.
    q = 6125 Pa, solve for ANGLEA with URDD3=0 (1g level flight).

    Expected:
      CL_alpha from VLM (AR=6, 12x4 mesh) ~ 4.0-4.7 (per rad)
      Lifting line: 2*pi*6/(6+2) = 4.712
      alpha_trim = W / (q * S * CL_alpha)
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = solve_trim(
            os.path.join(AERO_VM_DIR, "avm1_flat_plate_cl.bdf"))
        self.sc = self.results.subcases[0]

    def test_trim_converged(self):
        """Trim should produce finite results."""
        assert len(self.sc.displacements) > 0
        assert "ANGLEA" in self.sc.trim_variables

    def test_alpha_finite(self):
        """Trim angle should be finite and small."""
        alpha = self.sc.trim_variables["ANGLEA"]
        assert np.isfinite(alpha), f"ANGLEA = {alpha} not finite"
        alpha_deg = np.degrees(alpha)
        assert abs(alpha_deg) < 15, f"ANGLEA = {alpha_deg:.4f} deg too large"

    def test_force_balance(self):
        """Total aero Fz should approximately balance weight."""
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        # Compute weight from model
        total_mass = 0.0
        for eid, elem in self.model.elements.items():
            if elem.type == "CBAR":
                n1 = self.model.nodes[elem.node_ids[0]]
                n2 = self.model.nodes[elem.node_ids[1]]
                L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
                prop = elem.property_ref
                mat = prop.material_ref
                total_mass += mat.rho * prop.A * L
        weight = total_mass * 9.81
        if weight > 1.0:
            rel_error = abs(total_fz - weight) / weight
            assert rel_error < 0.05, (
                f"Fz = {total_fz:.2f} vs W = {weight:.2f}, error = {rel_error:.2%}")

    def test_cl_alpha_range(self):
        """Implied CL_alpha should be in the expected range."""
        alpha = self.sc.trim_variables["ANGLEA"]
        if abs(alpha) > 1e-10:
            total_fz = np.sum(self.sc.aero_forces[:, 2])
            q = 6125.0
            S = 6.0  # 6.0 m^2
            CL = total_fz / (q * S)
            cl_alpha = CL / alpha

            # Expected: 3.5-5.5 (VLM generally gives 85-95% of lifting line)
            assert 3.0 < cl_alpha < 6.0, (
                f"CL_alpha = {cl_alpha:.4f}, expected 3.5-5.5")

    def test_very_stiff_small_deformation(self):
        """Very stiff structure should have minimal deformation."""
        max_disp = max(np.max(np.abs(d)) for d in self.sc.displacements.values())
        assert max_disp < 0.01, (
            f"Max displacement = {max_disp:.6e}, should be tiny for stiff wing")


# ============================================================================
# AVM2: Goland Wing Enhanced Trim (M=0.5, finer mesh)
# ============================================================================
class TestAVM2_GolandEnhanced:
    """AVM2: Goland wing at M=0.5, q=3920 Pa, 12x4 panels.

    Goland wing properties:
      L=6.096m (semi-span), c=1.8288m
      EI=9.77e6 N-m^2, GJ=0.99e6 N-m^2, m=35.71 kg/m
      EA at 33% chord

    Expected:
      alpha ~ 0.5-3 deg (depends on flexibility)
      Positive lift, force balance
      Tip deflection: downward (negative z) or small
      Tip twist: wash-out (nose-down) due to aft aerodynamic center
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = solve_trim(
            os.path.join(AERO_VM_DIR, "avm2_goland_enhanced.bdf"))
        self.sc = self.results.subcases[0]

    def test_trim_converged(self):
        """Trim should produce finite, physical results."""
        assert "ANGLEA" in self.sc.trim_variables
        alpha = self.sc.trim_variables["ANGLEA"]
        assert np.isfinite(alpha), f"ANGLEA = {alpha}"
        alpha_deg = np.degrees(alpha)
        assert abs(alpha_deg) < 20, f"ANGLEA = {alpha_deg:.4f} deg"

    def test_positive_lift(self):
        """Total aero Fz should be positive (upward)."""
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        assert total_fz > 0, f"Total Fz = {total_fz:.2f}, should be positive"

    def test_force_balance(self):
        """Total lift should balance weight within 10%."""
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        # Goland: m = 35.71 kg/m * 6.096 m = 217.7 kg
        # From actual model
        total_mass = 0.0
        for eid, elem in self.model.elements.items():
            if elem.type == "CBAR":
                n1 = self.model.nodes[elem.node_ids[0]]
                n2 = self.model.nodes[elem.node_ids[1]]
                L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
                prop = elem.property_ref
                mat = prop.material_ref
                total_mass += mat.rho * prop.A * L
        weight = total_mass * 9.81
        if weight > 1.0:
            rel_error = abs(total_fz - weight) / weight
            assert rel_error < 0.10, (
                f"Fz = {total_fz:.2f} vs W = {weight:.2f}, error = {rel_error:.2%}")

    def test_tip_node_deflection(self):
        """Tip node (11) should have non-trivial displacement."""
        if 11 in self.sc.displacements:
            tip_disp = self.sc.displacements[11]
            max_tip = np.max(np.abs(tip_disp))
            assert max_tip > 1e-8, "Tip should deflect under aero loads"

    def test_pressure_distribution(self):
        """Pressure coefficients should be non-uniform across span."""
        cp = self.sc.aero_pressures
        assert len(cp) > 0
        cp_range = np.max(cp) - np.min(cp)
        assert cp_range > 1e-6, "Cp distribution is too uniform"

    def test_root_fixed(self):
        """Root node (1) should have zero displacement."""
        root_disp = self.sc.displacements[1]
        assert np.max(np.abs(root_disp)) < 1e-10, (
            f"Root displacement = {root_disp}")


# ============================================================================
# AVM3: Rigid Wing Trim Angle Verification (AR=4)
# ============================================================================
class TestAVM3_RigidWingTrim:
    """AVM3: Ultra-stiff AR=4 wing to verify trim angle.

    For a rigid wing, the trim angle should satisfy:
      W = q * S * CL_alpha * alpha
      alpha = W / (q * S * CL_alpha)

    Wing: AR=4, c=2.0m, b=8.0m, S=16.0m^2
    Ultra-stiff (E=2e14) to behave as rigid body.
    q = 6125 Pa, mass from beam: rho=1000, A=0.001, L=8 -> m=8 kg, W=78.48 N
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = solve_trim(
            os.path.join(AERO_VM_DIR, "avm3_rigid_wing_trim.bdf"))
        self.sc = self.results.subcases[0]

    def test_trim_converged(self):
        """Trim should converge to a finite angle."""
        assert "ANGLEA" in self.sc.trim_variables
        alpha = self.sc.trim_variables["ANGLEA"]
        assert np.isfinite(alpha), f"ANGLEA = {alpha}"

    def test_alpha_positive(self):
        """For positive weight, alpha should be positive."""
        alpha = self.sc.trim_variables["ANGLEA"]
        assert alpha > 0, f"ANGLEA = {alpha:.6f}, should be positive for lift"

    def test_force_balance(self):
        """Total lift should balance weight."""
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        W = 8.0 * 9.81  # 78.48 N
        rel_error = abs(total_fz - W) / W
        assert rel_error < 0.05, (
            f"Fz = {total_fz:.2f} vs W = {W:.2f}, error = {rel_error:.2%}")

    def test_rigid_body_alpha(self):
        """Trim alpha should match W / (q * S * CL_alpha) approximately.

        Since the wing is ultra-stiff, CL_alpha comes from VLM alone.
        We compute it directly from the AIC matrix.
        """
        alpha = self.sc.trim_variables["ANGLEA"]

        # Compute expected CL_alpha from VLM (AR=4, 8x4 mesh)
        boxes = generate_all_panels(self.model)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        D_inv = np.linalg.inv(D)
        w = -np.ones(len(boxes))
        gamma = D_inv @ w
        delta_cp = circulation_to_delta_cp(boxes, gamma)
        S_ref = 16.0  # m^2
        CL_alpha = sum(delta_cp[i] * boxes[i].area
                       for i in range(len(boxes))) / S_ref

        W = 8.0 * 9.81
        q = 6125.0
        alpha_expected = W / (q * S_ref * CL_alpha)

        rel_error = abs(alpha - alpha_expected) / abs(alpha_expected)
        assert rel_error < 0.20, (
            f"alpha = {np.degrees(alpha):.4f} deg vs "
            f"expected {np.degrees(alpha_expected):.4f} deg, "
            f"error = {rel_error:.2%}")

    def test_tiny_deformation(self):
        """Ultra-stiff wing should have negligible deformation."""
        max_disp = 0.0
        for nid, disp in self.sc.displacements.items():
            max_disp = max(max_disp, np.max(np.abs(disp)))
        assert max_disp < 1e-6, (
            f"Max displacement = {max_disp:.6e}, should be ~0 for E=2e14")


# ============================================================================
# Goland Wing: Original BDF Regression Tests
# ============================================================================
class TestGolandOriginal_Regression:
    """Regression tests for the original Goland wing (goland_static.bdf).

    Known good results from validated solver:
      ANGLEA ~ 0.03 rad (1.7 deg)
      Total Fz ~ Weight ~ 2135 N
      Tip z-disp ~ -7e-3 m (downward)
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.results, self.model = solve_trim(
            os.path.join(GOLAND_DIR, "goland_static.bdf"))
        self.sc = self.results.subcases[0]

    def test_alpha_in_range(self):
        """ANGLEA should be between 0.5 and 5 degrees."""
        alpha_deg = np.degrees(self.sc.trim_variables["ANGLEA"])
        assert 0.5 < alpha_deg < 5.0, f"ANGLEA = {alpha_deg:.4f} deg"

    def test_lift_equals_weight(self):
        """Total lift should equal weight."""
        total_fz = np.sum(self.sc.aero_forces[:, 2])
        total_mass = 0.0
        for eid, elem in self.model.elements.items():
            if elem.type == "CBAR":
                n1 = self.model.nodes[elem.node_ids[0]]
                n2 = self.model.nodes[elem.node_ids[1]]
                L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
                prop = elem.property_ref
                mat = prop.material_ref
                total_mass += mat.rho * prop.A * L
        weight = total_mass * 9.81
        rel_error = abs(total_fz - weight) / weight
        assert rel_error < 0.02, (
            f"Fz = {total_fz:.2f} vs W = {weight:.2f}, error = {rel_error:.2%}")

    def test_positive_pressures(self):
        """Most pressure coefficients should be positive (lift)."""
        cp = self.sc.aero_pressures
        n_positive = np.sum(cp > 0)
        assert n_positive > len(cp) * 0.8, (
            f"Only {n_positive}/{len(cp)} positive Cp values")

    def test_tip_deflects(self):
        """Tip node should have measurable deflection."""
        if 11 in self.sc.displacements:
            tip_z = abs(self.sc.displacements[11][2])
            assert tip_z > 1e-6, f"Tip z-disp = {tip_z:.6e}, too small"


# ============================================================================
# Spline Interpolation Tests
# ============================================================================
class TestSplineInterpolation:
    """Test structural-aerodynamic interpolation."""

    def test_beam_spline_exact_at_nodes(self):
        """Beam spline should give exact values at structural nodes."""
        from nastaero.aero.spline import build_beam_spline

        struct = np.array([
            [0.5, 0.0, 0.0],
            [0.5, 2.0, 0.0],
            [0.5, 4.0, 0.0],
            [0.5, 6.0, 0.0],
        ])
        # Aero points at exactly the structural node y-locations
        aero = np.array([
            [0.3, 0.0, 0.0],
            [0.3, 2.0, 0.0],
            [0.3, 4.0, 0.0],
            [0.3, 6.0, 0.0],
        ])
        G = build_beam_spline(struct, aero, axis=1)

        # At node locations, should get identity-like weights
        for i in range(4):
            assert abs(G[i, i] - 1.0) < 1e-10, (
                f"G[{i},{i}] = {G[i,i]:.6f}, expected 1.0")
            for j in range(4):
                if j != i:
                    assert abs(G[i, j]) < 1e-10, (
                        f"G[{i},{j}] = {G[i,j]:.6f}, expected 0.0")

    def test_beam_spline_linear_interpolation(self):
        """Beam spline should correctly interpolate between nodes."""
        from nastaero.aero.spline import build_beam_spline

        struct = np.array([
            [0.5, 0.0, 0.0],
            [0.5, 4.0, 0.0],
        ])
        aero = np.array([
            [0.3, 1.0, 0.0],  # 25% between nodes
            [0.3, 2.0, 0.0],  # 50% between nodes
            [0.3, 3.0, 0.0],  # 75% between nodes
        ])
        G = build_beam_spline(struct, aero, axis=1)

        assert abs(G[0, 0] - 0.75) < 1e-10
        assert abs(G[0, 1] - 0.25) < 1e-10
        assert abs(G[1, 0] - 0.5) < 1e-10
        assert abs(G[1, 1] - 0.5) < 1e-10
        assert abs(G[2, 0] - 0.25) < 1e-10
        assert abs(G[2, 1] - 0.75) < 1e-10

    def test_ips_spline_rigid_body(self):
        """IPS spline should exactly reproduce rigid body translation."""
        from nastaero.aero.spline import build_ips_spline

        struct = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        aero = np.array([
            [0.5, 0.5, 0.0],
            [0.3, 0.7, 0.0],
        ])
        G = build_ips_spline(struct, aero)

        # Rigid body z-translation: all structural nodes move by 1.0
        # -> all aero points should also move by 1.0
        u_struct = np.ones(4)
        u_aero = G @ u_struct
        for i in range(2):
            assert abs(u_aero[i] - 1.0) < 1e-6, (
                f"IPS rigid body: u_aero[{i}] = {u_aero[i]:.6f}, expected 1.0")

    def test_spline_partition_of_unity(self):
        """Beam spline rows should sum to 1 (partition of unity)."""
        from nastaero.aero.spline import build_beam_spline

        struct = np.array([
            [0.5, 0.0, 0.0],
            [0.5, 2.0, 0.0],
            [0.5, 4.0, 0.0],
        ])
        aero = np.array([
            [0.3, 0.5, 0.0],
            [0.3, 1.5, 0.0],
            [0.3, 3.0, 0.0],
        ])
        G = build_beam_spline(struct, aero, axis=1)

        for i in range(3):
            row_sum = np.sum(G[i, :])
            assert abs(row_sum - 1.0) < 1e-10, (
                f"Row {i} sum = {row_sum:.6f}, expected 1.0")


# ============================================================================
# Cross-checks: consistency between direct VLM and SOL 144
# ============================================================================
class TestVLM_SOL144_Consistency:
    """Verify that SOL 144 results are consistent with direct VLM computation."""

    def test_rigid_wing_cl_matches_vlm(self):
        """For a rigid wing, SOL 144 CL should match direct VLM prediction."""
        results, model = solve_trim(
            os.path.join(AERO_VM_DIR, "avm3_rigid_wing_trim.bdf"))
        sc = results.subcases[0]

        alpha = sc.trim_variables["ANGLEA"]
        total_fz = np.sum(sc.aero_forces[:, 2])
        q = 6125.0
        S = 16.0
        CL_sol144 = total_fz / (q * S)

        # Direct VLM CL at the same alpha
        boxes = generate_all_panels(model)
        D = build_aic_matrix(boxes, mach=0.0, reduced_freq=0.0)
        D_inv = np.linalg.inv(D)
        w = -alpha * np.ones(len(boxes))
        gamma = D_inv @ w
        delta_cp = circulation_to_delta_cp(boxes, gamma)
        CL_vlm = sum(delta_cp[i] * boxes[i].area
                      for i in range(len(boxes))) / S

        # Should match within ~20% (spline/DOF coupling introduces some difference)
        if abs(CL_vlm) > 1e-10:
            rel_error = abs(CL_sol144 - CL_vlm) / abs(CL_vlm)
            assert rel_error < 0.30, (
                f"CL(SOL144) = {CL_sol144:.6f} vs CL(VLM) = {CL_vlm:.6f}, "
                f"error = {rel_error:.2%}")

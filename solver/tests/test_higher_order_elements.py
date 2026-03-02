"""Tests for CQUAD8 and CTRIA6 higher-order shell elements."""
from __future__ import annotations
import numpy as np
import pytest
from nastaero.bdf.cards.elements import CQUAD8, CTRIA6
from nastaero.bdf.cards.properties import PSHELL
from nastaero.bdf.cards.materials import MAT1
from nastaero.bdf.cards.grid import GRID
from nastaero.bdf.model import BDFModel
from nastaero.fem.dof_manager import DOFManager
from nastaero.fem.assembly import assemble_global_matrices
from nastaero.elements.quad8 import CQuad8Element
from nastaero.elements.tria6 import CTria6Element


# ============================================================
# Card parsing tests
# ============================================================

class TestCQUAD8Parsing:
    def test_basic_parsing(self):
        fields = ["CQUAD8", "1", "10",
                  "1", "2", "3", "4", "5", "6",
                  "7", "8"]
        e = CQUAD8.from_fields(fields)
        assert e.eid == 1
        assert e.pid == 10
        assert e.node_ids == [1, 2, 3, 4, 5, 6, 7, 8]
        assert e.type == "CQUAD8"

    def test_with_theta_zoffs(self):
        fields = ["CQUAD8", "5", "20",
                  "10", "20", "30", "40", "50", "60",
                  "70", "80", "", "", "", "", "45.0", "0.5"]
        e = CQUAD8.from_fields(fields)
        assert e.theta_mcid == pytest.approx(45.0)
        assert e.zoffs == pytest.approx(0.5)


class TestCTRIA6Parsing:
    def test_basic_parsing(self):
        fields = ["CTRIA6", "2", "10",
                  "1", "2", "3", "4", "5", "6"]
        e = CTRIA6.from_fields(fields)
        assert e.eid == 2
        assert e.pid == 10
        assert e.node_ids == [1, 2, 3, 4, 5, 6]
        assert e.type == "CTRIA6"

    def test_with_theta_zoffs(self):
        fields = ["CTRIA6", "3", "15",
                  "10", "20", "30", "40", "50", "60",
                  "30.0", "1.5"]
        e = CTRIA6.from_fields(fields)
        assert e.theta_mcid == pytest.approx(30.0)
        assert e.zoffs == pytest.approx(1.5)


# ============================================================
# Element formulation tests
# ============================================================

def _make_quad8_nodes(Lx=10.0, Ly=10.0):
    """Create 8 nodes for a rectangular CQUAD8 in the XY plane."""
    return np.array([
        [0, 0, 0],        # 0: corner
        [Lx, 0, 0],       # 1: corner
        [Lx, Ly, 0],      # 2: corner
        [0, Ly, 0],        # 3: corner
        [Lx/2, 0, 0],     # 4: midside 0-1
        [Lx, Ly/2, 0],    # 5: midside 1-2
        [Lx/2, Ly, 0],    # 6: midside 2-3
        [0, Ly/2, 0],     # 7: midside 3-0
    ], dtype=float)


def _make_tria6_nodes(L=10.0):
    """Create 6 nodes for a right-triangle CTRIA6 in the XY plane."""
    return np.array([
        [0, 0, 0],        # 0: corner
        [L, 0, 0],        # 1: corner
        [0, L, 0],        # 2: corner
        [L/2, 0, 0],      # 3: midside 0-1
        [L/2, L/2, 0],    # 4: midside 1-2
        [0, L/2, 0],      # 5: midside 2-0
    ], dtype=float)


class TestCQuad8Element:
    E = 70e3
    nu = 0.3
    t = 1.0
    rho = 2.7e-9

    def test_stiffness_shape(self):
        xyz = _make_quad8_nodes()
        elem = CQuad8Element(xyz, self.E, self.nu, self.t, self.rho)
        K = elem.stiffness_matrix()
        assert K.shape == (48, 48)

    def test_mass_shape(self):
        xyz = _make_quad8_nodes()
        elem = CQuad8Element(xyz, self.E, self.nu, self.t, self.rho)
        M = elem.mass_matrix()
        assert M.shape == (48, 48)

    def test_stiffness_symmetry(self):
        xyz = _make_quad8_nodes()
        elem = CQuad8Element(xyz, self.E, self.nu, self.t, self.rho)
        K = elem.stiffness_matrix()
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_stiffness_positive_semidefinite(self):
        xyz = _make_quad8_nodes()
        elem = CQuad8Element(xyz, self.E, self.nu, self.t, self.rho)
        K = elem.stiffness_matrix()
        eigvals = np.linalg.eigvalsh(K)
        # Should have 6 zero eigenvalues (rigid body modes) and rest positive
        assert np.sum(eigvals < -1e-6) == 0, f"Negative eigenvalues: {eigvals[eigvals < -1e-6]}"

    def test_mass_positive(self):
        xyz = _make_quad8_nodes()
        elem = CQuad8Element(xyz, self.E, self.nu, self.t, self.rho)
        M = elem.mass_matrix()
        diag = np.diag(M)
        assert np.all(diag >= 0)
        assert np.sum(diag > 0) > 0  # at least some positive

    def test_total_mass(self):
        Lx, Ly = 10.0, 10.0
        xyz = _make_quad8_nodes(Lx, Ly)
        elem = CQuad8Element(xyz, self.E, self.nu, self.t, self.rho)
        M = elem.mass_matrix()
        # Total translational mass (sum x-DOF diagonal)
        total = sum(M[6*i, 6*i] for i in range(8))
        expected = self.rho * self.t * Lx * Ly
        assert total == pytest.approx(expected, rel=0.05)

    def test_dof_count(self):
        xyz = _make_quad8_nodes()
        elem = CQuad8Element(xyz, self.E, self.nu, self.t, self.rho)
        assert elem.dof_count() == 48

    def test_shape_function_partition_of_unity(self):
        """Shape functions should sum to 1 at any point."""
        for xi, eta in [(-0.5, 0.3), (0.0, 0.0), (0.7, -0.2)]:
            N, _, _ = CQuad8Element.shape_functions(xi, eta)
            assert np.sum(N) == pytest.approx(1.0, abs=1e-12)

    def test_shape_function_at_nodes(self):
        """Shape function i should be 1 at node i, 0 at others."""
        node_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1),
                       (0, -1), (1, 0), (0, 1), (-1, 0)]
        for i, (xi, eta) in enumerate(node_coords):
            N, _, _ = CQuad8Element.shape_functions(xi, eta)
            for j in range(8):
                expected = 1.0 if i == j else 0.0
                assert N[j] == pytest.approx(expected, abs=1e-12), \
                    f"N[{j}] at node {i} ({xi},{eta}) = {N[j]}, expected {expected}"


class TestCTria6Element:
    E = 70e3
    nu = 0.3
    t = 1.0
    rho = 2.7e-9

    def test_stiffness_shape(self):
        xyz = _make_tria6_nodes()
        elem = CTria6Element(xyz, self.E, self.nu, self.t, self.rho)
        K = elem.stiffness_matrix()
        assert K.shape == (36, 36)

    def test_mass_shape(self):
        xyz = _make_tria6_nodes()
        elem = CTria6Element(xyz, self.E, self.nu, self.t, self.rho)
        M = elem.mass_matrix()
        assert M.shape == (36, 36)

    def test_stiffness_symmetry(self):
        xyz = _make_tria6_nodes()
        elem = CTria6Element(xyz, self.E, self.nu, self.t, self.rho)
        K = elem.stiffness_matrix()
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_stiffness_positive_semidefinite(self):
        xyz = _make_tria6_nodes()
        elem = CTria6Element(xyz, self.E, self.nu, self.t, self.rho)
        K = elem.stiffness_matrix()
        eigvals = np.linalg.eigvalsh(K)
        assert np.sum(eigvals < -1e-6) == 0, f"Negative eigenvalues: {eigvals[eigvals < -1e-6]}"

    def test_mass_positive(self):
        xyz = _make_tria6_nodes()
        elem = CTria6Element(xyz, self.E, self.nu, self.t, self.rho)
        M = elem.mass_matrix()
        diag = np.diag(M)
        assert np.all(diag >= 0)
        assert np.sum(diag > 0) > 0

    def test_total_mass(self):
        L = 10.0
        xyz = _make_tria6_nodes(L)
        elem = CTria6Element(xyz, self.E, self.nu, self.t, self.rho)
        M = elem.mass_matrix()
        total = sum(M[6*i, 6*i] for i in range(6))
        expected = self.rho * self.t * 0.5 * L * L
        assert total == pytest.approx(expected, rel=0.05)

    def test_dof_count(self):
        xyz = _make_tria6_nodes()
        elem = CTria6Element(xyz, self.E, self.nu, self.t, self.rho)
        assert elem.dof_count() == 36

    def test_shape_function_partition_of_unity(self):
        for L1, L2 in [(0.2, 0.3), (1./3., 1./3.), (0.1, 0.1)]:
            N, _, _ = CTria6Element.shape_functions(L1, L2)
            assert np.sum(N) == pytest.approx(1.0, abs=1e-12)

    def test_shape_function_at_nodes(self):
        """Shape function i should be 1 at node i."""
        node_coords = [(1, 0), (0, 1), (0, 0),
                        (0.5, 0.5), (0, 0.5), (0.5, 0)]
        for i, (L1, L2) in enumerate(node_coords):
            N, _, _ = CTria6Element.shape_functions(L1, L2)
            for j in range(6):
                expected = 1.0 if i == j else 0.0
                assert N[j] == pytest.approx(expected, abs=1e-12), \
                    f"N[{j}] at node {i} (L1={L1},L2={L2}) = {N[j]}, expected {expected}"


# ============================================================
# Assembly integration tests
# ============================================================

def _build_model_quad8():
    """Build a single CQUAD8 element model for assembly testing."""
    model = BDFModel()
    mat = MAT1()
    mat.mid = 1; mat.E = 70e3; mat.nu = 0.3; mat.rho = 2.7e-9
    mat.G = 70e3 / (2 * 1.3)
    model.materials[1] = mat

    prop = PSHELL()
    prop.pid = 10; prop.mid = 1; prop.t = 1.0
    model.properties[10] = prop

    xyz = _make_quad8_nodes()
    e = CQUAD8()
    e.eid = 1; e.pid = 10
    e.node_ids = list(range(1, 9))
    model.elements[1] = e

    for i in range(8):
        g = GRID()
        g.nid = i + 1; g.xyz = xyz[i]; g.xyz_global = xyz[i].copy()
        model.nodes[g.nid] = g

    model.cross_reference()
    return model


def _build_model_tria6():
    """Build a single CTRIA6 element model for assembly testing."""
    model = BDFModel()
    mat = MAT1()
    mat.mid = 1; mat.E = 70e3; mat.nu = 0.3; mat.rho = 2.7e-9
    mat.G = 70e3 / (2 * 1.3)
    model.materials[1] = mat

    prop = PSHELL()
    prop.pid = 10; prop.mid = 1; prop.t = 1.0
    model.properties[10] = prop

    xyz = _make_tria6_nodes()
    e = CTRIA6()
    e.eid = 1; e.pid = 10
    e.node_ids = list(range(1, 7))
    model.elements[1] = e

    for i in range(6):
        g = GRID()
        g.nid = i + 1; g.xyz = xyz[i]; g.xyz_global = xyz[i].copy()
        model.nodes[g.nid] = g

    model.cross_reference()
    return model


class TestCQUAD8Assembly:
    def test_assembly_shape(self):
        model = _build_model_quad8()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        assert K.shape[0] == 48  # 8 nodes * 6 DOFs
        assert M.shape[0] == 48

    def test_assembly_nonzero(self):
        model = _build_model_quad8()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        assert K.nnz > 0
        assert M.nnz > 0

    def test_assembly_symmetry(self):
        model = _build_model_quad8()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        diff_K = K - K.T
        assert abs(diff_K).max() < 1e-6
        diff_M = M - M.T
        assert abs(diff_M).max() < 1e-6

    def test_cross_reference(self):
        model = _build_model_quad8()
        elem = model.elements[1]
        assert elem.property_ref is not None
        assert elem.property_ref.material_ref is not None


class TestCTRIA6Assembly:
    def test_assembly_shape(self):
        model = _build_model_tria6()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        assert K.shape[0] == 36  # 6 nodes * 6 DOFs
        assert M.shape[0] == 36

    def test_assembly_nonzero(self):
        model = _build_model_tria6()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        assert K.nnz > 0
        assert M.nnz > 0

    def test_assembly_symmetry(self):
        model = _build_model_tria6()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        diff_K = K - K.T
        assert abs(diff_K).max() < 1e-6
        diff_M = M - M.T
        assert abs(diff_M).max() < 1e-6

    def test_cross_reference(self):
        model = _build_model_tria6()
        elem = model.elements[1]
        assert elem.property_ref is not None
        assert elem.property_ref.material_ref is not None

"""Tests for PCOMP composite laminate property."""
from __future__ import annotations
import numpy as np
import pytest
from nastaero.bdf.cards.properties import PCOMP, _rotate_Q
from nastaero.bdf.cards.materials import MAT1, MAT8
from nastaero.bdf.cards.elements import CQUAD4
from nastaero.bdf.cards.grid import GRID
from nastaero.bdf.model import BDFModel
from nastaero.fem.dof_manager import DOFManager
from nastaero.fem.assembly import assemble_global_matrices


class TestPCOMPParsing:
    def test_basic_laminate(self):
        fields = ["PCOMP", "10", "-0.25", "0.0", "", "", "", "", "",
                  "1", "0.125", "0.0", "YES",
                  "1", "0.125", "90.0", "YES"]
        p = PCOMP.from_fields(fields)
        assert p.pid == 10
        assert len(p.plies) == 2
        assert p.plies[0] == (1, 0.125, 0.0, "YES")
        assert p.plies[1] == (1, 0.125, 90.0, "YES")
        assert p.t == pytest.approx(0.25)
        assert p.mid == 1

    def test_sym_laminate(self):
        fields = ["PCOMP", "20", "", "", "", "", "", "", "SYM",
                  "1", "0.1", "0.0", "NO",
                  "1", "0.1", "45.0", "NO"]
        p = PCOMP.from_fields(fields)
        assert p.lam == "SYM"
        assert len(p.plies) == 4  # mirrored: [0, 45, 45, 0]
        assert p.plies[0][2] == pytest.approx(0.0)
        assert p.plies[1][2] == pytest.approx(45.0)
        assert p.plies[2][2] == pytest.approx(45.0)
        assert p.plies[3][2] == pytest.approx(0.0)
        assert p.t == pytest.approx(0.4)

    def test_header_fields(self):
        fields = ["PCOMP", "30", "0.5", "1.2", "100.0", "HILL", "20.0", "0.02", "",
                  "2", "0.2", "30.0", "YES"]
        p = PCOMP.from_fields(fields)
        assert p.z0 == pytest.approx(0.5)
        assert p.nsm == pytest.approx(1.2)
        assert p.sb == pytest.approx(100.0)
        assert p.ft == "HILL"
        assert p.tref == pytest.approx(20.0)
        assert p.ge == pytest.approx(0.02)

    def test_empty_ply_fields_skipped(self):
        fields = ["PCOMP", "40", "", "", "", "", "", "", "",
                  "1", "0.1", "0.0", "NO",
                  "", "", "", "",
                  "1", "0.1", "90.0", "NO"]
        p = PCOMP.from_fields(fields)
        assert len(p.plies) == 2


class TestPCOMPCLT:
    def _make_mat1(self, E=70e3, nu=0.3, rho=2.7e-9):
        m = MAT1()
        m.mid = 1; m.E = E; m.nu = nu; m.rho = rho
        m.G = E / (2 * (1 + nu))
        return m

    def _make_mat8(self, E1=135e3, E2=10e3, nu12=0.3, G12=5e3, rho=1.6e-9):
        m = MAT8()
        m.mid = 1; m.E1 = E1; m.E2 = E2; m.nu12 = nu12; m.G12 = G12
        m.rho = rho; m.E = E1; m.G = G12; m.nu = nu12
        return m

    def test_isotropic_laminate_equals_mat1(self):
        """A [0] laminate with MAT1 should give back the same E."""
        mat = self._make_mat1(E=70e3, nu=0.3)
        materials = {1: mat}
        fields = ["PCOMP", "1", "", "", "", "", "", "", "",
                  "1", "1.0", "0.0", "NO"]
        p = PCOMP.from_fields(fields)
        E_eq, nu_eq, t, rho_eq = p.equivalent_isotropic(materials)
        assert t == pytest.approx(1.0)
        assert E_eq == pytest.approx(70e3 / (1 - 0.3**2), rel=0.01)
        assert rho_eq == pytest.approx(2.7e-9)

    def test_cross_ply_mat8(self):
        """[0/90]s with MAT8 should give quasi-balanced E_eq."""
        mat = self._make_mat8()
        materials = {1: mat}
        fields = ["PCOMP", "2", "", "", "", "", "", "", "SYM",
                  "1", "0.125", "0.0", "NO",
                  "1", "0.125", "90.0", "NO"]
        p = PCOMP.from_fields(fields)
        E_eq, nu_eq, t, rho_eq = p.equivalent_isotropic(materials)
        assert t == pytest.approx(0.5)
        assert E_eq > 0
        # For [0/90]s, E_eq should be between E2 and E1
        assert E_eq > mat.E2
        assert E_eq < mat.E1 / (1 - mat.nu12**2) + 1
        assert rho_eq == pytest.approx(1.6e-9)

    def test_rotate_Q_identity_at_zero(self):
        Q = np.array([[100, 30, 0], [30, 50, 0], [0, 0, 20]], dtype=float)
        Qbar = _rotate_Q(Q, 0.0)
        np.testing.assert_allclose(Qbar, Q, atol=1e-10)

    def test_rotate_Q_symmetry(self):
        Q = np.array([[100, 30, 0], [30, 50, 0], [0, 0, 20]], dtype=float)
        Qbar = _rotate_Q(Q, 45.0)
        np.testing.assert_allclose(Qbar, Qbar.T, atol=1e-10)

    def test_caching(self):
        mat = self._make_mat1()
        materials = {1: mat}
        fields = ["PCOMP", "5", "", "", "", "", "", "", "",
                  "1", "0.5", "0.0", "NO"]
        p = PCOMP.from_fields(fields)
        r1 = p.equivalent_isotropic(materials)
        r2 = p.equivalent_isotropic(materials)
        assert r1 == r2


class TestPCOMPIntegration:
    def test_cross_reference(self):
        model = BDFModel()
        mat = MAT8()
        mat.mid = 1; mat.E1 = 135e3; mat.E2 = 10e3; mat.nu12 = 0.3
        mat.G12 = 5e3; mat.rho = 1.6e-9; mat.E = 135e3; mat.G = 5e3; mat.nu = 0.3
        model.materials[1] = mat

        fields = ["PCOMP", "10", "", "", "", "", "", "", "",
                  "1", "0.1", "0.0", "NO", "1", "0.1", "90.0", "NO"]
        p = PCOMP.from_fields(fields)
        model.properties[10] = p

        e = CQUAD4()
        e.eid = 1; e.pid = 10
        e.node_ids = [1, 2, 3, 4]
        model.elements[1] = e

        for i, (x, y) in enumerate([(0, 0), (10, 0), (10, 10), (0, 10)]):
            g = GRID()
            g.nid = i + 1
            g.xyz = np.array([float(x), float(y), 0.0])
            g.xyz_global = g.xyz.copy()
            model.nodes[g.nid] = g

        model.cross_reference()

        assert e.property_ref is p
        assert len(p.ply_materials) == 2
        assert p.ply_materials[0] is mat
        E_eq, _, _, _ = p.equivalent_isotropic()
        assert E_eq > 0

    def test_assembly_with_pcomp(self):
        model = BDFModel()
        mat = MAT1()
        mat.mid = 1; mat.E = 70e3; mat.nu = 0.3; mat.rho = 2.7e-9
        mat.G = 70e3 / (2 * 1.3)
        model.materials[1] = mat

        fields = ["PCOMP", "10", "", "", "", "", "", "", "",
                  "1", "1.0", "0.0", "NO"]
        p = PCOMP.from_fields(fields)
        model.properties[10] = p

        e = CQUAD4()
        e.eid = 1; e.pid = 10; e.node_ids = [1, 2, 3, 4]
        model.elements[1] = e

        for i, (x, y) in enumerate([(0, 0), (10, 0), (10, 10), (0, 10)]):
            g = GRID()
            g.nid = i + 1
            g.xyz = np.array([float(x), float(y), 0.0])
            g.xyz_global = g.xyz.copy()
            model.nodes[g.nid] = g

        model.cross_reference()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)

        assert K.shape[0] == 24
        assert K.nnz > 0
        # Symmetry check
        diff = K - K.T
        assert abs(diff).max() < 1e-6

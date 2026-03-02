"""Tests for CONM2 concentrated mass element."""
from __future__ import annotations
import numpy as np
import pytest
from nastaero.bdf.cards.mass import CONM2
from nastaero.bdf.cards.elements import CQUAD4
from nastaero.bdf.cards.properties import PSHELL
from nastaero.bdf.cards.materials import MAT1
from nastaero.bdf.cards.grid import GRID
from nastaero.bdf.model import BDFModel
from nastaero.fem.dof_manager import DOFManager
from nastaero.fem.assembly import assemble_global_matrices


class TestCONM2Parsing:
    def test_basic_fields(self):
        fields = ["CONM2", "10", "5", "", "100.0", "0.0", "0.0", "0.0", "",
                  "1.0", "0.0", "2.0", "0.0", "0.0", "3.0"]
        m = CONM2.from_fields(fields)
        assert m.eid == 10
        assert m.node_id == 5
        assert m.cid == -1
        assert m.mass == pytest.approx(100.0)
        assert m.I11 == pytest.approx(1.0)
        assert m.I21 == pytest.approx(0.0)
        assert m.I22 == pytest.approx(2.0)
        assert m.I31 == pytest.approx(0.0)
        assert m.I32 == pytest.approx(0.0)
        assert m.I33 == pytest.approx(3.0)

    def test_full_inertia_with_offdiag(self):
        fields = ["CONM2", "1", "1", "", "50.0", "1.0", "2.0", "3.0", "",
                  "10.0", "1.5", "20.0", "2.5", "3.5", "30.0"]
        m = CONM2.from_fields(fields)
        assert m.mass == pytest.approx(50.0)
        assert m.offset[0] == pytest.approx(1.0)
        assert m.offset[1] == pytest.approx(2.0)
        assert m.offset[2] == pytest.approx(3.0)
        assert m.I11 == pytest.approx(10.0)
        assert m.I21 == pytest.approx(1.5)
        assert m.I22 == pytest.approx(20.0)
        assert m.I31 == pytest.approx(2.5)
        assert m.I32 == pytest.approx(3.5)
        assert m.I33 == pytest.approx(30.0)

    def test_minimal_fields(self):
        fields = ["CONM2", "5", "3", "", "25.0"]
        m = CONM2.from_fields(fields)
        assert m.eid == 5
        assert m.mass == pytest.approx(25.0)
        np.testing.assert_allclose(m.offset, [0, 0, 0])
        assert m.I11 == 0.0
        assert m.I22 == 0.0
        assert m.I33 == 0.0

    def test_with_cid(self):
        fields = ["CONM2", "7", "2", "1", "10.0", "5.0", "0.0", "0.0"]
        m = CONM2.from_fields(fields)
        assert m.cid == 1
        assert m.offset[0] == pytest.approx(5.0)

    def test_node_ids_property(self):
        fields = ["CONM2", "1", "42", "", "1.0"]
        m = CONM2.from_fields(fields)
        assert m.node_ids == [42]
        assert m.type == "CONM2"


class TestCONM2Assembly:
    def _make_model_with_conm2(self, mass, offset=None, I11=0, I21=0, I22=0,
                                I31=0, I32=0, I33=0):
        """Create minimal model: single node + CONM2."""
        model = BDFModel()
        g = GRID()
        g.nid = 1
        g.xyz = np.array([0.0, 0.0, 0.0])
        g.xyz_global = g.xyz.copy()
        model.nodes[1] = g

        m = CONM2()
        m.eid = 1; m.node_id = 1; m.cid = -1; m.mass = mass
        m.offset = np.array(offset if offset else [0.0, 0.0, 0.0])
        m.I11 = I11; m.I21 = I21; m.I22 = I22
        m.I31 = I31; m.I32 = I32; m.I33 = I33
        model.masses[1] = m

        # Need a dummy element so assembly doesn't produce empty K
        # Add a grounded spring to avoid singular K
        from nastaero.bdf.cards.elements import CELAS2
        sp = CELAS2()
        sp.eid = 100; sp.k = 1.0; sp.g1 = 1; sp.c1 = 1; sp.g2 = 0; sp.c2 = 0
        model.springs[100] = sp

        model.cross_reference()
        return model

    def test_diagonal_mass(self):
        """Zero offset, diagonal inertia → translational + rotational diagonal."""
        model = self._make_model_with_conm2(
            mass=10.0, I11=1.0, I22=2.0, I33=3.0)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # Translational mass on diagonal
        for i in range(3):
            assert Md[i, i] == pytest.approx(10.0)

        # Rotational inertia on diagonal
        assert Md[3, 3] == pytest.approx(1.0)
        assert Md[4, 4] == pytest.approx(2.0)
        assert Md[5, 5] == pytest.approx(3.0)

    def test_off_diagonal_inertia(self):
        """Full 3x3 symmetric inertia with off-diagonal terms."""
        model = self._make_model_with_conm2(
            mass=5.0, I11=10.0, I21=1.5, I22=20.0, I31=2.5, I32=3.5, I33=30.0)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # Check full 3x3 rotational block (DOFs 3,4,5)
        I_expected = np.array([[10.0, 1.5, 2.5],
                               [1.5, 20.0, 3.5],
                               [2.5, 3.5, 30.0]])
        I_actual = Md[3:6, 3:6]
        np.testing.assert_allclose(I_actual, I_expected, atol=1e-10)

    def test_offset_parallel_axis(self):
        """Offset mass: parallel axis theorem adds m*(r·r*I - r⊗r) to inertia."""
        offset = [10.0, 0.0, 0.0]
        mass = 5.0
        model = self._make_model_with_conm2(mass=mass, offset=offset,
                                            I11=1.0, I22=2.0, I33=3.0)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # Parallel axis theorem: I_node = I_cg + m*(r·r*I - r⊗r)
        r = np.array(offset)
        r_sq = np.dot(r, r)  # 100
        I_cg = np.diag([1.0, 2.0, 3.0])
        I_parallel = I_cg + mass * (r_sq * np.eye(3) - np.outer(r, r))
        # r⊗r = [[100,0,0],[0,0,0],[0,0,0]], r·r = 100
        # I_parallel = [[1,0,0],[0,2,0],[0,0,3]] + 5*([[100,0,0],[0,100,0],[0,0,100]] - [[100,0,0],[0,0,0],[0,0,0]])
        # = [[1,0,0],[0,2,0],[0,0,3]] + [[0,0,0],[0,500,0],[0,0,500]]
        # = [[1,0,0],[0,502,0],[0,0,503]]

        I_actual = Md[3:6, 3:6]
        np.testing.assert_allclose(I_actual, I_parallel, atol=1e-10)

    def test_offset_coupling(self):
        """Offset creates translation-rotation coupling via m*skew(r)."""
        offset = [0.0, 0.0, 5.0]
        mass = 10.0
        model = self._make_model_with_conm2(mass=mass, offset=offset)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # skew([0,0,5]) = [[0,-5,0],[5,0,0],[0,0,0]]
        # m*S = [[0,-50,0],[50,0,0],[0,0,0]]
        # Upper-right: M[0:3, 3:6] = m*S
        expected_coupling = mass * np.array([[0, -5, 0],
                                              [5, 0, 0],
                                              [0, 0, 0]])
        actual_coupling = Md[0:3, 3:6]
        np.testing.assert_allclose(actual_coupling, expected_coupling, atol=1e-10)

        # Lower-left: M[3:6, 0:3] = (m*S)^T = m*S (since M is assembled as symmetric)
        actual_lower = Md[3:6, 0:3]
        np.testing.assert_allclose(actual_lower, expected_coupling.T, atol=1e-10)

    def test_mass_symmetry(self):
        """Full CONM2 with offset: M should be symmetric."""
        model = self._make_model_with_conm2(
            mass=8.0, offset=[3.0, 4.0, 5.0],
            I11=10.0, I21=1.0, I22=20.0, I31=2.0, I32=3.0, I33=30.0)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # M should be symmetric
        np.testing.assert_allclose(Md, Md.T, atol=1e-10)

    def test_total_mass_sum(self):
        """Multiple CONM2: translational mass should sum correctly."""
        model = BDFModel()
        total = 0.0
        for i in range(1, 4):
            g = GRID()
            g.nid = i; g.xyz = np.array([float(i), 0.0, 0.0])
            g.xyz_global = g.xyz.copy()
            model.nodes[i] = g

            m = CONM2()
            m.eid = i; m.node_id = i; m.cid = -1; m.mass = float(i) * 10
            m.offset = np.zeros(3)
            m.I11 = 0; m.I21 = 0; m.I22 = 0; m.I31 = 0; m.I32 = 0; m.I33 = 0
            model.masses[i] = m
            total += m.mass

        model.cross_reference()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # Sum of all x-translation diagonal should be total mass
        mass_sum = sum(Md[dof_mgr.get_node_dofs(i)[0], dof_mgr.get_node_dofs(i)[0]]
                       for i in range(1, 4))
        assert mass_sum == pytest.approx(total)

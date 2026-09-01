"""Tests for CONM2 concentrated mass element."""
from __future__ import annotations
import numpy as np
import pytest
from ascent_load.bdf.cards.mass import CONM2
from ascent_load.bdf.cards.elements import CQUAD4
from ascent_load.bdf.cards.properties import PSHELL
from ascent_load.bdf.cards.materials import MAT1
from ascent_load.bdf.cards.grid import GRID
from ascent_load.bdf.model import BDFModel
from ascent_load.fem.dof_manager import DOFManager
from ascent_load.fem.assembly import assemble_global_matrices


class TestCONM2Parsing:
    def test_basic_fields(self):
        fields = ["CONM2", "10", "5", "", "100.0", "0.0", "0.0", "0.0", "",
                  "1.0", "0.0", "2.0", "0.0", "0.0", "3.0"]
        m = CONM2.from_fields(fields)
        assert m.eid == 10
        assert m.node_id == 5
        assert m.cid == 0  # QRG: 공란 = 0 (기본좌표계 오프셋)
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
        m.eid = 1; m.node_id = 1; m.cid = 0; m.mass = mass
        m.offset = np.array(offset if offset else [0.0, 0.0, 0.0])
        m.I11 = I11; m.I21 = I21; m.I22 = I22
        m.I31 = I31; m.I32 = I32; m.I33 = I33
        model.masses[1] = m

        # Need a dummy element so assembly doesn't produce empty K
        # Add a grounded spring to avoid singular K
        from ascent_load.bdf.cards.elements import CELAS2
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
        K, M, _ = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # Translational mass on diagonal
        for i in range(3):
            assert Md[i, i] == pytest.approx(10.0)

        # Rotational inertia on diagonal
        assert Md[3, 3] == pytest.approx(1.0)
        assert Md[4, 4] == pytest.approx(2.0)
        assert Md[5, 5] == pytest.approx(3.0)

    def test_off_diagonal_inertia(self):
        """Full 3x3 symmetric inertia with off-diagonal terms.

        MSC 규약: I21/I31/I32는 관성곱의 크기이고 텐서에는 음부호가
        자동으로 붙는다.
        """
        model = self._make_model_with_conm2(
            mass=5.0, I11=10.0, I21=1.5, I22=20.0, I31=2.5, I32=3.5, I33=30.0)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M, _ = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # Check full 3x3 rotational block (DOFs 3,4,5)
        I_expected = np.array([[10.0, -1.5, -2.5],
                               [-1.5, 20.0, -3.5],
                               [-2.5, -3.5, 30.0]])
        I_actual = Md[3:6, 3:6]
        np.testing.assert_allclose(I_actual, I_expected, atol=1e-10)

    def test_offset_parallel_axis(self):
        """Offset mass: parallel axis theorem adds m*(r·r*I - r⊗r) to inertia."""
        offset = [10.0, 0.0, 0.0]
        mass = 5.0
        model = self._make_model_with_conm2(mass=mass, offset=offset,
                                            I11=1.0, I22=2.0, I33=3.0)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M, _ = assemble_global_matrices(model, dof_mgr)
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
        K, M, _ = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # skew([0,0,5]) = [[0,-5,0],[5,0,0],[0,0,0]]
        # v_cg = u_dot + omega x r = u_dot - skew(r) omega 이므로
        # Upper-right: M[0:3, 3:6] = -m*skew(r)
        expected_coupling = -mass * np.array([[0, -5, 0],
                                              [5, 0, 0],
                                              [0, 0, 0]])
        actual_coupling = Md[0:3, 3:6]
        np.testing.assert_allclose(actual_coupling, expected_coupling, atol=1e-10)

        # Lower-left: M[3:6, 0:3] = (-m*S)^T = +m*S
        actual_lower = Md[3:6, 0:3]
        np.testing.assert_allclose(actual_lower, expected_coupling.T, atol=1e-10)

    def test_mass_symmetry(self):
        """Full CONM2 with offset: M should be symmetric."""
        model = self._make_model_with_conm2(
            mass=8.0, offset=[3.0, 4.0, 5.0],
            I11=10.0, I21=1.0, I22=20.0, I31=2.0, I32=3.0, I33=30.0)
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M, _ = assemble_global_matrices(model, dof_mgr)
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
            m.eid = i; m.node_id = i; m.cid = 0; m.mass = float(i) * 10
            m.offset = np.zeros(3)
            m.I11 = 0; m.I21 = 0; m.I22 = 0; m.I31 = 0; m.I32 = 0; m.I33 = 0
            model.masses[i] = m
            total += m.mass

        model.cross_reference()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        K, M, _ = assemble_global_matrices(model, dof_mgr)
        Md = M.toarray()

        # Sum of all x-translation diagonal should be total mass
        mass_sum = sum(Md[dof_mgr.get_node_dofs(i)[0], dof_mgr.get_node_dofs(i)[0]]
                       for i in range(1, 4))
        assert mass_sum == pytest.approx(total)


class TestCONM2Semantics2026Audit:
    """CONM2 규약 회귀 시험 (2026-08 감사).

    - 공란 CID는 QRG 기본값 0이고, 명시적 -1은 X1~X3가 기본좌표계
      기준 질량 CG의 절대좌표라는 별개 의미다.
    - 오프셋 병진-회전 결합은 M[병진,회전] = -m*skew(r)다.
    """

    def _mass_matrix(self, cid, xyz_node, x123, mass=3.0):
        from ascent_load.bdf.cards.elements import CELAS2
        model = BDFModel()
        g = GRID()
        g.nid = 1
        g.xyz = np.array(xyz_node, dtype=float)
        g.xyz_global = g.xyz.copy()
        model.nodes[1] = g

        c = CONM2()
        c.eid = 1; c.node_id = 1; c.cid = cid; c.mass = mass
        c.offset = np.array(x123, dtype=float)
        model.masses[1] = c

        sp = CELAS2()
        sp.eid = 100; sp.k = 1.0; sp.g1 = 1; sp.c1 = 1; sp.g2 = 0; sp.c2 = 0
        model.springs[100] = sp

        model.cross_reference()
        dof_mgr = DOFManager(sorted(model.nodes.keys()))
        _, M, _ = assemble_global_matrices(model, dof_mgr)
        return M.toarray()

    def test_blank_cid_defaults_to_zero(self):
        """공란 CID는 -1이 아니라 0이어야 한다 (QRG 기본값)."""
        m = CONM2.from_fields(["CONM2", "1", "1", "", "2.0"])
        assert m.cid == 0

    def test_offset_coupling_kinetic_energy(self):
        """강체 운동의 운동에너지가 1/2 m |u + w x r|^2 과 일치해야 한다.

        결합 부호가 반대면 오프셋이 절점을 통해 반전되어
        (1-L) 대신 (1+L)이 나온다.
        """
        mass, L = 3.0, 7.0
        r = np.array([0.0, L, 0.0])
        M = self._mass_matrix(0, [0.0, 0.0, 0.0], r, mass)

        v = np.zeros(6)
        v[0] = 1.0   # u = (1, 0, 0)
        v[5] = 1.0   # theta = (0, 0, 1)
        ke = 0.5 * v @ M @ v
        v_cg = np.array([1.0, 0.0, 0.0]) + np.cross([0.0, 0.0, 1.0], r)
        assert ke == pytest.approx(0.5 * mass * v_cg @ v_cg)

    def test_cid_minus_one_is_absolute_cg(self):
        """CID=-1의 X1~X3는 절대좌표이므로 등가 오프셋과 같아야 한다."""
        mass, L = 3.0, 7.0
        grid = np.array([10.0, 20.0, 30.0])
        M_off = self._mass_matrix(0, [0.0, 0.0, 0.0], [0.0, L, 0.0], mass)
        M_abs = self._mass_matrix(-1, grid, grid + np.array([0.0, L, 0.0]),
                                  mass)
        np.testing.assert_allclose(M_abs, M_off, atol=1e-10)

    def test_cid_minus_one_cg_at_grid_has_no_coupling(self):
        """CID=-1 좌표가 그리드와 같으면 팔이 0이라 결합항이 없어야 한다."""
        grid = [2761.38, 0.0, 1851.62]
        M = self._mass_matrix(-1, grid, grid, 1.4e-4)
        np.testing.assert_allclose(M[0:3, 3:6], 0.0, atol=1e-20)

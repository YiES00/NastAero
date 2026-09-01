"""Tests for element stress recovery."""
from __future__ import annotations

import numpy as np
import pytest

from ascent_load.elements.quad4 import CQuad4Element
from ascent_load.elements.tria3 import CTria3Element
from ascent_load.fem.stress_recovery import (
    _recover_cquad4,
    _recover_ctria3,
    _recover_cbar,
    _recover_crod,
    compute_von_mises,
)


class TestVonMises:
    """Test von Mises computation from stress components."""

    def test_uniaxial(self):
        """Uniaxial tension: vm = |σxx|."""
        S = np.array([[100.0, 0, 0, 0, 0, 0]])
        vm = compute_von_mises(S)
        assert vm[0] == pytest.approx(100.0, rel=1e-10)

    def test_pure_shear(self):
        """Pure shear: vm = sqrt(3) * τ."""
        S = np.array([[0, 0, 50.0, 0, 0, 0]])
        vm = compute_von_mises(S)
        assert vm[0] == pytest.approx(50.0 * np.sqrt(3), rel=1e-10)

    def test_biaxial_equal(self):
        """Equal biaxial: vm = σ (for σxx = σyy = σ)."""
        sigma = 120.0
        S = np.array([[sigma, sigma, 0, 0, 0, 0]])
        vm = compute_von_mises(S)
        assert vm[0] == pytest.approx(sigma, rel=1e-10)

    def test_membrane_plus_bending(self):
        """With bending, von Mises should use max of top/bot surfaces."""
        # Pure bending: σxx_bend = ±100 MPa, no membrane
        S = np.array([[0, 0, 0, 100.0, 0, 0]])
        vm = compute_von_mises(S)
        # top surface: σxx = 0 + 100 = 100
        # bot surface: σxx = 0 - 100 = -100
        # vm = max(100, 100) = 100
        assert vm[0] == pytest.approx(100.0, rel=1e-10)

    def test_membrane_and_bending_asymmetric(self):
        """Membrane + bending: top and bottom differ."""
        # Membrane σxx = 50, bending σxx = 30
        S = np.array([[50.0, 0, 0, 30.0, 0, 0]])
        vm = compute_von_mises(S)
        # top: σxx = 50 + 30 = 80, vm = 80
        # bot: σxx = 50 - 30 = 20, vm = 20
        assert vm[0] == pytest.approx(80.0, rel=1e-10)

    def test_vectorized(self):
        """Test batch computation."""
        S = np.array([
            [100, 0, 0, 0, 0, 0],
            [0, 100, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 50],   # pure bending shear
        ], dtype=float)
        vm = compute_von_mises(S)
        assert len(vm) == 3
        assert vm[0] == pytest.approx(100.0, rel=1e-10)
        assert vm[1] == pytest.approx(100.0, rel=1e-10)
        # bending shear ±50: top σxy=50, bot σxy=-50 → same vm
        assert vm[2] == pytest.approx(50.0 * np.sqrt(3), rel=1e-10)


class TestCQuad4Stress:
    """Test CQUAD4 stress recovery with known deformation."""

    @pytest.fixture
    def flat_plate_model(self):
        """Create a simple mock model with one CQUAD4 element."""
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        model = MagicMock()
        # 4 nodes forming a 100x100mm plate
        nodes = {}
        for i, (x, y) in enumerate([(0, 0), (100, 0), (100, 100), (0, 100)]):
            nid = i + 1
            node = SimpleNamespace(
                xyz=np.array([float(x), float(y), 0.0]),
                xyz_global=np.array([float(x), float(y), 0.0]),
            )
            nodes[nid] = node
        model.nodes = nodes

        # Element
        elem = SimpleNamespace(
            type="CQUAD4",
            pid=1,
            node_ids=[1, 2, 3, 4],
        )
        model.elements = {1: elem}

        # Property: PSHELL, t=2mm, aluminum
        prop = SimpleNamespace(
            pid=1, mid=1, t=2.0,
            material_ref=SimpleNamespace(E=70000.0, nu=0.3, rho=2.7e-9),
        )
        model.properties = {1: prop}
        model.materials = {1: prop.material_ref}

        # DOF manager mock
        dof_mgr = MagicMock()
        f_dofs = list(range(24))  # all 24 DOFs free
        f_dof_index = {d: d for d in f_dofs}

        # get_dof: node i, comp c → (i-1)*6 + (c-1)
        def mock_get_dof(nid, comp):
            return (nid - 1) * 6 + (comp - 1)
        dof_mgr.get_dof = mock_get_dof

        return model, dof_mgr, f_dofs, f_dof_index

    def test_uniform_stretch_x(self, flat_plate_model):
        """Uniform stretch in x: should give σxx > 0, σyy from Poisson."""
        model, dof_mgr, f_dofs, f_dof_index = flat_plate_model
        elem = model.elements[1]

        E = 70000.0; nu = 0.3; t = 2.0
        dx = 1.0  # 1 mm stretch in x

        # Displacement: nodes 2,3 move +dx in x
        u_free = np.zeros(24)
        u_free[6] = dx   # node 2, DOF 1 (x)
        u_free[12] = dx  # node 3, DOF 1 (x)

        s_mem, s_bend = _recover_cquad4(
            elem, model, dof_mgr, f_dof_index, u_free)

        # Membrane strain: εxx = dx/100 = 0.01
        eps_xx = dx / 100.0
        expected_sxx = E / (1 - nu**2) * eps_xx
        expected_syy = E / (1 - nu**2) * nu * eps_xx

        assert s_mem[0] == pytest.approx(expected_sxx, rel=0.05)
        assert s_mem[1] == pytest.approx(expected_syy, rel=0.05)
        assert abs(s_mem[2]) < 1e-6  # no shear
        # No bending for in-plane stretch
        assert np.max(np.abs(s_bend)) < 1e-3


class TestCTria3Stress:
    """Test CTRIA3 stress recovery."""

    def test_uniform_stretch(self):
        """Uniform stretch gives correct constant stress."""
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        model = MagicMock()
        nodes = {}
        for i, (x, y) in enumerate([(0, 0), (100, 0), (50, 86.6)]):
            nid = i + 1
            nodes[nid] = SimpleNamespace(
                xyz=np.array([float(x), float(y), 0.0]),
                xyz_global=np.array([float(x), float(y), 0.0]),
            )
        model.nodes = nodes

        elem = SimpleNamespace(
            type="CTRIA3", pid=1, node_ids=[1, 2, 3],
        )
        model.elements = {1: elem}

        prop = SimpleNamespace(
            pid=1, mid=1, t=2.0,
            material_ref=SimpleNamespace(E=70000.0, nu=0.3, rho=2.7e-9),
        )
        model.properties = {1: prop}
        model.materials = {1: prop.material_ref}

        dof_mgr = MagicMock()
        f_dofs = list(range(18))
        f_dof_index = {d: d for d in f_dofs}
        dof_mgr.get_dof = lambda nid, comp: (nid - 1) * 6 + (comp - 1)

        # Stretch in x: node 2 moves +1mm in x
        u_free = np.zeros(18)
        u_free[6] = 1.0  # node 2, x-displacement

        s_mem, s_bend = _recover_ctria3(
            elem, model, dof_mgr, f_dof_index, u_free)

        # Stress should be nonzero in xx
        assert abs(s_mem[0]) > 0.1
        # Bending should be near-zero for in-plane load
        assert np.max(np.abs(s_bend)) < 1e-3

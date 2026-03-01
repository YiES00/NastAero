"""Tests for element stiffness and mass matrices."""
import numpy as np
import pytest
from nastaero.elements.bar import CBarElement
from nastaero.elements.quad4 import CQuad4Element
from nastaero.elements.tria3 import CTria3Element


class TestCBarElement:
    def setup_method(self):
        # Unit beam along X, v-vector along Y
        self.bar = CBarElement(
            node1_xyz=np.array([0.0, 0.0, 0.0]),
            node2_xyz=np.array([1.0, 0.0, 0.0]),
            v_vector=np.array([0.0, 1.0, 0.0]),
            E=200e9, G=77e9, A=0.01, I1=8.33e-6, I2=8.33e-6, J=1.41e-5,
            rho=7850.0, nsm=0.0,
        )

    def test_stiffness_shape(self):
        K = self.bar.stiffness_matrix()
        assert K.shape == (12, 12)

    def test_stiffness_symmetry(self):
        K = self.bar.stiffness_matrix()
        assert np.allclose(K, K.T, atol=1e-6)

    def test_mass_shape(self):
        M = self.bar.mass_matrix()
        assert M.shape == (12, 12)

    def test_mass_symmetry(self):
        M = self.bar.mass_matrix()
        assert np.allclose(M, M.T, atol=1e-6)

    def test_axial_stiffness(self):
        """EA/L term at (0,0)."""
        K = self.bar._local_stiffness()
        EA_L = 200e9 * 0.01 / 1.0
        assert K[0, 0] == pytest.approx(EA_L, rel=1e-6)

    def test_mass_total(self):
        """Total translational mass check via consistent mass matrix.

        For consistent mass: M[0,0] = mL/3, M[0,6] = mL/6
        Row sum of axial DOFs = mL/3 + mL/6 = mL/2 per node → total = mL.
        """
        M = self.bar.mass_matrix()
        total_mass = 7850.0 * 0.01 * 1.0
        # For consistent mass, sum of all axial mass entries = total_mass
        axial_mass = M[0, 0] + M[0, 6] + M[6, 0] + M[6, 6]
        assert axial_mass == pytest.approx(total_mass, rel=0.01)


class TestCQuad4Element:
    def setup_method(self):
        # Unit square in XY plane
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self.quad = CQuad4Element(nodes, E=200e9, nu=0.3, t=0.01, rho=7850.0)

    def test_stiffness_shape(self):
        K = self.quad.stiffness_matrix()
        assert K.shape == (24, 24)

    def test_stiffness_symmetry(self):
        K = self.quad.stiffness_matrix()
        assert np.allclose(K, K.T, atol=1.0)  # Absolute tolerance for large values

    def test_mass_shape(self):
        M = self.quad.mass_matrix()
        assert M.shape == (24, 24)

    def test_mass_positive_diagonal(self):
        M = self.quad.mass_matrix()
        diag = np.diag(M)
        # Translational DOFs should have positive mass
        for i in range(4):
            for j in range(3):  # x, y, z
                assert diag[i*6 + j] > 0, f"Node {i+1}, DOF {j+1}: mass={diag[i*6+j]}"


class TestCTria3Element:
    def setup_method(self):
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self.tri = CTria3Element(nodes, E=200e9, nu=0.3, t=0.01, rho=7850.0)

    def test_stiffness_shape(self):
        K = self.tri.stiffness_matrix()
        assert K.shape == (18, 18)

    def test_stiffness_symmetry(self):
        K = self.tri.stiffness_matrix()
        assert np.allclose(K, K.T, atol=1.0)

    def test_mass_shape(self):
        M = self.tri.mass_matrix()
        assert M.shape == (18, 18)

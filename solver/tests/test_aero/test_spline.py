"""Tests for structural-aerodynamic spline interpolation."""
import numpy as np
import pytest
from nastaero.aero.spline import build_ips_spline, build_beam_spline, _green_function


class TestGreenFunction:
    def test_zero(self):
        """G(0) = 0."""
        assert _green_function(0.0) == pytest.approx(0.0)

    def test_positive(self):
        """G(r) should be defined for positive r."""
        g = _green_function(1.0)
        assert isinstance(g, float)

    def test_monotonic(self):
        """G(r) should increase with r for large enough r."""
        g1 = _green_function(2.0)
        g2 = _green_function(10.0)
        assert g2 > g1


class TestIPSSpline:
    def test_interpolation_identity(self):
        """If aero points == struct points, G_ka should be ~identity."""
        struct = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        G = build_ips_spline(struct, struct.copy())
        assert np.allclose(G, np.eye(3), atol=1e-6)

    def test_rigid_body_exact(self):
        """Constant displacement should be interpolated exactly."""
        struct = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        aero = np.array([
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
        ])
        G = build_ips_spline(struct, aero)
        # Constant z-displacement of 1.0 at all struct nodes
        z_struct = np.ones(4)
        z_aero = G @ z_struct
        assert np.allclose(z_aero, 1.0, atol=1e-6)

    def test_linear_field_exact(self):
        """Linear displacement field should be reproduced exactly."""
        struct = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        aero = np.array([
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
        ])
        G = build_ips_spline(struct, aero)
        # Linear field: z = 2*x + 3*y
        z_struct = 2.0 * struct[:, 0] + 3.0 * struct[:, 1]
        z_aero = G @ z_struct
        z_expected = 2.0 * aero[:, 0] + 3.0 * aero[:, 1]
        assert np.allclose(z_aero, z_expected, atol=1e-6)

    def test_output_shape(self):
        """G_ka shape should be (n_aero, n_struct)."""
        struct = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        aero = np.array([[0.5, 0, 0], [1.5, 0, 0]], dtype=float)
        G = build_ips_spline(struct, aero)
        assert G.shape == (2, 3)

    def test_empty_inputs(self):
        """Empty inputs should return empty matrix."""
        G = build_ips_spline(np.zeros((0, 3)), np.zeros((0, 3)))
        assert G.shape == (0, 0)


class TestBeamSpline:
    def test_interpolation_at_nodes(self):
        """Beam spline should be exact at structural node locations."""
        struct = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
        ])
        G = build_beam_spline(struct, struct.copy(), axis=1)
        assert np.allclose(G, np.eye(3), atol=1e-10)

    def test_midpoint_interpolation(self):
        """Midpoint should get equal weights from neighbors."""
        struct = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        aero = np.array([[0.0, 1.0, 0.0]])
        G = build_beam_spline(struct, aero, axis=1)
        assert G[0, 0] == pytest.approx(0.5)
        assert G[0, 1] == pytest.approx(0.5)

    def test_output_shape(self):
        struct = np.array([[0, 0, 0], [0, 3, 0], [0, 6, 0]], dtype=float)
        aero = np.array([[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 5, 0]], dtype=float)
        G = build_beam_spline(struct, aero, axis=1)
        assert G.shape == (4, 3)

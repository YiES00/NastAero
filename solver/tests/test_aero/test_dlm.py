"""Tests for DLM kernel and AIC matrix."""
import numpy as np
import pytest
from nastaero.aero.panel import generate_panel_mesh, AeroBox
from nastaero.aero.dlm import build_aic_matrix, _biot_savart_segment, _horseshoe_normalwash


class MockCAERO1:
    def __init__(self, p1, p4, chord1, chord4, nspan, nchord):
        self.p1 = np.array(p1)
        self.p4 = np.array(p4)
        self.chord1 = chord1
        self.chord4 = chord4
        self.nspan = nspan
        self.nchord = nchord
        self.eid = 1001


class TestBiotSavart:
    def test_midpoint_above_segment(self):
        """Velocity at point directly above center of a y-aligned segment."""
        p1 = np.array([0.0, -1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        xc = np.array([0.0, 0.0, 1.0])
        v = _biot_savart_segment(xc, p1, p2)
        # r1×r2 -> velocity in +x direction for this geometry
        assert v[0] > 0
        assert abs(v[1]) < 1e-10
        assert abs(v[2]) < 1e-10

    def test_far_field_decay(self):
        """Induced velocity should decay with distance."""
        p1 = np.array([0.0, -1.0, 0.0])
        p2 = np.array([0.0, 1.0, 0.0])
        v1 = _biot_savart_segment(np.array([0.0, 0.0, 1.0]), p1, p2)
        v2 = _biot_savart_segment(np.array([0.0, 0.0, 10.0]), p1, p2)
        assert np.linalg.norm(v1) > np.linalg.norm(v2)


class TestAICMatrix:
    def test_aic_shape(self):
        """AIC matrix should be n x n."""
        caero = MockCAERO1([0, 0, 0], [0, 4, 0], 2.0, 2.0, 2, 2)
        boxes = generate_panel_mesh(caero)
        D = build_aic_matrix(boxes, mach=0.0)
        assert D.shape == (4, 4)

    def test_aic_nonzero_diagonal(self):
        """Diagonal elements should be non-zero (self-influence)."""
        caero = MockCAERO1([0, 0, 0], [0, 4, 0], 2.0, 2.0, 2, 2)
        boxes = generate_panel_mesh(caero)
        D = build_aic_matrix(boxes, mach=0.0)
        for i in range(4):
            assert abs(D[i, i]) > 1e-15

    def test_aic_invertible(self):
        """AIC matrix should be non-singular."""
        caero = MockCAERO1([0, 0, 0], [0, 6, 0], 2.0, 2.0, 4, 2)
        boxes = generate_panel_mesh(caero)
        D = build_aic_matrix(boxes, mach=0.0)
        assert abs(np.linalg.det(D)) > 1e-30

    def test_flat_plate_lift(self):
        """Flat plate at angle of attack should produce positive lift.

        For a unit angle of attack, the normalwash w/V = -1.
        The AIC: {delta_cp} = D^{-1} * {w/V}
        All delta_cp should be positive (net upward lift).
        """
        caero = MockCAERO1([0, 0, 0], [0, 10, 0], 2.0, 2.0, 5, 2)
        boxes = generate_panel_mesh(caero)
        D = build_aic_matrix(boxes, mach=0.0)
        # Unit alpha: w/V = -1 for all boxes
        w = -np.ones(len(boxes))
        delta_cp = np.linalg.solve(D, w)
        # Total lift coefficient (should be positive for positive alpha)
        total_force_z = 0.0
        for i, box in enumerate(boxes):
            total_force_z += delta_cp[i] * box.area * box.normal[2]
        assert total_force_z > 0

    def test_compressibility_effect(self):
        """AIC at M>0 should differ from M=0 (Prandtl-Glauert)."""
        caero = MockCAERO1([0, 0, 0], [0, 4, 0], 2.0, 2.0, 2, 2)
        boxes = generate_panel_mesh(caero)
        D0 = build_aic_matrix(boxes, mach=0.0)
        D5 = build_aic_matrix(boxes, mach=0.5)
        assert not np.allclose(D0, D5)

    def test_empty_boxes(self):
        """Empty box list should return empty matrix."""
        D = build_aic_matrix([], mach=0.0)
        assert D.shape == (0, 0)

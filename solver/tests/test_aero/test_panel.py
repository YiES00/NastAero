"""Tests for aerodynamic panel mesh generation."""
import numpy as np
import pytest
from nastaero.aero.panel import generate_panel_mesh, AeroBox


class MockCAERO1:
    """Mock CAERO1 card for testing."""
    def __init__(self, p1, p4, chord1, chord4, nspan, nchord):
        self.p1 = np.array(p1)
        self.p4 = np.array(p4)
        self.chord1 = chord1
        self.chord4 = chord4
        self.nspan = nspan
        self.nchord = nchord
        self.eid = 1001


class TestPanelMesh:
    def test_box_count(self):
        """nspan x nchord boxes should be generated."""
        caero = MockCAERO1([0, 0, 0], [0, 6, 0], 2.0, 2.0, 4, 2)
        boxes = generate_panel_mesh(caero)
        assert len(boxes) == 8  # 4 x 2

    def test_rectangular_panel(self):
        """Rectangular wing: all boxes should be equal size."""
        caero = MockCAERO1([0, 0, 0], [0, 4, 0], 2.0, 2.0, 4, 2)
        boxes = generate_panel_mesh(caero)
        areas = [b.area for b in boxes]
        assert all(abs(a - areas[0]) < 1e-10 for a in areas)

    def test_control_point_location(self):
        """Control point should be at 3/4 chord, midspan of each box."""
        caero = MockCAERO1([0, 0, 0], [0, 2, 0], 4.0, 4.0, 1, 1)
        boxes = generate_panel_mesh(caero)
        assert len(boxes) == 1
        # 3/4 chord = 3.0, midspan = 1.0
        assert boxes[0].control_point[0] == pytest.approx(3.0)
        assert boxes[0].control_point[1] == pytest.approx(1.0)

    def test_doublet_point_location(self):
        """Doublet point should be at 1/4 chord, midspan of each box."""
        caero = MockCAERO1([0, 0, 0], [0, 2, 0], 4.0, 4.0, 1, 1)
        boxes = generate_panel_mesh(caero)
        # 1/4 chord = 1.0, midspan = 1.0
        assert boxes[0].doublet_point[0] == pytest.approx(1.0)
        assert boxes[0].doublet_point[1] == pytest.approx(1.0)

    def test_normal_direction(self):
        """Normal should point in +z for a flat wing in XY plane."""
        caero = MockCAERO1([0, 0, 0], [0, 6, 0], 2.0, 2.0, 4, 2)
        boxes = generate_panel_mesh(caero)
        for box in boxes:
            assert abs(box.normal[2]) > 0.99  # Nearly +z

    def test_tapered_wing(self):
        """Tapered wing: inboard boxes should be larger than outboard."""
        caero = MockCAERO1([0, 0, 0], [0, 6, 0], 4.0, 2.0, 6, 1)
        boxes = generate_panel_mesh(caero)
        # First box (inboard) should have larger area than last (outboard)
        assert boxes[0].area > boxes[-1].area

    def test_goland_wing(self):
        """Test Goland wing panel: 8 span x 2 chord = 16 boxes."""
        caero = MockCAERO1([0, 0, 0], [0, 6.096, 0], 1.8288, 1.8288, 8, 2)
        boxes = generate_panel_mesh(caero)
        assert len(boxes) == 16
        # Total area should be span * chord
        total_area = sum(b.area for b in boxes)
        expected_area = 6.096 * 1.8288
        assert total_area == pytest.approx(expected_area, rel=1e-6)

    def test_box_ids_sequential(self):
        """Box IDs should be sequential starting from 0."""
        caero = MockCAERO1([0, 0, 0], [0, 4, 0], 2.0, 2.0, 2, 2)
        boxes = generate_panel_mesh(caero)
        ids = [b.box_id for b in boxes]
        assert ids == [0, 1, 2, 3]

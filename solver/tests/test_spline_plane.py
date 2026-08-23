# 스플라인 면내 투영 좌표계 시험 (2026-08 감사)
"""IPS 스플라인이 패널 국부 평면에서 동작하는지 검증.

감사에서 확인된 결함: IPS가 전역 x-y 투영에 하드코딩돼 있어
수직 핀(x-z 평면)에서는 투영이 완전히 퇴화하고, 경사 V-미익에서는
면외(z) 위치 정보를 잃어 힘 작용점의 높이가 어긋났다(ILC-8 기준
1018 mm 면에서 z-팔 +35 mm 편의).
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
from nastaero.aero.panel import generate_panel_mesh
from nastaero.solvers.sol144 import (
    _nodes_are_collinear, _project_plane, _spline_plane_frame)


def _panel(p1, p4, c1, c4, nspan=4, nchord=3):
    return generate_panel_mesh(SimpleNamespace(
        nspan=nspan, nchord=nchord,
        p1=np.array(p1, dtype=float), p4=np.array(p4, dtype=float),
        chord1=c1, chord4=c4))


class TestPlaneFrame:
    def test_horizontal_surface_matches_global_xy(self):
        """수평면에서는 종전 전역 x-y 투영과 같아야 한다(무회귀)."""
        wing = _panel([0, 0, 0], [0, 1000, 0], 500.0, 300.0)
        e1, e2 = _spline_plane_frame(wing, list(range(len(wing))))
        np.testing.assert_allclose(e1, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(np.abs(e2), [0, 1, 0], atol=1e-12)

    def test_vertical_fin_spans_chord_and_height(self):
        """수직 핀에서는 면내 축이 코드와 높이를 잡아야 한다."""
        fin = _panel([0, 0, 0], [0, 0, 1000], 500.0, 300.0)
        e1, e2 = _spline_plane_frame(fin, list(range(len(fin))))
        np.testing.assert_allclose(e1, [1, 0, 0], atol=1e-12)
        assert abs(abs(e2[2]) - 1.0) < 1e-12, f"e2가 z축이어야 한다: {e2}"

    def test_frame_is_orthonormal(self):
        for p4 in ([0, 1000, 0], [0, 0, 1000], [0, 700, 700]):
            boxes = _panel([0, 0, 0], p4, 500.0, 300.0)
            e1, e2 = _spline_plane_frame(boxes, list(range(len(boxes))))
            assert abs(np.dot(e1, e2)) < 1e-12
            assert abs(np.linalg.norm(e1) - 1) < 1e-12
            assert abs(np.linalg.norm(e2) - 1) < 1e-12


class TestFinDegeneracy:
    def _fin_nodes(self):
        return np.array([[x, 0.0, z]
                         for x in (50.0, 250.0, 450.0)
                         for z in (0.0, 500.0, 1000.0)])

    def test_global_xy_projection_is_degenerate(self):
        """핀 중면 절점은 전역 x-y 투영에서 퇴화한다(문제의 근거)."""
        assert _nodes_are_collinear(self._fin_nodes())

    def test_in_plane_projection_is_not_degenerate(self):
        """면내 투영에서는 퇴화하지 않아 IPS를 쓸 수 있어야 한다."""
        fin = _panel([0, 0, 0], [0, 0, 1000], 500.0, 300.0)
        e1, e2 = _spline_plane_frame(fin, list(range(len(fin))))
        nodes = self._fin_nodes()
        proj = _project_plane(nodes, e1, e2)
        assert not _nodes_are_collinear(nodes, plane_xy=proj)

    def test_projection_preserves_in_plane_distances(self):
        """면내 투영은 등거리 사상이어야 한다(면내 점 사이 거리 보존)."""
        fin = _panel([0, 0, 0], [0, 0, 1000], 500.0, 300.0)
        e1, e2 = _spline_plane_frame(fin, list(range(len(fin))))
        nodes = self._fin_nodes()
        proj = _project_plane(nodes, e1, e2)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                d3 = np.linalg.norm(nodes[i] - nodes[j])
                d2 = np.linalg.norm(proj[i] - proj[j])
                assert d2 == pytest.approx(d3, rel=1e-12)

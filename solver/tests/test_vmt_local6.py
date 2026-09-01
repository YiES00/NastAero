# 구성품 국부좌표계 6분력 단면하중 복원(vmt.py r3 MC1)의 해석해·불변성 시험
from __future__ import annotations
import numpy as np
import pytest
from types import SimpleNamespace

from ascent_load.loads_analysis.component_id import ComponentDef
from ascent_load.loads_analysis.vmt import (
    compute_vmt, component_local_frame,
)


def _model(coords):
    nodes = {i + 1: SimpleNamespace(xyz_global=np.asarray(c, dtype=float))
             for i, c in enumerate(coords)}
    return SimpleNamespace(nodes=nodes)


def _wing_comp(nids, sign=1.0):
    return ComponentDef(name='Test Wing', node_ids=list(nids), span_axis=1,
                        shear_axis=2, bending_axis=0, torsion_axis=1,
                        integration_sign=sign)


def _grid_wing(n_span=21, n_chord=3, span=1000.0, chord=200.0):
    coords = []
    for y in np.linspace(0.0, span, n_span):
        for x in np.linspace(0.0, chord, n_chord):
            coords.append([x, y, 0.0])
    return coords


class TestFrame:
    def test_flat_right_wing_frame(self):
        coords = _grid_wing()
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        xyz = np.array([m.nodes[n].xyz_global for n in comp.node_ids])
        frame, chord_ax, third_ax = component_local_frame(xyz, comp)
        e1, e2, e3 = frame
        assert np.allclose(e1, [0, 1, 0], atol=1e-9)
        assert np.allclose(e3, [0, 0, 1], atol=1e-9)
        # 우익에서 e2 = e3 x e1 = -x (전방) — 우수좌표계 유지
        assert np.allclose(e2, [-1, 0, 0], atol=1e-9)
        assert np.allclose(np.cross(e1, e2), e3, atol=1e-12)
        assert chord_ax == 0 and third_ax == 2

    def test_left_wing_outboard_orientation(self):
        coords = [[x, -y, 0.0] for x, y, _ in _grid_wing()]
        m = _model(coords)
        comp = _wing_comp(m.nodes, sign=-1.0)
        xyz = np.array([m.nodes[n].xyz_global for n in comp.node_ids])
        frame, _, _ = component_local_frame(xyz, comp)
        assert frame[0][1] < 0  # e1은 바깥쪽(-Y)


class TestAnalytic:
    def test_flat_wing_fz_matches_legacy(self):
        # 평판 우익 + 절점별 무작위 Fz — 전역 경로와 국부 경로가
        # 성분 대응(V=Vz, T=Mx, M=-My)으로 정확히 일치해야 한다.
        coords = _grid_wing()
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        rng = np.random.default_rng(42)
        forces = {nid: np.array([0, 0, rng.normal(100, 30), 0, 0, 0.0])
                  for nid in m.nodes}
        c = compute_vmt(m, forces, comp, n_stations=10)
        assert c.local_stations is not None
        assert np.allclose(c.local_stations, c.stations, atol=1e-9)
        assert np.allclose(c.local_Vz, c.shear, rtol=1e-9, atol=1e-9)
        assert np.allclose(c.local_Mx, c.torsion, rtol=1e-9, atol=1e-6)
        assert np.allclose(c.local_My, -c.bending_moment, rtol=1e-9, atol=1e-6)
        assert np.allclose(c.local_N, 0.0, atol=1e-9)
        assert np.allclose(c.local_Vy, 0.0, atol=1e-9)

    def test_tip_point_load_cantilever(self):
        # EA 위 일렬 절점 보, 끝단 집중 Fz=P: Vz=P, My=-P(L-s), 비틀림=0
        L, P = 1000.0, 500.0
        ys = np.linspace(0.0, L, 11)
        coords = [[0.0, y, 0.0] for y in ys]
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        forces = {nid: np.zeros(6) for nid in m.nodes}
        forces[len(coords)][2] = P  # 마지막 절점 = 끝단
        c = compute_vmt(m, forces, comp, n_stations=10)
        for i, s in enumerate(c.local_stations):
            assert c.local_Vz[i] == pytest.approx(P)
            assert c.local_My[i] == pytest.approx(-P * (L - s), abs=1e-6)
            assert abs(c.local_Mx[i]) < 1e-9
            assert abs(c.local_N[i]) < 1e-12
            assert abs(c.local_Vy[i]) < 1e-12

    def test_tip_couple_pure_torsion(self):
        # 끝단 시위 양끝의 상반 Fz 짝힘 — 모든 스테이션에서
        # Vz=0, Mx=c*P (절단점 위치와 무관)
        chord, P = 200.0, 300.0
        coords = _grid_wing(n_span=11, n_chord=2, chord=chord)
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        forces = {nid: np.zeros(6) for nid in m.nodes}
        tip_le = 21  # (0, 1000, 0)  [n_span=11 격자의 마지막 행]
        tip_te = 22  # (200, 1000, 0)
        assert np.allclose(m.nodes[tip_le].xyz_global, [0, 1000, 0])
        assert np.allclose(m.nodes[tip_te].xyz_global, [chord, 1000, 0])
        forces[tip_le][2] = +P
        forces[tip_te][2] = -P
        c = compute_vmt(m, forces, comp, n_stations=10)
        assert np.allclose(c.local_Vz, 0.0, atol=1e-9)
        assert np.allclose(c.local_Mx, chord * P, rtol=1e-9)
        assert np.allclose(c.torsion, chord * P, rtol=1e-9)  # 전역 경로도 동일

    def test_pure_nodal_moment(self):
        # 끝단 절점의 직접 모멘트 m_y — 비틀림으로 전량 전달
        m_y = 4.0e5
        coords = [[0.0, y, 0.0] for y in np.linspace(0, 1000, 11)]
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        forces = {nid: np.zeros(6) for nid in m.nodes}
        forces[len(coords)][4] = m_y
        c = compute_vmt(m, forces, comp, n_stations=10)
        assert np.allclose(c.local_Mx, m_y, rtol=1e-12)
        assert np.allclose(c.local_Vz, 0.0, atol=1e-12)
        assert np.allclose(c.local_My, 0.0, atol=1e-6)

    def test_swept_wing_no_spurious_torsion(self):
        # 30도 후퇴한 부재축 위의 끝단 하중 — 국부 비틀림은 0이어야
        # 하고, 전역축 경로는 후퇴 유발 허위 비틀림을 만든다.
        L, P, sweep = 1000.0, 500.0, np.deg2rad(30.0)
        ys = np.linspace(0.0, L, 21)
        coords = [[y * np.tan(sweep), y, 0.0] for y in ys]
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        forces = {nid: np.zeros(6) for nid in m.nodes}
        forces[len(coords)][2] = P
        c = compute_vmt(m, forces, comp, n_stations=10)
        assert np.max(np.abs(c.local_Mx)) < 1e-6 * P * L
        # 전역 경로의 허위 비틀림은 스팬 오프셋 x 하중 규모
        assert np.max(np.abs(c.torsion)) > 0.2 * P * L * np.tan(sweep)

    def test_vtail_40deg_normal_load(self):
        # 40도 경사면(V-테일 반쪽)의 면법선 등분포 하중 — 국부 Vz는
        # 전량을, 전역 V(Fz)는 cos(40deg)만 포착한다.
        gam = np.deg2rad(40.0)
        e_span = np.array([0.0, np.cos(gam), np.sin(gam)])
        n_hat = np.array([0.0, -np.sin(gam), np.cos(gam)])
        coords, f = [], 50.0
        n_span, n_chord = 21, 3
        for u in np.linspace(0.0, 1000.0, n_span):
            for x in np.linspace(0.0, 200.0, n_chord):
                coords.append(np.array([x, 0, 0]) + u * e_span)
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        forces = {nid: np.concatenate([f * n_hat, np.zeros(3)])
                  for nid in m.nodes}
        c = compute_vmt(m, forces, comp, n_stations=10)
        e1, e2, e3 = c.local_frame
        assert np.allclose(e1, e_span, atol=1e-9)
        assert np.allclose(e3, n_hat, atol=1e-9)
        total = f * n_span * n_chord
        assert c.local_Vz[0] == pytest.approx(total, rel=1e-9)
        assert abs(c.local_N[0]) < 1e-9
        assert abs(c.local_Vy[0]) < 1e-9
        # 전역 경로의 뿌리 전단은 cos(gamma)만 남는다
        assert c.shear[0] == pytest.approx(total * np.cos(gam), rel=1e-6)


class TestInvariance:
    def _random_case(self):
        coords = _grid_wing()
        m = _model(coords)
        comp = _wing_comp(m.nodes)
        rng = np.random.default_rng(7)
        forces = {nid: rng.normal(0, 100, 6) for nid in m.nodes}
        return m, comp, forces

    @staticmethod
    def _transform(m, forces, R, t):
        m2 = _model([R @ m.nodes[n].xyz_global + t for n in sorted(m.nodes)])
        f2 = {n: np.concatenate([R @ forces[n][:3], R @ forces[n][3:]])
              for n in sorted(m.nodes)}
        return m2, f2

    def test_translation_invariance(self):
        m, comp, forces = self._random_case()
        c0 = compute_vmt(m, forces, comp, n_stations=10)
        m2, f2 = self._transform(m, forces, np.eye(3),
                                 np.array([3000.0, -500.0, 1200.0]))
        c1 = compute_vmt(m2, f2, comp, n_stations=10)
        for k in ('local_N', 'local_Vy', 'local_Vz',
                  'local_Mx', 'local_My', 'local_Mz'):
            assert np.allclose(getattr(c1, k), getattr(c0, k),
                               rtol=1e-9, atol=1e-6), k

    def test_yaw_rotation_full_invariance(self):
        # 상향 힌트축(Z)에 대한 회전: 자동 프레임이 구조와 함께 돌아
        # 6성분 전부 불변
        m, comp, forces = self._random_case()
        c0 = compute_vmt(m, forces, comp, n_stations=10)
        psi = np.deg2rad(25.0)
        R = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0], [0, 0, 1.0]])
        m2, f2 = self._transform(m, forces, R, np.zeros(3))
        c1 = compute_vmt(m2, f2, comp, n_stations=10)
        for k in ('local_N', 'local_Vy', 'local_Vz',
                  'local_Mx', 'local_My', 'local_Mz'):
            assert np.allclose(getattr(c1, k), getattr(c0, k),
                               rtol=1e-8, atol=1e-5), k

    def test_general_rotation_physical_invariants(self):
        # 일반 회전: 축력·비틀림과 횡전단/횡모멘트의 크기(노름)는
        # 프레임 datum(전역 상향 힌트)과 무관한 물리량으로 불변
        m, comp, forces = self._random_case()
        c0 = compute_vmt(m, forces, comp, n_stations=10)
        th = np.deg2rad(20.0)
        R = np.array([[1, 0, 0],
                      [0, np.cos(th), -np.sin(th)],
                      [0, np.sin(th), np.cos(th)]])
        m2, f2 = self._transform(m, forces, R, np.zeros(3))
        c1 = compute_vmt(m2, f2, comp, n_stations=10)
        assert np.allclose(c1.local_N, c0.local_N, rtol=1e-8, atol=1e-5)
        assert np.allclose(c1.local_Mx, c0.local_Mx, rtol=1e-8, atol=1e-4)
        v0 = np.hypot(c0.local_Vy, c0.local_Vz)
        v1 = np.hypot(c1.local_Vy, c1.local_Vz)
        assert np.allclose(v1, v0, rtol=1e-8, atol=1e-5)
        m0 = np.hypot(c0.local_My, c0.local_Mz)
        m1 = np.hypot(c1.local_My, c1.local_Mz)
        assert np.allclose(m1, m0, rtol=1e-8, atol=1e-4)

    def test_root_station_equals_total(self):
        # 뿌리 절단의 6분력 = 전 하중 합력·합모멘트의 프레임 투영
        m, comp, forces = self._random_case()
        c = compute_vmt(m, forces, comp, n_stations=10)
        e1, e2, e3 = c.local_frame
        F = np.sum([forces[n][:3] for n in m.nodes], axis=0)
        cut0 = c.local_cut_points[0]
        M = np.sum([np.cross(m.nodes[n].xyz_global - cut0, forces[n][:3])
                    + forces[n][3:] for n in m.nodes], axis=0)
        assert c.local_N[0] == pytest.approx(F @ e1, abs=1e-6)
        assert c.local_Vy[0] == pytest.approx(F @ e2, abs=1e-6)
        assert c.local_Vz[0] == pytest.approx(F @ e3, abs=1e-6)
        assert c.local_Mx[0] == pytest.approx(M @ e1, abs=1e-3)
        assert c.local_My[0] == pytest.approx(M @ e2, abs=1e-3)
        assert c.local_Mz[0] == pytest.approx(M @ e3, abs=1e-3)

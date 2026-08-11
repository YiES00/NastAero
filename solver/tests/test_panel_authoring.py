# Panel 탭 코어 로직 시험 — CAERO 카드 왕복, W2GJ 라이터 왕복(플립 포함), 스플라인 제안
from __future__ import annotations

import numpy as np
import pytest

from nastaero.aero.panel_authoring import (
    caero1_card_text, mirror_caero1, set1_card_text, spline1_card_text,
    suggest_spline_nodes, w2gj_dmi_text,
)


def _parse_text(tmp_path, bulk: str):
    from nastaero.bdf.parser import parse_bdf

    p = tmp_path / "deck.bdf"
    p.write_text("SOL 144\nCEND\nBEGIN BULK\n" + bulk + "ENDDATA\n")
    return parse_bdf(str(p))


class TestCaeroCardText:
    def test_round_trip(self, tmp_path):
        text = caero1_card_text(1001, 1, nspan=4, nchord=2,
                                p1=[0.0, 0.5, 0.0], chord1=1.5,
                                p4=[0.2, 5.0, 0.1], chord4=1.0)
        m = _parse_text(tmp_path, text + "\n")
        c = m.caero_panels[1001]
        assert c.nspan == 4 and c.nchord == 2
        assert c.p1[1] == pytest.approx(0.5)
        assert c.p4[0] == pytest.approx(0.2)
        assert c.chord1 == pytest.approx(1.5)
        assert c.chord4 == pytest.approx(1.0)

    def test_mirror(self):
        m1, c1, m4, c4 = mirror_caero1(
            1001, [0.0, 0.5, 0.0], 1.5, [0.2, 5.0, 0.1], 1.0)
        assert m1[1] == pytest.approx(-0.5)
        assert m4[1] == pytest.approx(-5.0)
        assert (c1, c4) == (1.5, 1.0)


class TestW2gjWriter:
    def _model(self, tmp_path):
        # 우익(1001) + 좌익 미러(2001) — 미러는 기하 법선 -z (플립 대상)
        right = caero1_card_text(1001, 1, nspan=2, nchord=4,
                                 p1=[0.0, 0.0, 0.0], chord1=1.0,
                                 p4=[0.0, 4.0, 0.0], chord4=1.0)
        m1, c1, m4, c4 = mirror_caero1(2001, [0.0, 0.0, 0.0], 1.0,
                                       [0.0, 4.0, 0.0], 1.0)
        left = caero1_card_text(2001, 1, nspan=2, nchord=4,
                                p1=m1, chord1=c1, p4=m4, chord4=c4)
        return _parse_text(tmp_path, right + "\n" + left + "\n")

    def test_round_trip_wash_equals_camber_slope(self, tmp_path):
        from nastaero.aero.airfoil_camber import (
            AirfoilCamber, PanelAirfoilConfig,
        )
        from nastaero.aero.airfoil_camber import compute_camber_normalwash
        from nastaero.aero.panel import generate_all_panels

        model = self._model(tmp_path)
        boxes = generate_all_panels(model, use_nastran_eid=True)
        af = AirfoilCamber.from_naca_string("NACA2412")
        cfg = PanelAirfoilConfig(
            panel_airfoils={1001: (af, None), 2001: (af, None)})

        # 좌익 미러 패널은 실제로 플립되었는지 전제 확인
        flips = {b.box_id: b.normal_flipped for b in boxes}
        assert not any(flips[b] for b in range(1001, 1009))
        assert all(flips[b] for b in range(2001, 2009))

        text = w2gj_dmi_text(boxes, model.caero_panels, cfg)
        m2 = self._model(tmp_path)  # 새 모델에 W2GJ만 추가해 재파싱
        deck = (caero1_card_text(1001, 1, 2, 4, [0, 0, 0], 1.0,
                                 [0, 4, 0], 1.0) + "\n"
                + caero1_card_text(2001, 1, 2, 4, [0, 0, 0], 1.0,
                                   [0, -4, 0], 1.0) + "\n" + text)
        m2 = _parse_text(tmp_path, deck)
        w2 = m2.dmis["W2GJ"].matrix[:, 0]

        # 읽기 경로 재현: wash = -flip * w2gj == dz/dx
        flip = np.array([-1.0 if b.normal_flipped else 1.0 for b in boxes])
        wash = -flip * w2
        dzdx = compute_camber_normalwash(boxes, model.caero_panels, cfg)
        assert np.allclose(wash, dzdx, atol=2e-5)
        # 좌우 물리 동일: 정렬된 wash 분포가 같아야 함
        assert np.allclose(np.sort(wash[:8]), np.sort(wash[8:]), atol=2e-5)
        # 파일 값 자체는 좌우 부호 거울상 (GACOMP과 동일 규약)
        assert np.allclose(np.sort(w2[:8]), np.sort(-w2[8:]), atol=2e-5)


class TestSplineSuggest:
    def _model_with_nodes(self, tmp_path):
        cards = [caero1_card_text(1001, 1, nspan=2, nchord=2,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 4.0, 0.0], chord4=2.0)]
        nid = 1
        for y in np.linspace(0.2, 3.8, 5):
            for x in np.linspace(0.1, 1.9, 3):
                cards.append(f"GRID    {nid:<8}        {x:<8.3f}{y:<8.3f}0.05")
                nid += 1
        cards.append("GRID    900             10.     20.     0.")   # 멀리
        cards.append("GRID    901             1.      2.      5.")   # 면외
        return _parse_text(tmp_path, "\n".join(cards) + "\n")

    def test_nearby_nodes_only(self, tmp_path):
        m = self._model_with_nodes(tmp_path)
        ids = suggest_spline_nodes(m, m.caero_panels[1001])
        assert set(ids) == set(range(1, 16))   # 격자 15점만
        assert 900 not in ids and 901 not in ids

    def test_max_nodes_thinning(self, tmp_path):
        m = self._model_with_nodes(tmp_path)
        ids = suggest_spline_nodes(m, m.caero_panels[1001], max_nodes=6)
        assert len(ids) <= 6 and set(ids) <= set(range(1, 16))

    def test_set1_spline1_round_trip(self, tmp_path):
        ids = list(range(1, 12))   # 연속줄 필요한 개수
        text = (set1_card_text(77, ids) + "\n"
                + spline1_card_text(501, 1001, 1001, 1004, 77) + "\n")
        m = _parse_text(tmp_path, text)
        assert m.sets[77].ids == ids
        s = m.splines[501]
        assert (s.caero, s.box1, s.box2, s.setg) == (1001, 1001, 1004, 77)


class TestHardPointPreference:
    """스파·리브(하드포인트) 우선 선정 + 시위 2열 보장."""

    def _model(self, tmp_path):
        # 5x3 외피 격자 + 앞스파(x=0.1)·뒷스파(x=1.9)를 잇는 CBAR
        cards = [caero1_card_text(1001, 1, nspan=2, nchord=2,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 4.0, 0.0], chord4=2.0)]
        nid = 1
        grid = {}
        for iy, y in enumerate(np.linspace(0.2, 3.8, 5)):
            for ix, x in enumerate(np.linspace(0.1, 1.9, 3)):
                cards.append(
                    f"GRID    {nid:<8}        {x:<8.3f}{y:<8.3f}0.0")
                grid[(ix, iy)] = nid
                nid += 1
        eid = 100
        for iy in range(4):   # 스파 캡: 앞(ix=0)·뒤(ix=2) 기둥의 CBAR
            for ix in (0, 2):
                cards.append(f"CBAR    {eid:<8}1       "
                             f"{grid[(ix, iy)]:<8}{grid[(ix, iy + 1)]:<8}")
                eid += 1
        return _parse_text(tmp_path, "\n".join(cards) + "\n")

    def test_hard_points_detected(self, tmp_path):
        from nastaero.aero.panel_authoring import hard_point_ids

        m = self._model(tmp_path)
        hard = hard_point_ids(m)
        # 앞/뒤 스파 절점(ix=0,2)만 하드 — 중앙열(ix=1)은 제외
        spar = {1 + 3 * iy + ix for iy in range(5) for ix in (0, 2)}
        mid = {1 + 3 * iy + 1 for iy in range(5)}
        assert spar <= hard
        assert not (mid & hard)

    def test_prefer_hard_points(self, tmp_path):
        m = self._model(tmp_path)
        ids = suggest_spline_nodes(m, m.caero_panels[1001],
                                   prefer_hard_points=True)
        spar = {1 + 3 * iy + ix for iy in range(5) for ix in (0, 2)}
        assert set(ids) == spar   # 스파 절점 10개만

    def test_chordwise_two_rows_kept(self, tmp_path):
        m = self._model(tmp_path)
        ids = suggest_spline_nodes(m, m.caero_panels[1001],
                                   prefer_hard_points=True, max_nodes=6)
        xs = {round(float(getattr(m.nodes[n], "xyz_global", m.nodes[n].xyz)[0]), 2)
              for n in ids}
        assert 0.1 in xs and 1.9 in xs   # 앞/뒤 스파 모두 유지
        assert len(ids) <= 6


class TestPidFilterAndCurvature:
    """PID 필터와 연속 곡면(붐) 배제 — 접합선만 하드로 남는지."""

    def _model(self, tmp_path):
        # PID 1: 평판 외피 3x3 격자(x-y면) + PID 2: 중앙(x=1)에 수직 웹
        # PID 3: 별도 '붐' — 코스한 각기둥 튜브(모든 절점이 꺾임)
        cards = [caero1_card_text(1001, 1, nspan=2, nchord=2,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 4.0, 0.0], chord4=2.0)]
        nid = 1
        g = {}
        for iy, y in enumerate([0.5, 2.0, 3.5]):
            for ix, x in enumerate([0.0, 1.0, 2.0]):
                cards.append(f"GRID    {nid:<8}        {x:<8.3f}{y:<8.3f}0.0")
                g[(ix, iy)] = nid
                nid += 1
        # 웹 하단 절점 (z=-0.5, x=1)
        w = {}
        for iy, y in enumerate([0.5, 2.0, 3.5]):
            cards.append(f"GRID    {nid:<8}        1.0     {y:<8.3f}-0.5")
            w[iy] = nid
            nid += 1
        eid = 100
        for iy in range(2):        # 외피 (PID 1)
            for ix in range(2):
                cards.append(
                    f"CQUAD4  {eid:<8}1       {g[(ix, iy)]:<8}"
                    f"{g[(ix + 1, iy)]:<8}{g[(ix + 1, iy + 1)]:<8}"
                    f"{g[(ix, iy + 1)]:<8}")
                eid += 1
        for iy in range(2):        # 수직 웹 (PID 2), 접합선 = x=1 열
            cards.append(
                f"CQUAD4  {eid:<8}2       {g[(1, iy)]:<8}{g[(1, iy + 1)]:<8}"
                f"{w[iy + 1]:<8}{w[iy]:<8}")
            eid += 1
        # 붐: 사각 단면 튜브 (패널 근처, 4면 모두 90도 꺾임 — PID 3)
        b = {}
        ring = [(3.0, 0.2), (3.4, 0.2), (3.4, -0.2), (3.0, -0.2)]
        for iy, y in enumerate([1.0, 2.0]):
            for k, (bx, bz) in enumerate(ring):
                cards.append(
                    f"GRID    {nid:<8}        {bx:<8.3f}{y:<8.3f}{bz:<8.3f}")
                b[(k, iy)] = nid
                nid += 1
        for k in range(4):
            k2 = (k + 1) % 4
            cards.append(
                f"CQUAD4  {eid:<8}3       {b[(k, 0)]:<8}{b[(k2, 0)]:<8}"
                f"{b[(k2, 1)]:<8}{b[(k, 1)]:<8}")
            eid += 1
        return _parse_text(tmp_path, "\n".join(cards) + "\n"), g, w, b

    def test_junction_hard_boom_rejected(self, tmp_path):
        from nastaero.aero.panel_authoring import hard_point_ids

        m, g, w, b = self._model(tmp_path)
        hard = hard_point_ids(m)
        junction = {g[(1, iy)] for iy in range(3)}   # 외피-웹 접합선
        boom = {n for n in b.values()}
        assert junction <= hard          # 접합선은 하드
        assert not (boom & hard)         # 연속 꺾임 튜브는 배제
        flat = {g[(0, 0)], g[(2, 2)]}    # 평탄 외피 코너는 하드 아님
        assert not (flat & hard)

    def test_pid_filter(self, tmp_path):
        m, g, w, b = self._model(tmp_path)
        web_nodes = set(w.values()) | {g[(1, iy)] for iy in range(3)}
        ids = suggest_spline_nodes(m, m.caero_panels[1001],
                                   offset_tol=1.0, pids={2})
        assert set(ids) <= web_nodes
        assert {g[(1, iy)] for iy in range(3)} <= set(ids)  # 접합선 포함
        # PID 필터는 하드포인트 휴리스틱과 무관하게 우선 적용
        ids2 = suggest_spline_nodes(m, m.caero_panels[1001],
                                    offset_tol=1.0, pids={1})
        assert set(ids2) == set(g.values())

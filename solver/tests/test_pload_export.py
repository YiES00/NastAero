# PLOAD4 내보내기 시험 — 박스별 보존, 법선 부호, 웹 제외, 상하 외피 분배, 카드 형식
from __future__ import annotations

import numpy as np
import pytest

from ascent_load.loads_analysis.pload_export import (
    map_box_forces_to_skin, write_pload4_cards,
)


def _parse(tmp_path, bulk: str):
    from ascent_load.bdf.parser import parse_bdf

    p = tmp_path / "m.bdf"
    p.write_text("SOL 101\nCEND\nBEGIN BULK\n" + bulk + "ENDDATA\n")
    return parse_bdf(str(p))


def _grid(nid, x, y, z):
    return f"GRID    {nid:<8}        {x:<8.3f}{y:<8.3f}{z:<8.3f}"


def _quad(eid, pid, n1, n2, n3, n4):
    return (f"CQUAD4  {eid:<8}{pid:<8}{n1:<8}{n2:<8}{n3:<8}{n4:<8}")


class TestPloadMapping:
    def _boxes(self, tmp_path, extra_cards=""):
        """CAERO 1x2 (2박스, 시위 2.0 스팬 0..4) + 외피 2장 (z=0)."""
        from ascent_load.aero.panel import generate_all_panels
        from ascent_load.aero.panel_authoring import caero1_card_text

        cards = [caero1_card_text(1001, 1, nspan=2, nchord=1,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 4.0, 0.0], chord4=2.0)]
        # 외피: 박스마다 정확히 한 장 (0..2 x 0..2), (0..2 x 2..4)
        cards += [_grid(1, 0, 0, 0), _grid(2, 2, 0, 0),
                  _grid(3, 2, 2, 0), _grid(4, 0, 2, 0),
                  _grid(5, 2, 4, 0), _grid(6, 0, 4, 0)]
        cards += [_quad(10, 1, 1, 2, 3, 4),      # 법선 +z
                  _quad(11, 1, 4, 3, 5, 6)]      # 법선 +z
        if extra_cards:
            cards.append(extra_cards)
        m = _parse(tmp_path, "\n".join(cards) + "\n")
        boxes = generate_all_panels(m, use_nastran_eid=True)
        return m, boxes

    def test_per_box_conservation(self, tmp_path):
        m, boxes = self._boxes(tmp_path)
        F = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 40.0]])
        pressures, rep = map_box_forces_to_skin(m, boxes, F, offset_tol=0.5)
        assert rep["n_covered"] == 2 and not rep["uncovered"]
        # 요소 면적 4.0 → p = F/A
        assert pressures[10] == pytest.approx(25.0)
        assert pressures[11] == pytest.approx(10.0)
        assert rep["force_mapped"][2] == pytest.approx(140.0)
        assert rep["residual_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_flipped_element_gets_negative_pressure(self, tmp_path):
        # 요소 절점 순서 반전(법선 -z) → 압력 부호 반전, 합력 방향 유지
        m, boxes = self._boxes(tmp_path)
        # 요소 10 절점 순서를 뒤집은 모델 재구성
        from ascent_load.aero.panel_authoring import caero1_card_text

        cards = [caero1_card_text(1001, 1, nspan=2, nchord=1,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 4.0, 0.0], chord4=2.0),
                 _grid(1, 0, 0, 0), _grid(2, 2, 0, 0),
                 _grid(3, 2, 2, 0), _grid(4, 0, 2, 0),
                 _grid(5, 2, 4, 0), _grid(6, 0, 4, 0),
                 _quad(10, 1, 4, 3, 2, 1),       # 법선 -z (반전)
                 _quad(11, 1, 4, 3, 5, 6)]
        m2 = _parse(tmp_path, "\n".join(cards) + "\n")
        from ascent_load.aero.panel import generate_all_panels

        boxes2 = generate_all_panels(m2, use_nastran_eid=True)
        F = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 40.0]])
        pressures, rep = map_box_forces_to_skin(m2, boxes2, F,
                                                offset_tol=0.5)
        assert pressures[10] == pytest.approx(-25.0)   # 법선 반대 → 음압
        assert rep["force_mapped"][2] == pytest.approx(140.0)  # 합력 동일

    def test_web_excluded_by_alignment(self, tmp_path):
        # 수직 웹(법선 ⊥ 박스 법선)은 매핑에서 제외
        web = "\n".join([
            _grid(7, 1.0, 0.0, -0.5), _grid(8, 1.0, 2.0, -0.5),
            _quad(20, 2, 1, 4, 8, 7).replace("1       4",
                                             "1       4"),
        ])
        # 웹: 절점 1,4는 z=0, 7,8은 z=-0.5 → x=1 평면의 수직판이 아님.
        # 간단히 y-z 평면 수직판을 직접 구성
        cards_web = "\n".join([
            _grid(7, 1.0, 0.0, -0.5), _grid(8, 1.0, 2.0, -0.5),
            _grid(9, 1.0, 2.0, 0.0), _grid(15, 1.0, 0.0, 0.0),
            _quad(20, 2, 15, 9, 8, 7),
        ])
        m, boxes = self._boxes(tmp_path, cards_web)
        F = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 40.0]])
        pressures, _ = map_box_forces_to_skin(m, boxes, F, offset_tol=0.6)
        assert 20 not in pressures

    def test_upper_lower_split_conserves(self, tmp_path):
        # 상/하 외피가 모두 밴드 안이면 압력이 반분되고 합력은 보존
        extra = "\n".join([
            _grid(21, 0, 0, -0.2), _grid(22, 2, 0, -0.2),
            _grid(23, 2, 2, -0.2), _grid(24, 0, 2, -0.2),
            _grid(25, 2, 4, -0.2), _grid(26, 0, 4, -0.2),
            _quad(30, 1, 21, 22, 23, 24),
            _quad(31, 1, 24, 23, 25, 26),
        ])
        m, boxes = self._boxes(tmp_path, extra)
        F = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 40.0]])
        pressures, rep = map_box_forces_to_skin(m, boxes, F, offset_tol=0.5)
        assert pressures[10] == pytest.approx(12.5)    # 100/(4+4)
        assert pressures[30] == pytest.approx(12.5)
        assert rep["force_mapped"][2] == pytest.approx(140.0)

    def test_uncovered_box_reported(self, tmp_path):
        from ascent_load.aero.panel import generate_all_panels
        from ascent_load.aero.panel_authoring import caero1_card_text

        # 외피 없는 두 번째 CAERO (VTP처럼)
        cards = [caero1_card_text(1001, 1, nspan=1, nchord=1,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 2.0, 0.0], chord4=2.0),
                 caero1_card_text(5001, 1, nspan=1, nchord=1,
                                  p1=[10.0, 0.0, 0.0], chord1=1.0,
                                  p4=[10.0, 0.0, 2.0], chord4=1.0),
                 _grid(1, 0, 0, 0), _grid(2, 2, 0, 0),
                 _grid(3, 2, 2, 0), _grid(4, 0, 2, 0),
                 _quad(10, 1, 1, 2, 3, 4)]
        m = _parse(tmp_path, "\n".join(cards) + "\n")
        boxes = generate_all_panels(m, use_nastran_eid=True)
        F = np.array([[0.0, 0.0, 50.0], [0.0, 30.0, 0.0]])
        pressures, rep = map_box_forces_to_skin(m, boxes, F, offset_tol=0.5)
        assert rep["n_covered"] == 1
        assert rep["uncovered"] == [5001]


class TestPloadWriter:
    def test_card_format_and_report_header(self, tmp_path):
        path = str(tmp_path / "p.bdf")
        rep = {"n_covered": 2, "n_boxes": 2, "residual_pct": 0.0,
               "force_in": np.array([0.0, 0.0, 140.0]),
               "force_mapped": np.array([0.0, 0.0, 140.0])}
        write_pload4_cards({10: 25.0, 11: -0.01}, path, load_sid=3,
                           label="SUBCASE 3", report=rep)
        text = open(path).read()
        assert "PLOAD4*" in text
        assert "SUBCASE 3" in text
        assert "boxes covered: 2/2" in text
        line = [l for l in text.splitlines()
                if l.startswith("PLOAD4*")][0]
        # 큰필드: SID(16) EID(16) P1(16)
        assert int(line[8:24]) == 3
        assert int(line[24:40]) == 10
        assert float(line[40:56]) == pytest.approx(25.0)


class TestMappingRefinements:
    """박스별 CAERO 시위 밴드·연속 곡면 배제·PID 필터."""

    def test_per_caero_offset_band(self, tmp_path):
        # 큰 시위 CAERO(밴드 0.3)와 작은 시위 CAERO(밴드 0.06) 아래에
        # 같은 오프셋(0.1)의 외피 → 큰 쪽만 매핑
        from ascent_load.aero.panel import generate_all_panels
        from ascent_load.aero.panel_authoring import caero1_card_text

        cards = [caero1_card_text(1001, 1, nspan=1, nchord=1,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 2.0, 0.0], chord4=2.0),
                 caero1_card_text(5001, 1, nspan=1, nchord=1,
                                  p1=[10.0, 0.0, 0.0], chord1=0.4,
                                  p4=[10.0, 2.0, 0.0], chord4=0.4),
                 _grid(1, 0, 0, 0.1), _grid(2, 2, 0, 0.1),
                 _grid(3, 2, 2, 0.1), _grid(4, 0, 2, 0.1),
                 _grid(5, 10.0, 0, 0.1), _grid(6, 10.4, 0, 0.1),
                 _grid(7, 10.4, 2, 0.1), _grid(8, 10.0, 2, 0.1),
                 _quad(10, 1, 1, 2, 3, 4),
                 _quad(11, 1, 5, 6, 7, 8)]
        m = _parse(tmp_path, "\n".join(cards) + "\n")
        boxes = generate_all_panels(m, use_nastran_eid=True)
        F = np.array([[0, 0, 100.0], [0, 0, 40.0]])
        pressures, rep = map_box_forces_to_skin(m, boxes, F)
        assert 10 in pressures            # 시위 2.0 → 밴드 0.3 > 0.1
        assert 11 not in pressures        # 시위 0.4 → 밴드 0.06 < 0.1
        assert rep["uncovered"] == [5001]

    def test_curved_tube_excluded(self, tmp_path):
        # 밴드 안의 코스한 사각 튜브(연속 꺾임) → 압력 미도장
        from ascent_load.aero.panel import generate_all_panels
        from ascent_load.aero.panel_authoring import caero1_card_text

        cards = [caero1_card_text(1001, 1, nspan=1, nchord=1,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 2.0, 0.0], chord4=2.0),
                 _grid(1, 0, 0, 0), _grid(2, 2, 0, 0),
                 _grid(3, 2, 2, 0), _grid(4, 0, 2, 0),
                 _quad(10, 1, 1, 2, 3, 4)]
        # 사각 튜브 (0.6..1.4)x, z -0.1..0.3 — 상하 페이싯이 밴드 안
        ring = [(0.6, 0.3), (1.4, 0.3), (1.4, -0.1), (0.6, -0.1)]
        nid = 20
        b = {}
        for iy, y in enumerate([0.5, 1.5]):
            for k, (bx, bz) in enumerate(ring):
                cards.append(_grid(nid, bx, y, bz))
                b[(k, iy)] = nid
                nid += 1
        eid = 30
        for k in range(4):
            k2 = (k + 1) % 4
            cards.append(_quad(eid, 3, b[(k, 0)], b[(k2, 0)],
                               b[(k2, 1)], b[(k, 1)]))
            eid += 1
        m = _parse(tmp_path, "\n".join(cards) + "\n")
        boxes = generate_all_panels(m, use_nastran_eid=True)
        F = np.array([[0, 0, 100.0]])
        pressures, _ = map_box_forces_to_skin(m, boxes, F, offset_tol=0.5)
        assert 10 in pressures
        assert not any(e in pressures for e in range(30, 34))
        # 배제 없이 돌리면 튜브 상/하 페이싯이 칠해짐 (대조)
        p2, _ = map_box_forces_to_skin(m, boxes, F, offset_tol=0.5,
                                       exclude_curved=False)
        assert any(e in p2 for e in range(30, 34))

    def test_pid_filter_overrides(self, tmp_path):
        from ascent_load.aero.panel import generate_all_panels
        from ascent_load.aero.panel_authoring import caero1_card_text

        cards = [caero1_card_text(1001, 1, nspan=1, nchord=1,
                                  p1=[0.0, 0.0, 0.0], chord1=2.0,
                                  p4=[0.0, 2.0, 0.0], chord4=2.0),
                 _grid(1, 0, 0, 0), _grid(2, 2, 0, 0),
                 _grid(3, 2, 2, 0), _grid(4, 0, 2, 0),
                 _grid(5, 0, 0, 0.1), _grid(6, 2, 0, 0.1),
                 _grid(7, 2, 2, 0.1), _grid(8, 0, 2, 0.1),
                 _quad(10, 1, 1, 2, 3, 4),
                 _quad(11, 2, 5, 6, 7, 8)]
        m = _parse(tmp_path, "\n".join(cards) + "\n")
        boxes = generate_all_panels(m, use_nastran_eid=True)
        F = np.array([[0, 0, 100.0]])
        pressures, rep = map_box_forces_to_skin(m, boxes, F,
                                                offset_tol=0.5, pids={2})
        assert set(pressures) == {11}
        assert rep["force_mapped"][2] == pytest.approx(100.0)

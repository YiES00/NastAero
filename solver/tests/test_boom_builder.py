# VTOL 붐/파일런 생성기 단위시험 — 허브 GRID 일치, 파싱 왕복, 방향벡터 유효성
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nastaero.bdf.parser import parse_bdf
from nastaero.rotor.boom_builder import _f, _tube_section, build_vtol_boom_bulk
from nastaero.rotor.rotor_config import VTOLConfig

KC = Path(__file__).parent / "validation" / "GACOMP"

pytestmark = pytest.mark.skipif(
    not KC.is_dir(),
    reason="comparison-model data not present in this archive")


@pytest.fixture(scope="module")
def gacomp_model():
    return parse_bdf(str(KC / "p400r3-free-trim.bdf"))


@pytest.fixture(scope="module")
def boom_deck(gacomp_model, tmp_path_factory):
    """붐 벌크를 원본 기체와 합친 임시 덱 (INCLUDE 상대경로 회피)."""
    vtol = VTOLConfig.gacomp_tilt_rotor_12()
    bulk = build_vtol_boom_bulk(gacomp_model, vtol)
    master = (KC / "p400r3-free-trim.bdf").read_text()
    lines = master.split("\n")
    i_end = next(i for i, l in enumerate(lines)
                 if l.strip().upper().startswith("ENDDATA"))
    lines[i_end:i_end] = bulk.split("\n")
    # INCLUDE가 GACOMP 폴더 기준이므로 그 폴더에 임시 저장
    p = KC / "_test_boom_deck.bdf"
    p.write_text("\n".join(lines))
    yield parse_bdf(str(p)), vtol
    p.unlink()


class TestTubeSection:
    def test_od120_t4(self):
        a, i, j = _tube_section(120.0, 4.0)
        assert abs(a - np.pi * (60 ** 2 - 56 ** 2)) < 1e-9
        assert abs(j - 2 * i) < 1e-9

    def test_fit8(self):
        assert len(_f(2457520.9)) <= 8
        assert len(_f(2.81e-9)) <= 8
        assert float(_f(2.455e6)) == pytest.approx(2.455e6, rel=1e-3)


class TestBoomDeck:
    def test_hub_grids_match_config(self, boom_deck):
        model, vtol = boom_deck
        for r in vtol.rotors:
            assert r.hub_node_id in model.nodes, r.label
            p = model.nodes[r.hub_node_id].xyz_global
            assert np.allclose(p, r.hub_position, atol=0.1)

    def test_properties_parsed(self, boom_deck):
        model, _ = boom_deck
        boom = model.properties[990001]
        assert boom.A == pytest.approx(1457.7, rel=1e-3)
        assert model.materials[990001].E == pytest.approx(71000.0)

    def test_rotor_masses(self, boom_deck):
        model, vtol = boom_deck
        rotor_conms = [c for c in model.masses.values()
                       if getattr(c, "node_id", 0) >= 990000]
        assert len(rotor_conms) == len(vtol.rotors)
        total_kg = sum(c.mass for c in rotor_conms) * 1000
        assert total_kg == pytest.approx(vtol.total_rotor_mass_kg, rel=1e-3)

    def test_cbar_orientation_not_parallel(self, boom_deck):
        """방향벡터가 부재축과 평행하면 요소 강성이 퇴화한다."""
        model, _ = boom_deck
        for eid, e in model.elements.items():
            if not (990000 <= eid < 991000) or e.type != "CBAR":
                continue
            n1, n2 = e.node_ids
            axis = (model.nodes[n2].xyz_global
                    - model.nodes[n1].xyz_global)
            axis = axis / np.linalg.norm(axis)
            v = np.asarray(e.x, dtype=float)
            v = v / np.linalg.norm(v)
            assert abs(abs(np.dot(axis, v)) - 1.0) > 0.05, \
                f"CBAR {eid}: v parallel to axis"

    def test_pylons_attach_to_wing(self, boom_deck):
        model, _ = boom_deck
        wing_attach = 0
        for eid, e in model.elements.items():
            if 990000 <= eid < 991000 and e.type == "CBAR":
                for nid in e.node_ids:
                    if 300000 <= nid < 500000:
                        wing_attach += 1
        assert wing_attach == 12  # 6 붐 × 파일런 2
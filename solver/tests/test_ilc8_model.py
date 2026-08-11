# ILC-8 쉘 FE 생성기(models/ilc8.py) 스모크 테스트 — 질량/CG/로터 구성 정합
from __future__ import annotations

import math

import numpy as np
import pytest

from nastaero.bdf.parser import parse_bdf
from nastaero.loads_analysis.trim_loads import compute_node_masses
from nastaero.models.ilc8 import MTOW_T, CG_X_TARGET, build_ilc8, make_ilc8_vtol_config


@pytest.fixture(scope="module")
def ilc8_model(tmp_path_factory):
    out = tmp_path_factory.mktemp("ilc8")
    bdf = build_ilc8(str(out))
    return parse_bdf(bdf)


def test_ilc8_parses_clean(ilc8_model):
    m = ilc8_model
    assert len(m.nodes) > 2000
    assert len(m.elements) > 3000
    assert len(m.caero_panels) == 6
    assert len(m.splines) == 6
    assert len(m.trims) == 3


def test_ilc8_mass_and_cg(ilc8_model):
    """2-패스 밸러스트로 MTOW/CG 목표에 수렴해야 한다."""
    nm = compute_node_masses(ilc8_model)
    m_tot = sum(nm.values())
    assert m_tot == pytest.approx(MTOW_T, abs=0.002)
    cg_x = sum(mass * ilc8_model.nodes[n].xyz_global[0]
               for n, mass in nm.items() if n in ilc8_model.nodes) / m_tot
    assert cg_x == pytest.approx(CG_X_TARGET, abs=5.0)


def test_ilc8_symmetry(ilc8_model):
    """질량 분포 y-대칭 (CG y ≈ 0)."""
    nm = compute_node_masses(ilc8_model)
    m_tot = sum(nm.values())
    cg_y = sum(mass * ilc8_model.nodes[n].xyz_global[1]
               for n, mass in nm.items() if n in ilc8_model.nodes) / m_tot
    assert abs(cg_y) < 2.0


def test_ilc8_vtol_config_hub_nodes(ilc8_model):
    """VTOLConfig 허브 노드가 전부 덱에 존재하고 위치가 일치해야 한다."""
    cfg = make_ilc8_vtol_config()
    assert cfg.n_lift_rotors == 8
    for r in cfg.rotors:
        assert r.hub_node_id in ilc8_model.nodes, r.label
        p = ilc8_model.nodes[r.hub_node_id].xyz_global
        assert np.allclose(p, r.hub_position, atol=1.0), r.label


def test_ilc8_rotor_meets_hover_thrust():
    """블레이드 사이징: 호버 목표 추력을 콜렉티브 포화 없이 내야 한다."""
    from nastaero.rotor.bemt_solver import BEMTSolver

    cfg = make_ilc8_vtol_config()
    r = cfg.hover_rotors[0]
    target = MTOW_T * 1000 * 9.80665 / cfg.n_hover_rotors
    res = BEMTSolver(r.blade, r.n_blades).solve_for_thrust(
        target, r.rpm_hover, 1.225)
    assert res.thrust == pytest.approx(target, rel=0.02)
    assert math.degrees(res.collective_rad) < 20.0


def test_ilc8_materials_realistic(ilc8_model):
    """CFRP 준등방/UD + Al 7050 — 스미어드 등가 물성 범위 확인."""
    mats = ilc8_model.materials
    assert mats[1001].E == pytest.approx(52000.0)    # CFRP QI
    assert mats[1001].rho == pytest.approx(1.55e-9)
    assert mats[1002].E == pytest.approx(71000.0)    # Al 7050
    assert mats[1003].E == pytest.approx(105000.0)   # CFRP UD 캡


def test_ilc8_rotor_blades_fe(ilc8_model):
    """블레이드 FE: 리프트 8기x4엽x3세그 + 푸셔 3엽x3세그 CBAR."""
    blades = [e for e in ilc8_model.elements.values()
              if e.type == "CBAR" and getattr(e, "pid", 0) in (1801, 1802)]
    assert len(blades) == 8 * 4 * 3 + 3 * 3


def test_ilc8_display_deck(tmp_path_factory, ilc8_model):
    """표시 덱: 블레이드 CAERO 32개 추가, 해석 덱은 6개 유지."""
    out = tmp_path_factory.mktemp("ilc8d")
    build_ilc8(str(out))
    disp = parse_bdf(str(out / "ilc8_display.bdf"))
    assert len(ilc8_model.caero_panels) == 6
    assert len(disp.caero_panels) == 6 + 32
    assert len(disp.splines) == 6   # 로터 패널은 스플라인 없음

# VTOL 붐/파일런 FE 생성기 — 로터 허브를 CBAR 붐+스트럿으로 날개 스파에 연결하는 벌크 카드 생성
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import logger

# ID 대역 (기존 모델과 충돌 없는 99xxxx)
_MAT_ID = 990001
_PBAR_BOOM = 990001
_PBAR_PYLON = 990002
_NODE_BASE = 990100     # 중간 절점 (허브는 VTOLConfig.hub_node_id = 990001+)
_ELEM_BASE = 990100
_CONM_BASE = 990500

# 알루미늄 7075-T6 (N-mm-sec): E[MPa], nu, rho[tonne/mm^3]
_AL_E = 71000.0
_AL_NU = 0.33
_AL_RHO = 2.81e-9


def _tube_section(od: float, t: float) -> Tuple[float, float, float]:
    """원형 튜브 단면 (A, I, J)."""
    ri, ro = od / 2 - t, od / 2
    area = np.pi * (ro ** 2 - ri ** 2)
    inertia = np.pi / 4 * (ro ** 4 - ri ** 4)
    torsion = 2 * inertia
    return area, inertia, torsion


def _nearest_wing_node(model, target: np.ndarray, y_station: float,
                       prefer_low_z: bool = True) -> int:
    """붐 스테이션 근처에서 목표점에 가장 가까운 날개 절점을 찾는다.

    GACOMP 절점 대역: 좌 300k-399k, 우 400k-499k. 하면 부착을 위해
    수평거리 우선 + 낮은 z 선호.
    """
    lo, hi = (400000, 500000) if y_station > 0 else (300000, 400000)
    best, best_key = None, None
    for nid, g in model.nodes.items():
        if not (lo <= nid < hi):
            continue
        p = g.xyz_global
        if abs(p[1] - y_station) > 120.0:
            continue
        d_xy = np.hypot(p[0] - target[0], p[1] - target[1])
        key = (d_xy, p[2] if prefer_low_z else -p[2])
        if best_key is None or key < best_key:
            best, best_key = nid, key
    if best is None:
        raise ValueError(f"No wing node near y={y_station:.0f}, "
                         f"x={target[0]:.0f}")
    return best


def _f(v: float) -> str:
    """8열 고정 필드에 맞는 숫자 문자열 (gui.card_form.fit8과 동일 로직)."""
    s = f"{v:.6g}"
    if len(s) <= 8:
        return s
    for prec in range(7, 0, -1):
        s = f"{v:.{prec}g}"
        if len(s) <= 8:
            return s
    return f"{v:.1g}"[:8]


def _card(*fields) -> str:
    return "".join(f"{str(x):<8s}" for x in fields).rstrip()


def build_vtol_boom_bulk(model, vtol_config,
                         boom_od: float = 120.0, boom_t: float = 4.0,
                         pylon_od: float = 100.0, pylon_t: float = 4.0,
                         spar_x: Tuple[float, float] = (3650.0, 4540.0),
                         ) -> str:
    """VTOL 붐/파일런 벌크 데이터 (INCLUDE 파일 내용) 생성.

    반환 텍스트는 GRID(허브+붐 절점), CBAR(붐·파일런), PBAR/MAT1,
    CONM2(로터 질량)로 구성되며 ENDDATA 없이 벌크 카드만 담는다.
    """
    lines: List[str] = [
        "$ =====================================================",
        "$ VTOL rotor boom/pylon structure (generated)",
        f"$ boom tube OD{boom_od:.0f}xt{boom_t:.0f}, "
        f"pylon OD{pylon_od:.0f}xt{pylon_t:.0f}, AL7075",
        "$ =====================================================",
        _card("MAT1", _MAT_ID, _f(_AL_E), "", _f(_AL_NU), _f(_AL_RHO)),
    ]
    a, i, j = _tube_section(boom_od, boom_t)
    lines.append(_card("PBAR", _PBAR_BOOM, _MAT_ID, _f(a), _f(i), _f(i),
                       _f(j)))
    a, i, j = _tube_section(pylon_od, pylon_t)
    lines.append(_card("PBAR", _PBAR_PYLON, _MAT_ID, _f(a), _f(i), _f(i),
                       _f(j)))

    # 로터를 (부호 있는 y) 스테이션으로 짝짓기: 전방(x 작은)·후방(x 큰)
    stations: Dict[float, Dict[str, object]] = {}
    for r in vtol_config.rotors:
        y = round(float(r.hub_position[1]), 1)
        stations.setdefault(y, {})[
            "fwd" if r.hub_position[0] < 3900 else "aft"] = r

    node_id = _NODE_BASE
    eid = _ELEM_BASE
    conm = _CONM_BASE
    positions: Dict[int, np.ndarray] = {}   # 생성 절점 좌표 (방향벡터 판정용)

    def grid(nid: int, p) -> None:
        positions[nid] = np.asarray(p, dtype=float)
        lines.append(_card("GRID", nid, "", _f(p[0]), _f(p[1]), _f(p[2])))

    def _pos(nid: int) -> np.ndarray:
        if nid in positions:
            return positions[nid]
        return model.nodes[nid].xyz_global

    def cbar(pid: int, ga: int, gb: int) -> None:
        nonlocal eid
        # 방향벡터는 부재축과 평행하면 안 된다: 수평 부재(붐)는 +Z,
        # 수직에 가까운 부재(파일런)는 +X 사용
        axis = _pos(gb) - _pos(ga)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        v = ("0.", "0.", "1.") if abs(axis[2]) < 0.9 else ("1.0", "0.", "0.")
        lines.append(_card("CBAR", eid, pid, ga, gb, *v))
        eid += 1

    for y, pair in sorted(stations.items()):
        fwd, aft = pair.get("fwd"), pair.get("aft")
        if fwd is None or aft is None:
            logger.warning("Boom station y=%.0f: unpaired rotor, skipped", y)
            continue
        p_f = np.asarray(fwd.hub_position, dtype=float)
        p_a = np.asarray(aft.hub_position, dtype=float)
        lines.append(f"$ --- boom station y={y:.0f} "
                     f"({fwd.label} + {aft.label}) ---")
        # 허브 GRID (VTOLConfig hub_node_id와 동일 번호)
        grid(fwd.hub_node_id, p_f)
        grid(aft.hub_node_id, p_a)

        # 붐 직선상의 파일런 하단 절점 (전/후 스파 x 위치)
        boom_pts = []
        for x in spar_x:
            s = (x - p_f[0]) / (p_a[0] - p_f[0])
            p = p_f + s * (p_a - p_f)
            node_id += 1
            grid(node_id, p)
            boom_pts.append(node_id)

        # 붐: 허브F - 파일런F - 파일런A - 허브A
        chain = [fwd.hub_node_id, boom_pts[0], boom_pts[1], aft.hub_node_id]
        for ga, gb in zip(chain, chain[1:]):
            cbar(_PBAR_BOOM, ga, gb)

        # 파일런: 붐 절점 → 날개 하면 스파 절점 (기존 GRID에 직결)
        for bp, x in zip(boom_pts, spar_x):
            wing_nid = _nearest_wing_node(
                model, np.array([x, y, 0.0]), y_station=y)
            cbar(_PBAR_PYLON, bp, wing_nid)

        # 로터 질량 (tonne)
        for r in (fwd, aft):
            mass_t = float(getattr(r, "mass_kg", 0.0)) * 1e-3
            if mass_t > 0:
                conm += 1
                lines.append(_card("CONM2", conm, r.hub_node_id, 0,
                                   _f(mass_t)))

    return "\n".join(lines) + "\n"

# ILC-8 (NASA L+C 참조, 1,900kg 8로터 lift+cruise) 쉘 기반 전기체 FE/공력 모델 생성기
"""Procedural built-up shell FE model of the ILC-8 lift+cruise aircraft.

논문(ijass_hover_gust §Case Study)과 NASA L+C 참조 형상 기준:
  - 동체 9 m x D1.4 m (스킨 CQUAD4 + 프레임/론저론 CBAR)
  - 날개 시위 1.35 m, 반스팬 6 m, 고익(z=550), 2-스파 박스
    (상/하 스킨 + 스파 웹 + 리브 CQUAD4, 스파 캡 CBAR)
  - V-tail 상반각 40°, 러더베이터(ELEVR/ELEVL + AELINK ELEV/RUD)
  - 붐 4개(y=±2, ±4 m) 쉘 튜브 + 마스트, 로터 허브 GRID 8 + 푸셔
  - 질량 1,900 kg / 목표 CG는 밸러스트 2점으로 자동 해결
  - DLM: 날개/V-tail/러더베이터 CAERO1 + IPS SPLINE1
  - 1g 트림 서브케이스 3개(V=40/60/80 m/s EAS), 3-2-1 기준 지지

Usage:
    from ascent_load.models.ilc8 import build_ilc8, make_ilc8_vtol_config
    build_ilc8("tests/validation/ILC8")   # ilc8.bdf 생성
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import logger

# ---------------------------------------------------------------------------
# 주요 치수 (mm) — 논문 형상도와 일치
# ---------------------------------------------------------------------------
FUS_L = 9000.0
FUS_R = 700.0
WING_LE = 3850.0        # (참조) 직사각 시절 앞전 — MAC 위치 근거
WING_C = 1350.0         # 기준 시위 (AEROS refc, MAC≈1378과 정합)
WING_HALF = 6000.0
WING_Z = 550.0          # 날개 중립면 (고익)
WING_TC = 0.15
# 테이퍼 평면형 (NASA L+C 근접): λ=0.6, 면적 16.2 m² 유지,
# 25% 시위선 무후퇴 (AC 라인 y-불변)
WING_TAPER = 0.6
WING_C_ROOT = 2 * WING_C / (1 + WING_TAPER)          # 1687.5
WING_C_TIP = WING_TAPER * WING_C_ROOT                # 1012.5
WING_QC = WING_LE + 0.25 * WING_C                    # 4187.5 (25% 시위선 x)


def wing_chord_le(y: float) -> Tuple[float, float]:
    """스팬 위치 y의 (시위, 앞전 x) — 25% 시위선 고정 테이퍼."""
    t = min(abs(y) / WING_HALF, 1.0)
    c = WING_C_ROOT + (WING_C_TIP - WING_C_ROOT) * t
    return c, WING_QC - 0.25 * c
FS_XI, RS_XI = 0.15, 0.65          # 전/후 스파 시위 위치
VT_ROOT_LE = 7800.0
VT_ROOT_C = 900.0
VT_TIP_C = 600.0
VT_SPAN = 1900.0        # 경사 스팬
VT_DIH = math.radians(40.0)
VT_SWEEP = math.radians(20.0)
BOOM_YS = (-4000.0, -2000.0, 2000.0, 4000.0)
BOOM_X0, BOOM_X1 = 2000.0, 7000.0
BOOM_Z, BOOM_R = 250.0, 125.0
HUB_Z = 950.0
ROTOR_X_F, ROTOR_X_A = 3000.0, 6000.0
MTOW_T = 1.900          # tonne
CG_X_TARGET = 4450.0    # 로터 중심(4500)과 날개 AC(4190) 사이

# ID 블록
# 재료 — 요소 정식화가 등방성 전용이므로 복합재는 스미어드 등가 등방(MAT1).
# eVTOL 주구조는 카본/에폭시, 기계가공 피팅류는 알루미늄 (초기설계 관행).
MAT_CFRP_QI = 1001   # 준등방 라미네이트 (T700급): E 52 GPa, ρ 1.55e-9
MAT_AL7050 = 1002    # 피팅/프레임/마스트: E 71 GPa, ρ 2.83e-9
MAT_CFRP_UD = 1003   # 0° 지배 스파캡/론저론 스미어드: E 105 GPa
MAT_CFRP_BL = 1004   # 블레이드 스파 혼합 라미네이트: E 80 GPa
PSH_FUS, PSH_WSKIN, PSH_WEB, PSH_RIB = 1101, 1401, 1402, 1403
PSH_VT, PSH_VTWEB, PSH_BOOM = 1601, 1602, 1701
PB_FRAME, PB_LONG, PB_CAP, PB_MAST, PB_STRUT = 1201, 1202, 1404, 1702, 1703
PB_BLADE_L, PB_BLADE_P = 1801, 1802   # 리프트/푸셔 블레이드 보
PB_GEAR = 1704                        # 착륙장치 레그 (Al 7050)
HUB_BASE = 990100       # 허브 990101~990108, 푸셔 990201
BLADE_NODE_BASE = 991000   # 블레이드 절점 991r b s (로터 r, 블레이드 b, 스테이션 s)
# 블레이드 질량 (PBAR 밀도로 자동 산정): 리프트 0.36 kg/엽, 푸셔 0.26 kg/엽
BLADE_M_LIFT = 4 * 300.0 * 1.60e-9 * 750.0     # ≈1.44e-3 t/로터
BLADE_M_PUSH = 3 * 250.0 * 1.60e-9 * 650.0     # ≈0.78e-3 t


def _f(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))          # 정수 ID는 지수 표기 금지
    s = f"{v:.6g}"
    if len(s) <= 8:
        return s
    for prec in range(7, 0, -1):
        s = f"{v:.{prec}g}"
        if len(s) <= 8:
            return s
    return f"{v:.1g}"[:8]


class Deck:
    """8열 고정 필드 벌크 카드 작성기 (자동 연속행)."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.nodes: Dict[int, np.ndarray] = {}
        self._eid = 1

    def raw(self, text: str) -> None:
        self.lines.append(text)

    def card(self, *fields) -> None:
        vals = [_f(v) for v in fields]
        head, rest = vals[0], vals[1:]
        chunks = [rest[i:i + 8] for i in range(0, len(rest), 8)] or [[]]
        for i, chunk in enumerate(chunks):
            lead = head if i == 0 else "+"
            self.lines.append(
                "".join(f"{x:<8s}" for x in [lead] + chunk).rstrip())

    def grid(self, nid: int, p) -> int:
        p = np.asarray(p, dtype=float)
        self.nodes[nid] = p
        self.card("GRID", nid, "", p[0], p[1], p[2])
        return nid

    def quad(self, pid: int, n1: int, n2: int, n3: int, n4: int) -> None:
        self.card("CQUAD4", self._eid, pid, n1, n2, n3, n4)
        self._eid += 1

    def bar(self, pid: int, ga: int, gb: int) -> None:
        axis = self.nodes[gb] - self.nodes[ga]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        v = ("0.", "0.", "1.") if abs(axis[2]) < 0.9 else ("1.0", "0.", "0.")
        self.card("CBAR", self._eid, pid, ga, gb, *v)
        self._eid += 1

    def conm2(self, nid: int, mass_t: float, cid_counter: List[int]) -> None:
        cid_counter[0] += 1
        self.card("CONM2", cid_counter[0], nid, 0, mass_t)

    def rbe2(self, master: int, slaves: Sequence[int]) -> None:
        self.card("RBE2", self._eid, master, "123456", *slaves)
        self._eid += 1


# ---------------------------------------------------------------------------
# 동체
# ---------------------------------------------------------------------------
def _fus_radius(x: float) -> float:
    if x < 1500.0:       # 노즈 (1/4 타원)
        t = (1500.0 - x) / 1500.0
        return max(FUS_R * math.sqrt(max(1.0 - t * t, 0.0)), 120.0)
    if x <= 6500.0:
        return FUS_R
    return FUS_R + (x - 6500.0) / (FUS_L - 6500.0) * (150.0 - FUS_R)


def build_fuselage(d: Deck) -> Dict[str, List[int]]:
    xs = ([100.0, 450.0, 850.0, 1200.0]
          + [1500.0 + 250.0 * k for k in range(21)]      # 1500..6500
          + [6750.0 + 250.0 * k for k in range(10)])     # 6750..9000
    ntheta = 24
    ids = {}
    for i, x in enumerate(xs):
        r = _fus_radius(x)
        for j in range(ntheta):
            th = 2 * math.pi * j / ntheta
            nid = 100001 + i * 100 + j
            d.grid(nid, (x, r * math.cos(th), r * math.sin(th)))
            ids[(i, j)] = nid
    # 스킨
    for i in range(len(xs) - 1):
        for j in range(ntheta):
            j2 = (j + 1) % ntheta
            d.quad(PSH_FUS, ids[(i, j)], ids[(i + 1, j)],
                   ids[(i + 1, j2)], ids[(i, j2)])
    # 프레임 (2스테이션마다 링)
    for i in range(0, len(xs), 2):
        for j in range(ntheta):
            d.bar(PB_FRAME, ids[(i, j)], ids[(i, (j + 1) % ntheta)])
    # 론저론 4개 (45/135/225/315°)
    for j in (3, 9, 15, 21):
        for i in range(len(xs) - 1):
            d.bar(PB_LONG, ids[(i, j)], ids[(i + 1, j)])
    bottom = [ids[(i, 18)] for i in range(len(xs))]   # θ=270°
    return {"xs": xs, "ids": ids, "ntheta": ntheta, "bottom": bottom}


# ---------------------------------------------------------------------------
# 날개
# ---------------------------------------------------------------------------
XI = [0.0, 0.06, FS_XI, 0.30, 0.45, RS_XI, 0.82, 1.0]   # 상면 시위점
N_RING = 2 * len(XI) - 2                                  # LE/TE 공유 = 14


def _airfoil_z(xi: float, upper: bool) -> float:
    shape = math.sin(math.pi * xi ** 0.78) if 0 < xi < 1 else 0.0
    half = WING_C * WING_TC / 2.0 * shape
    return half if upper else -0.62 * half


def _wing_stations() -> List[float]:
    ys = list(np.linspace(700.0, WING_HALF, 21))
    for target in (2000.0, 4000.0):
        k = int(np.argmin([abs(y - target) for y in ys]))
        ys[k] = target
    return ys


def _wing_ring(d: Deck, base: int, s: int, y: float) -> List[int]:
    """스테이션 s의 링 절점 14개 생성. 순서: 상면 LE→TE(0..7), 하면 내부(8..13).

    시위/앞전은 테이퍼 평면형(wing_chord_le)을 따르고 두께도 시위에
    비례해 줄어든다 (일정 t/c).
    """
    c, le = wing_chord_le(y)
    scale = c / WING_C
    ring = []
    for k, xi in enumerate(XI):
        nid = base + s * 100 + k
        d.grid(nid, (le + xi * c, y, WING_Z + scale * _airfoil_z(xi, True)))
        ring.append(nid)
    for k, xi in enumerate(XI[1:-1]):
        nid = base + s * 100 + 8 + k
        d.grid(nid, (le + xi * c, y, WING_Z + scale * _airfoil_z(xi, False)))
        ring.append(nid)
    return ring


def build_wing(d: Deck, side: int, fus: dict) -> Dict[str, List[int]]:
    """side=+1 우현(400k), -1 좌현(300k)."""
    base = 400000 if side > 0 else 300000
    ys = [side * y for y in _wing_stations()]
    rings = [_wing_ring(d, base, s, y) for s, y in enumerate(ys)]
    up = lambda r, k: r[k]           # 상면 0..7
    lo = lambda r, k: r[0] if k == 0 else (r[7] if k == 7 else r[7 + k])

    for s in range(len(ys) - 1):
        a, b = rings[s], rings[s + 1]
        for k in range(7):           # 상면 스킨
            d.quad(PSH_WSKIN, up(a, k), up(a, k + 1), up(b, k + 1), up(b, k))
        for k in range(7):           # 하면 스킨
            d.quad(PSH_WSKIN, lo(a, k), lo(b, k), lo(b, k + 1), lo(a, k + 1))
        for k in (2, 5):             # 스파 웹 (FS=idx2, RS=idx5)
            d.quad(PSH_WEB, up(a, k), up(b, k), lo(b, k), lo(a, k))
    # 리브: 짝수 스테이션 + 붐/루트/팁
    rib_ss = {s for s in range(len(ys)) if s % 2 == 0}
    rib_ss |= {s for s, y in enumerate(ys) if abs(abs(y) - 2000) < 1
               or abs(abs(y) - 4000) < 1 or s == len(ys) - 1}
    for s in sorted(rib_ss):
        r = rings[s]
        for k in (2, 3, 4):
            d.quad(PSH_RIB, up(r, k), up(r, k + 1), lo(r, k + 1), lo(r, k))
    # 스파 캡 4줄
    for k, getter in ((2, up), (5, up), (2, lo), (5, lo)):
        for s in range(len(ys) - 1):
            d.bar(PB_CAP, getter(rings[s], k), getter(rings[s + 1], k))
    # 루트 → 동체 RBE2 (링 절점마다 최근접 동체 절점에 종속)
    fus_pts = {nid: d.nodes[nid] for (i, j), nid in fus["ids"].items()}
    for nid in rings[0]:
        p = d.nodes[nid]
        master = min(fus_pts, key=lambda m: np.linalg.norm(fus_pts[m] - p))
        d.rbe2(master, [nid])
    return {"ys": ys, "rings": rings, "base": base}


# ---------------------------------------------------------------------------
# V-tail
# ---------------------------------------------------------------------------
VXI = [0.0, 0.15, 0.40, 0.65, 0.85, 1.0]


def build_vtail(d: Deck, side: int, fus: dict) -> dict:
    base = 600000 if side > 0 else 500000
    n_st = 8
    u = np.array([0.0, side * math.cos(VT_DIH), math.sin(VT_DIH)])
    nrm = np.array([0.0, -side * math.sin(VT_DIH), math.cos(VT_DIH)])
    root = np.array([VT_ROOT_LE, side * 250.0, 300.0])
    rings = []
    for s in range(n_st):
        t = s / (n_st - 1)
        span = t * VT_SPAN
        le = root + span * u + np.array([span * math.tan(VT_SWEEP), 0, 0])
        c = VT_ROOT_C + t * (VT_TIP_C - VT_ROOT_C)
        ring = []
        for k, xi in enumerate(VXI):
            zair = c * 0.10 / 2 * (math.sin(math.pi * xi ** 0.8)
                                   if 0 < xi < 1 else 0.0)
            nid = base + s * 100 + k
            d.grid(nid, le + np.array([xi * c, 0, 0]) + zair * nrm)
            ring.append(nid)
        for k, xi in enumerate(VXI[1:-1]):
            zair = -c * 0.10 / 2 * (math.sin(math.pi * xi ** 0.8))
            nid = base + s * 100 + 6 + k
            d.grid(nid, le + np.array([xi * c, 0, 0]) + zair * nrm)
            ring.append(nid)
        rings.append(ring)
    up = lambda r, k: r[k]
    lo = lambda r, k: r[0] if k == 0 else (r[5] if k == 5 else r[5 + k])
    for s in range(n_st - 1):
        a, b = rings[s], rings[s + 1]
        for k in range(5):
            d.quad(PSH_VT, up(a, k), up(a, k + 1), up(b, k + 1), up(b, k))
            d.quad(PSH_VT, lo(a, k), lo(b, k), lo(b, k + 1), lo(a, k + 1))
        for k in (1, 3):            # 웹 (xi 0.15 / 0.65)
            d.quad(PSH_VTWEB, up(a, k), up(b, k), lo(b, k), lo(a, k))
        if s % 2 == 0:
            for k in (1, 2):
                d.quad(PSH_VTWEB, up(a, k), up(a, k + 1),
                       lo(a, k + 1), lo(a, k))
    for k, getter in ((1, up), (3, up), (1, lo), (3, lo)):
        for s in range(n_st - 1):
            d.bar(PB_CAP, getter(rings[s], k), getter(rings[s + 1], k))
    fus_pts = {nid: d.nodes[nid] for (i, j), nid in fus["ids"].items()}
    for nid in rings[0]:
        p = d.nodes[nid]
        master = min(fus_pts, key=lambda m: np.linalg.norm(fus_pts[m] - p))
        d.rbe2(master, [nid])
    return {"rings": rings, "u": u, "root": root}


# ---------------------------------------------------------------------------
# 붐 + 로터 허브
# ---------------------------------------------------------------------------
def build_booms(d: Deck, wings: dict) -> List[int]:
    """붐 쉘튜브 4개 + 마스트 + 허브 GRID. 반환: 허브 노드 8개 (전열4+후열4)."""
    hub_ids = []
    ntheta = 10
    for bi, y0 in enumerate(BOOM_YS):
        # 파일런 스트럿은 붐 위치의 로컬 스파 x에 맞춘다 (테이퍼 반영)
        c0, le0 = wing_chord_le(y0)
        spar_xs = (le0 + FS_XI * c0, le0 + RS_XI * c0)
        base = 710000 + bi * 10000
        xs = sorted(set(
            [BOOM_X0 + 250.0 * k for k in range(int((BOOM_X1 - BOOM_X0) / 250) + 1)]
            + [ROTOR_X_F, ROTOR_X_A, *spar_xs]))
        ids = {}
        for i, x in enumerate(xs):
            for j in range(ntheta):
                th = 2 * math.pi * j / ntheta
                nid = base + i * 100 + j
                d.grid(nid, (x, y0 + BOOM_R * math.cos(th),
                             BOOM_Z + BOOM_R * math.sin(th)))
                ids[(i, j)] = nid
        for i in range(len(xs) - 1):
            for j in range(ntheta):
                j2 = (j + 1) % ntheta
                d.quad(PSH_BOOM, ids[(i, j)], ids[(i + 1, j)],
                       ids[(i + 1, j2)], ids[(i, j2)])
        # 링 프레임(오벌링 억제): 매 2스테이션
        for i in range(0, len(xs), 2):
            for j in range(ntheta):
                d.bar(PB_FRAME, ids[(i, j)], ids[(i, (j + 1) % ntheta)])

        def center_spider(x_t: float, tag: int) -> int:
            """스테이션 x_t에 중심 GRID + RBE2 스파이더 — 점하중 도입부."""
            i_t = int(np.argmin([abs(x - x_t) for x in xs]))
            cnode = base + 9000 + tag
            d.grid(cnode, (xs[i_t], y0, BOOM_Z))
            d.rbe2(cnode, [ids[(i_t, j)] for j in range(ntheta)])
            return cnode

        # 파일런 스트럿: 붐 중심(스파이더) → 날개 하면 스파 절점
        side = 1 if y0 > 0 else -1
        wing = wings[side]
        s_wing = int(np.argmin([abs(y - y0) for y in wing["ys"]]))
        ring = wing["rings"][s_wing]
        lo_fs, lo_rs = ring[7 + 2], ring[7 + 5]   # 하면 FS/RS (idx9, idx12)
        for t, (x_sp, wnode) in enumerate(zip(spar_xs, (lo_fs, lo_rs))):
            d.bar(PB_STRUT, center_spider(x_sp, 10 + t), wnode)
        # 마스트 + 허브 (역시 스파이더 중심에서)
        for x_r, hub_off, tag in ((ROTOR_X_F, 1, 20), (ROTOR_X_A, 5, 21)):
            hub = HUB_BASE + hub_off + bi
            d.grid(hub, (x_r, y0, HUB_Z))
            d.bar(PB_MAST, center_spider(x_r, tag), hub)
            hub_ids.append(hub)
    return hub_ids


# ---------------------------------------------------------------------------
# 질량 (밸러스트 제외 고정 항목)
# ---------------------------------------------------------------------------
def build_masses(d: Deck, fus: dict, wings: dict, hub_ids: List[int],
                 cid: List[int], pusher: bool = True,
                 tilt_nacelle_t: float = 0.0,
                 nose_trim_t: float = 0.0) -> Optional[int]:
    """장비/배터리/승객/모터 CONM2. 반환: 푸셔 허브 노드(없으면 None).

    pusher=False + tilt_nacelle_t>0은 틸트 변형(ILC-8T): 푸셔를 빼고
    전열 허브 4곳에 틸트 나셀·액추에이터 질량을 더한다."""
    bottom, xs = fus["bottom"], fus["xs"]

    def floor_nodes(x0, x1):
        return [bottom[i] for i, x in enumerate(xs) if x0 <= x <= x1]

    # 배터리 총 0.62 t (MTOW 33%) — CFRP 경량화분을 배터리로 환원.
    # 날개 박스 0.39 + 동체 전방 0.08 + 테일콘 후방 팩 0.15 (CG 밸런스)
    bat_nodes = []
    for side in (1, -1):
        w = wings[side]
        for s, y in enumerate(w["ys"]):
            if 800 <= abs(y) <= 3600:
                r = w["rings"][s]
                bat_nodes += [r[7 + 2], r[7 + 3], r[7 + 4], r[7 + 5]]
    for nid in bat_nodes:
        d.conm2(nid, 0.38 / len(bat_nodes), cid)
    # 동체 배터리(전방 0.08 / 후방 0.16) + 승객/페이로드 0.45 t: 캐빈 플로어
    # (CG 4450 목표에 맞춰 캐빈 창을 로터 중심쪽으로 배치)
    for group, (x0, x1) in (("0.08", (2400, 4200)), ("0.16", (5400, 7000)),
                            ("0.45", (2600, 5000))):
        nodes = floor_nodes(x0, x1)
        for nid in nodes:
            d.conm2(nid, float(group) / len(nodes), cid)
    # 항전 0.06 (노즈), 시스템 0.075 (후방), 내장 0.035 (캐빈)
    # — 테이퍼 날개(루트 시위 확대)로 구조 도심이 전진한 만큼 후방 재배분
    for m, (x0, x1) in ((max(0.06 + nose_trim_t, 0.005), (100, 1300)),
                        (0.075, (5600, 6800)),
                        (0.035, (2400, 4800))):
        nodes = floor_nodes(x0, x1)
        for nid in nodes:
            d.conm2(nid, m / len(nodes), cid)
    # 착륙장치 질량은 build_gear가 레그 끝 절점에 부여 (하중 경로 재현)
    # 리프트 모터+허브 8 x 25 kg — 블레이드는 FE 보(밀도 질량)로 분리
    for k, hub in enumerate(hub_ids):
        extra = tilt_nacelle_t if k < 4 else 0.0   # 전열 = 틸트 나셀
        d.conm2(hub, 0.025 - BLADE_M_LIFT + extra, cid)
    if not pusher:
        return None
    # 푸셔: 테일콘 끝 허브 + RBE2 + (45 kg − 블레이드)
    pusher_nid = HUB_BASE + 101   # 990201
    d.grid(pusher_nid, (FUS_L + 150.0, 0.0, 0.0))
    tail_ring = [fus["ids"][(len(xs) - 1, j)] for j in range(fus["ntheta"])]
    d.rbe2(pusher_nid, tail_ring)  # 허브가 독립, 테일 링 종속 — 아래 주석 참고
    d.conm2(pusher_nid, 0.045 - BLADE_M_PUSH, cid)
    return pusher_nid


# ---------------------------------------------------------------------------
# 로터 블레이드 — FE 보 (형상 + 질량/동특성) 및 표시용 CAERO 패널
# ---------------------------------------------------------------------------
def build_gear(d: Deck, fus: dict, cid: List[int]) -> None:
    """착륙장치 레그 FE — 노즈 1주 + 메인 A형 좌/우.

    지면 반력이 레그 굽힘 → 동체 프레임 경로로 전달되도록 하고, 기어
    질량(노즈 30/메인 45 kg)은 레그 끝에 둔다. 드라이버 `_gear_nodes`는
    동체 최저 z 절점을 찾으므로 레그 끝이 자동으로 부착점이 된다.
    """
    xs, bottom = fus["xs"], fus["bottom"]

    def leg(x_att: float, y_end: float, base_nid: int, m_end: float) -> None:
        i = int(np.argmin([abs(x - x_att) for x in xs]))
        top = bottom[i]
        p0 = d.nodes[top]
        mid = base_nid
        end = base_nid + 1
        d.grid(mid, (p0[0], (p0[1] + y_end) / 2, (p0[2] - 1250.0) / 2))
        d.grid(end, (p0[0], y_end, -1250.0))
        d.bar(PB_GEAR, top, mid)
        d.bar(PB_GEAR, mid, end)
        # 드래그 브레이스: 앞/뒤 벨리 절점 → mid — 전후/횡 스윙 구속
        # (없으면 레그가 단일 기둥이라 0.1~2 Hz 진자 모드가 생긴다)
        for i_b in (max(i - 2, 0), min(i + 2, len(bottom) - 1)):
            d.bar(PB_GEAR, bottom[i_b], mid)
        d.conm2(end, m_end, cid)

    leg(1500.0, 0.0, 992001, 0.030)          # 노즈 (992002 = 끝)
    leg(4700.0, 600.0, 992011, 0.0225)       # 메인 우 (992012 = 끝)
    leg(4700.0, -600.0, 992021, 0.0225)      # 메인 좌 (992022 = 끝)


def build_rotor_blades(d: Deck, hub_ids: List[int],
                       pusher: Optional[int]) -> None:
    """허브에서 방사형 테이퍼 보 블레이드 — 리프트 4엽, 푸셔 3엽.

    질량은 PBAR 밀도로 관성/CG에 자동 반영되고, 인접 로터끼리 45°
    위상차를 둬 정지 시 블레이드 간섭이 없게 한다.
    """
    for r_i, hub in enumerate(hub_ids):
        hp = d.nodes[hub]
        phase = math.pi / 4 if (r_i % 2) else 0.0
        for b in range(4):
            th = math.pi / 2 * b + phase
            u = np.array([math.cos(th), math.sin(th), 0.0])
            prev = hub
            for s, rr in enumerate((300.0, 550.0, 750.0), start=1):
                nid = BLADE_NODE_BASE + r_i * 100 + b * 10 + s
                d.grid(nid, hp + u * rr)
                d.bar(PB_BLADE_L, prev, nid)
                prev = nid
    if pusher is None:
        return
    hp = d.nodes[pusher]
    for b in range(3):
        th = 2 * math.pi * b / 3
        u = np.array([0.0, math.cos(th), math.sin(th)])
        prev = pusher
        for s, rr in enumerate((280.0, 480.0, 650.0), start=1):
            nid = BLADE_NODE_BASE + 900 + b * 10 + s
            d.grid(nid, hp + u * rr)
            d.bar(PB_BLADE_P, prev, nid)
            prev = nid


def build_rotor_panels(d: Deck, hub_ids: List[int]) -> None:
    """표시 전용 리프트 블레이드 CAERO1 패널 (IGID 99, SPLINE 없음).

    DLM AIC에 들어가면 트림을 오염시키므로 해석 덱(ilc8.bdf)에는 넣지
    않고 형상 확인용 ilc8_display.bdf에만 포함한다.
    """
    for r_i, hub in enumerate(hub_ids):
        hp = d.nodes[hub]
        phase = math.pi / 4 if (r_i % 2) else 0.0
        for b in range(4):
            th = math.pi / 2 * b + phase
            u = np.array([math.cos(th), math.sin(th), 0.0])
            root = hp + u * 150.0
            tip = hp + u * 750.0
            c_r, c_t = 90.0, 55.0
            eid = 9000001 + r_i * 100000 + b * 10000
            d.card("CAERO1", eid, 1000, 0, 4, 1, 0, 0, 99,
                   root[0] - c_r / 2, root[1], root[2], c_r,
                   tip[0] - c_t / 2, tip[1], tip[2], c_t)


# ---------------------------------------------------------------------------
# 공력 (CAERO1 / SPLINE1 / AESTAT / AESURF / AELINK / TRIM)
# ---------------------------------------------------------------------------
def build_aero(d: Deck, wings: dict, vtails: dict) -> None:
    d.card("AEROS", 0, 0, WING_C, 2 * WING_HALF, 16.2e6)
    d.card("PAERO1", 1000)

    def set1(sid: int, nids: List[int]) -> None:
        d.card("SET1", sid, *nids)

    def spline(eid, caero, nboxes, setg):
        d.card("SPLINE1", eid, caero, caero, caero + nboxes - 1, setg)

    # 날개: 스파 캡 절점(상면 FS/RS) + LE/TE 상면 절점을 스플라인 세트로
    for side, ceid, sid in ((1, 1000001, 101), (-1, 2000001, 102)):
        w = wings[side]
        nids = []
        for r in w["rings"]:
            nids += [r[0], r[2], r[5], r[7]]
        set1(sid, nids)
        # 패널을 중심선까지 연장 — 동체 관통부(carryover) 양력 포함.
        # 평면형은 구조와 동일한 테이퍼 사다리꼴.
        c_r, le_r = wing_chord_le(0.0)
        c_t, le_t = wing_chord_le(WING_HALF)
        d.card("CAERO1", ceid, 1000, 0, 16, 6, 0, 0, 1,
               le_r, 0.0, WING_Z, c_r,
               le_t, side * WING_HALF, WING_Z, c_t)
        spline(30000 + sid, ceid, 16 * 6, sid)

    # V-tail: 본체(앞 70%) + 러더베이터(뒤 30%, 스팬 20~95%)
    for side, ceid_m, ceid_r, sid, hinge_cid, alid in (
            (1, 3000001, 3100001, 103, 11, 3111),
            (-1, 4000001, 4100001, 104, 21, 4111)):
        vt = vtails[side]
        nids = []
        for r in vt["rings"]:
            nids += [r[0], r[1], r[3], r[5]]
        set1(sid, nids)
        u, root = vt["u"], vt["root"]

        def le_c(t):
            span = t * VT_SPAN
            le = root + span * u + np.array([span * math.tan(VT_SWEEP), 0, 0])
            c = VT_ROOT_C + t * (VT_TIP_C - VT_ROOT_C)
            return le, c

        le_r, c_r = le_c(0.0)
        le_t, c_t = le_c(1.0)
        # 본체 (앞 70% 시위)
        d.card("CAERO1", ceid_m, 1000, 0, 6, 3, 0, 0, 1,
               le_r[0], le_r[1], le_r[2], 0.7 * c_r,
               le_t[0], le_t[1], le_t[2], 0.7 * c_t)
        spline(30000 + sid, ceid_m, 6 * 3, sid)
        # 러더베이터 (뒤 30%, 스팬 0.2~0.95)
        le_1, c_1 = le_c(0.20)
        le_2, c_2 = le_c(0.95)
        d.card("CAERO1", ceid_r, 1000, 0, 5, 2, 0, 0, 1,
               le_1[0] + 0.7 * c_1, le_1[1], le_1[2], 0.3 * c_1,
               le_2[0] + 0.7 * c_2, le_2[1], le_2[2], 0.3 * c_2)
        spline(31000 + sid, ceid_r, 5 * 2, sid)
        d.card("AELIST", alid, *[ceid_r + k for k in range(10)])
        # 힌지 좌표계: y축 = 힌지선 방향 (양쪽 모두 전역 +Y 성분 — 부호
        # 해석은 AELINK 계수가 담당)
        hp = le_1 + np.array([0.7 * c_1, 0, 0])
        hy = u.copy()
        if hy[1] < 0:
            hy = -hy   # 전역 +Y 성분 보장
        hz = np.array([0.0, -math.copysign(math.sin(VT_DIH), side),
                       math.cos(VT_DIH)])
        # CORD2R: 원점 / z축 위 점 / xz평면 점
        d.card("CORD2R", hinge_cid, 0,
               hp[0], hp[1], hp[2],
               hp[0] + hz[0] * 100, hp[1] + hz[1] * 100, hp[2] + hz[2] * 100,
               hp[0] + 100, hp[1], hp[2])

    # AESTAT / AESURF / AELINK
    for i, lbl in enumerate(("ANGLEA", "SIDES", "ROLL", "YAW",
                             "URDD2", "URDD4", "URDD6", "ELEV", "RUD")):
        d.card("AESTAT", 1711 + i, lbl)
    d.card("AESURF", 1001, "ELEVR", 11, 3111)
    d.card("AESURF", 1002, "ELEVL", 21, 4111)
    d.card("AELINK", 1, "ELEVR", "ELEV", 1.0, "RUD", 1.0)
    d.card("AELINK", 2, "ELEVL", "ELEV", 1.0, "RUD", -1.0)

    # 1g 트림 3케이스 (V = 40/60/80 m/s EAS, 해수면)
    for tid, V in ((1, 40.0), (2, 60.0), (3, 80.0)):
        mach = V / 340.3
        q_mpa = 0.5 * 1.225 * V * V * 1e-6
        d.card("TRIM", tid, mach, q_mpa,
               "ROLL", 0.0, "YAW", 0.0, "URDD2", 0.0,
               "URDD4", 0.0, "URDD6", 0.0, "SIDES", 0.0, "RUD", 0.0)


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------
def _properties(d: Deck) -> None:
    # 스미어드 등가 물성 (MAT1): CFRP 준등방/UD, Al 7050 — 값은 문헌 대표치
    d.card("MAT1", MAT_CFRP_QI, 52000.0, "", 0.31, 1.55e-9)
    d.card("MAT1", MAT_AL7050, 71000.0, "", 0.33, 2.83e-9)
    d.card("MAT1", MAT_CFRP_UD, 105000.0, "", 0.30, 1.57e-9)
    d.card("MAT1", MAT_CFRP_BL, 80000.0, "", 0.30, 1.60e-9)
    # 쉘 게이지는 막강성 t·E를 기존 Al 설계와 등가로 재산정
    # (예: 동체 0.8·73.1 ≈ 1.1·52, 붐 1.4·73.1 ≈ 2.0·52)
    for pid, t in ((PSH_FUS, 1.1), (PSH_WSKIN, 1.8), (PSH_WEB, 3.0),
                   (PSH_RIB, 1.4), (PSH_VT, 1.4), (PSH_VTWEB, 2.0),
                   (PSH_BOOM, 2.0)):
        d.card("PSHELL", pid, MAT_CFRP_QI, t)
    # 스파캡/론저론은 0° 지배 CFRP (EA 등가: 500·73.1k ≈ 350·105k),
    # 프레임/마스트/스트럿은 기계가공 Al 7050 유지
    for pid, mat, A, I in (
            (PB_FRAME, MAT_AL7050, 140.0, 1.2e5),
            (PB_LONG, MAT_CFRP_UD, 110.0, 3.0e4),
            (PB_CAP, MAT_CFRP_UD, 350.0, 6.0e4),
            (PB_MAST, MAT_AL7050, 730.0, 5.4e5),
            (PB_STRUT, MAT_AL7050, 730.0, 5.4e5),
            (PB_GEAR, MAT_AL7050, 1500.0, 2.0e6)):
        d.card("PBAR", pid, mat, A, I, I, 2 * I)
    # 블레이드: 1차 플랩 ~35 Hz 목표 (EI_flap = f²·(2π/1.875²)²·m̄·L⁴)
    d.card("PBAR", PB_BLADE_L, MAT_CFRP_BL, 300.0, 7.0e3, 3.0e4, 1.5e4)
    d.card("PBAR", PB_BLADE_P, MAT_CFRP_BL, 250.0, 5.0e3, 2.0e4, 1.0e4)


def _spc_ref(d: Deck, fus: dict) -> None:
    """3-2-1 기준 지지 (6 DOF, 정정) — inertia relief 게이트 통과."""
    xs, ids = fus["xs"], fus["ids"]
    i1 = int(np.argmin([abs(x - 4250) for x in xs]))
    i2 = int(np.argmin([abs(x - 4750) for x in xs]))
    n1 = ids[(i1, 18)]   # 하면 123
    n2 = ids[(i2, 18)]   # 하면 23
    n3 = ids[(i1, 0)]    # 측면 3
    d.card("SPC1", 2, "123", n1)
    d.card("SPC1", 2, "23", n2)
    d.card("SPC1", 2, "3", n3)


def generate(out_path: str, ballast: Optional[Tuple[float, float]] = None,
             rotor_panels: bool = False, pusher: bool = True,
             tilt_nacelle_t: float = 0.0, title: Optional[str] = None,
             ballast_x: Tuple[float, float] = (1000.0, 7500.0),
             nose_trim_t: float = 0.0) -> Deck:
    """ILC-8 덱 생성. ballast=(m_fwd_t, m_aft_t)는 2차 패스에서 주입.

    rotor_panels=True는 형상 확인용 블레이드 CAERO를 추가한다 (해석 금지).
    """
    d = Deck()
    cid = [800000]
    _properties(d)
    fus = build_fuselage(d)
    wings = {1: build_wing(d, 1, fus), -1: build_wing(d, -1, fus)}
    vtails = {1: build_vtail(d, 1, fus), -1: build_vtail(d, -1, fus)}
    hub_ids = build_booms(d, wings)
    pusher_nid = build_masses(d, fus, wings, hub_ids, cid,
                              pusher=pusher,
                              tilt_nacelle_t=tilt_nacelle_t,
                              nose_trim_t=nose_trim_t)
    build_gear(d, fus, cid)
    build_rotor_blades(d, hub_ids, pusher_nid)
    build_aero(d, wings, vtails)
    if rotor_panels:
        d.raw("$ DISPLAY DECK - rotor CAERO panels (IGID 99), do not solve")
        build_rotor_panels(d, hub_ids)
    _spc_ref(d, fus)
    d.card("EIGRL", 100, "", "", 12)
    if ballast:
        xs, bottom = fus["xs"], fus["bottom"]
        for m, xg in zip(ballast, ballast_x):
            if m > 1e-6:
                i = int(np.argmin([abs(x - xg) for x in xs]))
                d.conm2(bottom[i], m, cid)

    header = [
        "SOL 144",
        "CEND",
        f"TITLE    = {title or 'ILC-8 NASA L+C CLASS - FREE TRIM'}",
        "ECHO     = NONE",
        "SPC      = 2",
    ]
    for tid, V in ((1, 40.0), (2, 60.0), (3, 80.0)):
        header += [f"SUBCASE {tid}",
                   f"  LABEL    =1G LEVEL FLIGHT V={V:.0f} M/S EAS",
                   f"  TRIM     = {tid}"]
    header.append("BEGIN BULK")

    Path(out_path).write_text(
        "\n".join(header + d.lines + ["ENDDATA"]) + "\n")
    return d


def build_ilc8(out_dir: str) -> str:
    """2-패스 생성: 1차 → 질량/CG 측정 → 밸러스트 해결 → 최종 덱."""
    from ..bdf.parser import parse_bdf
    from ..loads_analysis.trim_loads import compute_node_masses

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bdf = out / "ilc8.bdf"

    generate(str(bdf))
    m = parse_bdf(str(bdf))
    nm = compute_node_masses(m)
    m_tot = sum(nm.values())
    cg_x = sum(mass * m.nodes[n].xyz_global[0]
               for n, mass in nm.items() if n in m.nodes) / m_tot
    dm = MTOW_T - m_tot
    x_a, x_b = 1000.0, 7500.0
    # m_a + m_b = dm ;  m_a x_a + m_b x_b = MTOW*CGT - m_tot*cg_x
    rhs = MTOW_T * CG_X_TARGET - m_tot * cg_x
    m_b = (rhs - dm * x_a) / (x_b - x_a)
    m_a = dm - m_b
    logger.info("ILC-8 pass1: m=%.4f t, cg_x=%.1f -> ballast fwd=%.4f t, "
                "aft=%.4f t", m_tot, cg_x, m_a, m_b)
    if m_a < -1e-4 or m_b < -1e-4:
        logger.warning("ILC-8 ballast negative (m_a=%.4f, m_b=%.4f) — "
                       "고정 질량 배분 조정 필요", m_a, m_b)
    final_ballast = (max(m_a, 0.0), max(m_b, 0.0))
    generate(str(bdf), ballast=final_ballast)
    # 형상 확인용 덱 (블레이드 CAERO 포함) — 해석에는 ilc8.bdf 사용
    generate(str(out / "ilc8_display.bdf"), ballast=final_ballast,
             rotor_panels=True)
    return str(bdf)


# ---------------------------------------------------------------------------
# VTOL 로터 구성 (허브 노드 = 생성 덱과 일치)
# ---------------------------------------------------------------------------
def make_ilc8_vtol_config():
    """ILC-8 로터 8기(리프트) + 푸셔 VTOLConfig — 논문 제원(R=0.75 m)."""
    from ..rotor.airfoil import RotorAirfoil
    from ..rotor.blade import BladeDef
    from ..rotor.rotor_config import (
        RotationDir, RotorDef, RotorType, VTOLConfig,
    )

    # 코드/회전수는 호버 블레이드 하중 CT/σ≈0.13이 되도록 사이징 —
    # (0.065 m/2100 rpm은 CT/σ≈0.28로 실속 포화, 목표 추력 미달)
    lift_blade = BladeDef(radius=0.75, root_cutout=0.15, n_elements=20,
                          mean_chord=0.11,
                          twist_root=math.radians(12.0),
                          twist_tip=math.radians(4.0),
                          airfoil=RotorAirfoil.naca0012())
    pusher_blade = BladeDef(radius=0.65, root_cutout=0.15, n_elements=20,
                            mean_chord=0.055,
                            twist_root=math.radians(16.0),
                            twist_tip=math.radians(6.0),
                            airfoil=RotorAirfoil.naca0012())
    rotors = []
    rid = 0
    for x, row in ((ROTOR_X_F, "F"), (ROTOR_X_A, "A")):
        for bi, y in enumerate(BOOM_YS):
            rid += 1
            rotors.append(RotorDef(
                rotor_id=rid,
                label=f"Lift Rotor {row}{bi + 1} (y={y / 1000:+.0f}m)",
                rotor_type=RotorType.LIFT,
                hub_position=np.array([x, y, HUB_Z]),
                shaft_axis=np.array([0.0, 0.0, 1.0]),
                blade=lift_blade, n_blades=4,
                rotation_dir=(RotationDir.CW if (rid % 2) else RotationDir.CCW),
                rpm_hover=2400.0, rpm_cruise=0.0,
                hub_node_id=(HUB_BASE + (1 if row == "F" else 5) + bi),
                mass_kg=25.0,
            ))
    rotors.append(RotorDef(
        rotor_id=9, label="Pusher", rotor_type=RotorType.CRUISE,
        hub_position=np.array([FUS_L + 150.0, 0.0, 0.0]),
        shaft_axis=np.array([-1.0, 0.0, 0.0]),
        blade=pusher_blade, n_blades=3, rotation_dir=RotationDir.CW,
        rpm_hover=0.0, rpm_cruise=2400.0,
        hub_node_id=HUB_BASE + 101, mass_kg=45.0,
    ))
    return VTOLConfig(config_type="ilc8_lift_cruise", rotors=rotors)

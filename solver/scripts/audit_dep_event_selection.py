# 논문 3 위험도 선정의 전수 감사 — ILC-8 200개 고장 사건 FE 결과 스윕 vs top-40
"""Exhaustive audit of the risk-based failure-case selection (Paper 3).

The selection ranks events by an analytical consequence proxy
(|delta-M| about the CG). This audit evaluates EVERY raw event of the
ILC-8 taxonomy (8 rotors x 5 phases x 5 modes = 200) with a strictly
stronger consequence model -- full-airframe FE section loads (wing
root Mx/Vy left/right, peak boom Mx) from the same nodal-load path
used by the certification pipeline -- and checks whether any event
OUTSIDE the risk-selected top-40 governs any structural envelope.

Scope note: the sweep is quasi-static (failure thrust state + 1-g
inertia with inertia-relief closure); transient amplification (DAF)
is a separate multiplier applied to whichever cases are analysed in
detail and does not affect the ranking audit.

Usage:  python scripts/audit_dep_event_selection.py
Output: prints the audit; writes dep_audit_results.json here.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)

from nastaero.bdf.parser import parse_bdf                      # noqa: E402
from nastaero.loads_analysis.certification.aircraft_config import (  # noqa: E402
    AircraftConfig,
)
from nastaero.loads_analysis.certification.vtol_transient_loads import (  # noqa: E402
    VTOLTransientLoadsRunner, _G,
)
from nastaero.models.ilc8 import make_ilc8_vtol_config         # noqa: E402

# ── 논문 3(ch6) 분류 상수 — reproduce_ch6_dep_failure.py와 동일 ──
PHASES = ["HV", "TO", "TR", "CR", "LD"]
MODES = ["M1", "M2", "M3", "M4", "M5"]
P_PHASE = {"HV": 0.10, "TO": 0.025, "TR": 0.025, "CR": 0.75, "LD": 0.10}
P_MODE = {"M1": 0.50, "M2": 0.05, "M3": 0.05, "M4": 0.20, "M5": 0.20}
PHASE_THRUST_FACTOR = {"HV": 1.0, "TO": 1.0, "TR": 0.5, "CR": 0.0, "LD": 1.0}
RUNAWAY_FACTOR = 1.5
BETA = 100.0
TOP_N = 40


def mirror_idx(rotors, i):
    """같은 |y|의 반대편 로터 인덱스."""
    yi = rotors[i].hub_position[1]
    best, dbest = i, 1e18
    for j, r in enumerate(rotors):
        if j == i:
            continue
        d = abs(r.hub_position[1] + yi)
        if d < dbest:
            best, dbest = j, d
    return best


def sameside_idx(rotors, i):
    """같은 쪽에서 가장 가까운 이웃 로터 인덱스."""
    yi = rotors[i].hub_position[1]
    best, dbest = i, 1e18
    for j, r in enumerate(rotors):
        yj = r.hub_position[1]
        if j == i or yi * yj <= 0:
            continue
        d = abs(yj - yi)
        if d < dbest:
            best, dbest = j, d
    return best


def event_thrusts(rotors, T_nom_hover, phase, mode, i):
    """사건의 로터별 추력 배열 (고장 반영). 반환 (thrusts, arm_m, dT)."""
    n = len(rotors)
    f = PHASE_THRUST_FACTOR[phase]
    T_nom = f * T_nom_hover
    T = np.full(n, T_nom)
    y_m = abs(rotors[i].hub_position[1]) / 1000.0
    if mode == "M1":                     # 단일 정지
        T[i] = 0.0
        arm, dT = y_m, -T_nom
    elif mode == "M2":                   # 이중 대칭 (거울쌍 동시 정지)
        j = mirror_idx(rotors, i)
        T[i] = 0.0
        T[j] = 0.0
        arm, dT = 0.0, -T_nom            # 롤 팔 상쇄 (프록시 관례)
    elif mode == "M3":                   # 이중 비대칭 (같은 쪽 2기)
        j = sameside_idx(rotors, i)
        T[i] = 0.0
        T[j] = 0.0
        arm, dT = y_m, -2 * T_nom
    elif mode == "M4":                   # 스턱 (직전 지령 유지 상한)
        T[i] = T_nom + T_nom
        arm, dT = y_m, +T_nom
    elif mode == "M5":                   # 폭주 (위상 무관 최대 추력)
        T[i] = RUNAWAY_FACTOR * T_nom_hover
        arm, dT = y_m, RUNAWAY_FACTOR * T_nom_hover - T_nom
    return T, arm, dT


def main() -> None:
    model = parse_bdf(os.path.join(SOLVER, "tests/validation/ILC8/ilc8.bdf"))
    with open(os.path.join(SOLVER,
                           "tests/validation/ILC8/ilc8_cert_config.yaml")) as f:
        acfg = AircraftConfig.from_dict(yaml.safe_load(f))
    vcfg = make_ilc8_vtol_config()
    runner = VTOLTransientLoadsRunner(model, vcfg, acfg)
    rotors = list(vcfg.hover_rotors)
    n = len(rotors)
    T_hover = runner._weight_N / n

    # 사건별 절점하중 → 단면하중 (돌풍 이벤트와 동일한 하중 경로)
    def section_loads(T_arr):
        rotor_nodal = {}
        total_Fz = 0.0
        for r, T in zip(rotors, T_arr):
            if T <= 1e-9:
                continue
            shaft = r.effective_shaft_axis
            fvec = np.zeros(6)
            fvec[:3] = T * shaft
            nid = r.hub_node_id
            rotor_nodal[nid] = rotor_nodal.get(nid, np.zeros(6)) + fvec
            total_Fz += fvec[2]
        nz = total_Fz / runner._weight_N if runner._weight_N > 0 else 0.0
        forces = {k: v.copy() for k, v in rotor_nodal.items()}
        for nid, m_node in runner._node_masses.items():
            if runner.bdf_model.nodes.get(nid) is None:
                continue
            fvec = np.zeros(6)
            fvec[2] = -m_node * runner._mass_to_kg * nz * _G
            forces[nid] = forces.get(nid, np.zeros(6)) + fvec
        Vy_L, Mx_L, Vy_R, Mx_R = runner._wing_root_section_loads(forces)
        boom = runner._boom_root_section_loads(forces)
        return np.array([Mx_L, Mx_R, Vy_L, Vy_R, boom])

    QTY = ["wing_Mx_L", "wing_Mx_R", "wing_Vy_L", "wing_Vy_R", "boom_Mx"]

    # 단계별 무고장 기준하중 (결과 = |사건 - 기준|)
    base = {p: section_loads(np.full(n, PHASE_THRUST_FACTOR[p] * T_hover))
            for p in PHASES}

    events = []
    eid = 0
    for i in range(n):
        for phase in PHASES:
            for mode in MODES:
                T_arr, arm, dT = event_thrusts(rotors, T_hover, phase,
                                               mode, i)
                loads = section_loads(T_arr)
                cons = np.abs(loads - base[phase])
                RI = ((phase == "TR" and mode in ("M3", "M4", "M5"))
                      or (phase == "HV" and mode == "M3"))
                dM = abs(arm * dT) / 1000.0          # 프록시 (kN·m)
                P = (1.0 / n) * P_PHASE[phase] * P_MODE[mode]
                events.append({
                    "eid": eid, "rotor": rotors[i].label, "phase": phase,
                    "mode": mode, "P": P, "proxy_dM_kNm": dM, "RI": RI,
                    "risk": P * (dM + (BETA if RI else 0.0)),
                    "fe": cons.tolist(),
                })
                eid += 1
    print(f"events: {len(events)} (rotors {n} x phases 5 x modes 5)")

    ranked = sorted(events, key=lambda e: -e["risk"])
    top = ranked[:TOP_N]
    top_ids = {e["eid"] for e in top}
    cov = sum(e["risk"] for e in top) / sum(e["risk"] for e in events)
    print(f"top-{TOP_N} risk coverage: {cov*100:.1f}%")

    # ── 감사: 각 수량의 지배(최대 결과) 사건이 top-40 안인가 ──
    misses = []
    print(f"\n{'quantity':10s} {'governing event':34s} {'in top40':>8s} "
          f"{'risk rank':>9s}")
    rank_of = {e["eid"]: k + 1 for k, e in enumerate(ranked)}
    for qi, q in enumerate(QTY):
        order = sorted(events, key=lambda e: -e["fe"][qi])
        gov = order[0]
        ok = gov["eid"] in top_ids
        tag = f"{gov['rotor']}/{gov['phase']}/{gov['mode']}"
        print(f"{q:10s} {tag:34s} {str(ok):>8s} {rank_of[gov['eid']]:>9d}")
        if not ok:
            misses.append((q, gov))
        # 상위 5위까지의 이탈도 기록
        for e in order[1:5]:
            if e["eid"] not in top_ids and e["fe"][qi] > 0.95 * gov["fe"][qi]:
                misses.append((f"{q}(top5,>95%)", e))

    print(f"\noutside-top40 governing events: {len(misses)}")
    for q, e in misses:
        print(f"  ! {q}: {e['rotor']}/{e['phase']}/{e['mode']} "
              f"risk rank {rank_of[e['eid']]}")

    with open(os.path.join(HERE, "dep_audit_results.json"), "w") as f:
        json.dump({"n_events": len(events), "top_n": TOP_N,
                   "coverage_pct": round(cov * 100, 1),
                   "misses": [{"quantity": q, **{k: e[k] for k in
                               ("rotor", "phase", "mode", "eid")},
                               "risk_rank": rank_of[e["eid"]]}
                              for q, e in misses],
                   "governing": {QTY[qi]: sorted(
                       events, key=lambda e: -e["fe"][qi])[0]["eid"]
                       for qi in range(len(QTY))}}, f,
                  ensure_ascii=False, indent=1)
    print("saved: dep_audit_results.json")


if __name__ == "__main__":
    main()

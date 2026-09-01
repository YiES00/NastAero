# 3D 헐 전용 ≥5% 우승 케이스의 V/M/T 상세 — 2D 선정 세트 축극값 대비 위치 정량화
"""Print the (V, M, T) numbers of the winning >=5 % 3-D-only rotor
patterns at their worst station, against the 2-D-selected set's axis
extremes and hull support at that station.

Usage:  python scripts/hull3d_winner_detail.py [MTOW|LIGHT_AFT]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)

from ascent_load.config import setup_logging                      # noqa: E402


def main() -> None:
    setup_logging("ERROR")
    os.chdir(SOLVER)
    ctx = sys.argv[1] if len(sys.argv) > 1 else "MTOW"
    dm, cgs = (0.0, 0.0) if ctx == "MTOW" else (-400.0, 150.0)

    with open(os.path.join(HERE,
              f"hull3d_combo_results_{ctx}.json")) as f:
        res = json.load(f)
    winners = [r for r in res["validated"]
               if r["status"] == "3D-only" and r["validated_pct"] >= 1.0]
    if not winners:
        print("no 3D-only winners in results")
        return

    from ascent_load.bdf.parser import parse_bdf
    from ascent_load.loads_analysis.certification.batch_runner import (
        CaseResult,
    )
    from ascent_load.loads_analysis.certification.vmt_bridge import (
        compute_vmt_for_batch,
    )
    from ascent_load.loads_analysis.trim_loads import compute_node_masses
    from ascent_load.models.ilc8 import make_ilc8_vtol_config
    from compare_hull_selection import run_selection
    from hull3d_severity_search import (
        BatchResult, ILC8, adjust_fuselage_masses, build_components,
        run_variant,
    )
    from hull3d_combo_search import hover_nodal_forces

    batch, vmt, meta = run_variant(ctx, dm, cgs, [0.0], combos=True)
    model = parse_bdf(os.path.join(ILC8, "ilc8.bdf"))
    mass_kg, cg_x = adjust_fuselage_masses(model, dm, cgs)
    components = build_components(model)
    _, dc2 = run_selection(batch, vmt, "2d")
    s2 = {d.case_id for d in dc2}
    labels = {c.case_id: c.label for c in batch.case_results}

    vtol_config = make_ilc8_vtol_config()
    hover_case = next(c for c in meta.values()
                      if getattr(c, "rotor_forces", None)
                      and "Hover 1.0g" in (c.trim_condition.label
                                           if c.trim_condition else ""))
    rotors = list(vtol_config.hover_rotors)
    hub_vecs = [(r.hub_node_id,
                 np.array(hover_case.rotor_forces[r.hub_node_id], float))
                for r in rotors]
    node_masses = compute_node_masses(model)
    total_mass = sum(node_masses.values())
    cg = np.zeros(3)
    for nid, m in node_masses.items():
        cg += m * model.nodes[nid].xyz_global
    cg /= total_mass
    g = 9806.65

    for w in winners:
        pat = np.array(w["pattern"])
        forces, nz = hover_nodal_forces(model, hub_vecs, pat,
                                        node_masses, cg, g)
        mini = BatchResult()
        cr = CaseResult(case_id=999999, category="vtol_combined_cmd",
                        converged=True, nodal_forces=forces, nz=nz,
                        label="winner")
        mini.case_results = [cr]
        mini.completed_ids = {999999}
        vmt_w = compute_vmt_for_batch(model, mini, components=components,
                                      fuselage_cg_x=cg_x)[999999]
        comp = w["worst_component"]
        st = np.asarray(vmt_w[comp]["stations"], float)
        si = int(np.argmin(np.abs(st - w["worst_station"])))
        pw = np.array([vmt_w[comp]["shear"][si],
                       vmt_w[comp]["bending"][si],
                       vmt_w[comp]["torsion"][si]])

        sel = [c for c in vmt if comp in vmt[c] and c in s2]
        S = np.array([[vmt[c][comp]["shear"][si],
                       vmt[c][comp]["bending"][si],
                       vmt[c][comp]["torsion"][si]] for c in sel])
        allc = [c for c in vmt if comp in vmt[c]]
        A = np.array([[vmt[c][comp]["shear"][si],
                       vmt[c][comp]["bending"][si],
                       vmt[c][comp]["torsion"][si]] for c in allc])
        span = np.ptp(A, axis=0)

        print(f"\n=== {w['class']}  exc {w['validated_pct']}%  "
              f"{comp} y={w['worst_station']:.0f} mm ===")
        print(f"  pattern l = {pat.tolist()}   nz = {w['nz']}")
        names = ["V(전단 N)", "M(굽힘 N·mm)", "T(비틀림 N·mm)"]
        for a in range(3):
            hi, lo = S[:, a].max(), S[:, a].min()
            frac = ((pw[a] - lo) / (hi - lo) * 100
                    if hi > lo else 0.0)
            print(f"  {names[a]:14s} case={pw[a]:14,.0f}  "
                  f"2D세트 [{lo:14,.0f}, {hi:14,.0f}]  "
                  f"축내 위치 {frac:5.1f}%")
        # 어느 2D 케이스와도 짝지어 지배되지 않음을 표시하는 최근접
        d = np.linalg.norm((S - pw) / span, axis=1)
        for k in np.argsort(d)[:3]:
            print(f"    근접 2D 선정 케이스: {labels[sel[k]][:56]}  "
                  f"(정규화 거리 {d[k]:.3f})")


if __name__ == "__main__":
    main()

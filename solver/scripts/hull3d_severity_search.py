# 3D 헐 전용 케이스가 2D 세트 대비 5% 이상 가혹해지는 조건 탐색 — 중량/CG/고도 확장 ILC-8 스윕
"""Search for load conditions where a 3-D-hull-only case exceeds the
2-D-selected set by >= 5 % (directional exceedance).

The baseline ILC-8 94-case matrix showed <= 0.82 % exceedance for
3-D-only additions. Hypothesis: that smallness comes from V/M/T being
nearly rank-1 correlated when a single parameter (q*nz at one weight)
dominates. Decorrelating the mechanisms should widen the 3-D hull:

  * light gross weight with unchanged wing/boom-mounted mass (motors,
    booms: 688 kg fixed) -> inertia-relief fraction grows, M/V ratio
    shifts
  * fwd/aft CG -> tail-load share and pitch trim change M vs T mix
  * altitude -> Pratt gust mass-ratio and Mach change gust nz vs q mix
  * rotor jam / OEI / transition (already in matrix) -> torsion axis

Method: run the full ILC-8 cert pipeline once per weight/CG variant
(each is a self-consistent aircraft: CONM2 masses adjusted, rotor
thrust targets and nz_max/gust recomputed from actual FE mass), merge
all cases into one campaign set, then compare 2-D vs 3-D selection and
score every non-selected case with the directional-exceedance metric
of compare_hull_selection.py (vs the 2-D set, and vs the 3-D set).

Usage:  python scripts/hull3d_severity_search.py
Output: prints the search; writes hull3d_severity_results.json here.
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
ILC8 = os.path.join(SOLVER, "tests/validation/ILC8")
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)

from ascent_load.bdf.parser import parse_bdf                      # noqa: E402
from ascent_load.config import setup_logging                      # noqa: E402
from ascent_load.loads_analysis.certification.aircraft_config import (  # noqa: E402
    AircraftConfig, SpeedSchedule, WeightCGCondition,
    ControlSurfaceLimits, LandingGearConfig,
)
from ascent_load.loads_analysis.certification.load_case_matrix import (  # noqa: E402
    LoadCaseMatrix,
)
from ascent_load.loads_analysis.certification.vtol_load_case_matrix import (  # noqa: E402
    VTOLLoadCaseMatrix,
)
from ascent_load.loads_analysis.certification.batch_runner import (  # noqa: E402
    BatchResult,
)
from ascent_load.loads_analysis.certification.vtol_batch_runner import (  # noqa: E402
    VTOLBatchRunner,
)
from ascent_load.loads_analysis.certification.vmt_bridge import (  # noqa: E402
    compute_vmt_for_batch,
)
from ascent_load.loads_analysis.component_id import (             # noqa: E402
    identify_components_manual,
)
from ascent_load.loads_analysis.trim_loads import compute_node_masses  # noqa: E402
from ascent_load.models.ilc8 import make_ilc8_vtol_config         # noqa: E402

from compare_hull_selection import exceedance, run_selection   # noqa: E402

G = 9.80665

# (변형명, 동체 CONM2 질량 변화 kg, CG 이동 mm, 해석 고도 m, 조합케이스)
VARIANTS = [
    ("BASE",      0.0,    0.0, [0.0, 3000.0], False),
    ("FWD_CG",    0.0, -150.0, [0.0], False),
    ("AFT_CG",    0.0, +150.0, [0.0], False),
    ("LIGHT",  -400.0,    0.0, [0.0], False),
    ("LIGHT_AFT", -400.0, +150.0, [0.0], False),
    # 고장×돌풍 조합 (SC-VTOL 고장 후 지속비행 중 돌풍 조우):
    ("COMBO",     0.0,    0.0, [0.0], True),
    ("COMBO_LA", -400.0, +150.0, [0.0], True),
]


def add_combo_cases(vtol_matrix, vtol_config, config):
    """고장(잼/OEI) × 호버 돌풍(nz=1±0.3) 조합 케이스를 매트릭스에 추가.

    SC-VTOL.2215 고장 상태의 비행하중 — 고장 후 지속비행 중 돌풍
    조우는 확률상 배제 불가(디버전 구간 노출)이므로 인증상 정당한
    조합. 로터별 모멘트 암이 달라 (V,M,T) 등급화된 조합을 만든다.
    """
    from ascent_load.loads_analysis.certification.vtol_conditions import (
        VTOLCondition, VTOLFlightPhase,
    )
    wc = config.weight_cg_conditions[0]
    n_added = 0
    for r in vtol_config.hover_rotors:
        for nz in (0.7, 1.3):
            cond = VTOLCondition(
                label=f"Jam R{r.rotor_id}+gust nz={nz}",
                phase=VTOLFlightPhase.ROTOR_JAM,
                V_eas=0.0, nz=nz, altitude_m=0.0,
                thrust_fraction=nz, failed_rotor_id=r.rotor_id,
                far_section="SC-VTOL.2150+2135")
            forces = vtol_matrix._compute_rotor_forces_hover(cond, wc)
            case = vtol_matrix._condition_to_cert_case(cond, wc, forces)
            vtol_matrix.cases.append(case)
            n_added += 1
        cond = VTOLCondition(
            label=f"OEI R{r.rotor_id}+maneuver nz=1.15",
            phase=VTOLFlightPhase.OEI,
            V_eas=0.0, nz=1.15, altitude_m=0.0,
            thrust_fraction=1.15, failed_rotor_id=r.rotor_id,
            far_section="SC-VTOL.2140+2135")
        forces = vtol_matrix._compute_rotor_forces_hover(cond, wc)
        case = vtol_matrix._condition_to_cert_case(cond, wc, forces)
        vtol_matrix.cases.append(case)
        n_added += 1

    # 천이 중 잼 — 공력 굽힘(M)+전단(V)+잼 비틀림(T)이 동시에 걸리는
    # 유일한 비행 상태. 기존 천이 조건을 재생성해 고장 로터만 지정.
    import dataclasses

    from ascent_load.loads_analysis.certification.vtol_conditions import (
        generate_transition_conditions,
    )
    trans = generate_transition_conditions(
        vtol_config.v_mca, vtol_config.v_transition_end, [0.0],
        wing_area_m2=config.wing_area_m2, CL_transition=1.0,
        weight_N=wc.weight_N)
    for cond in trans:
        if cond.thrust_fraction <= 0.05:      # 날개 전담 구간은 잼 무의미
            continue
        for r in vtol_config.hover_rotors:
            if not r.can_fail:
                continue
            c2 = dataclasses.replace(
                cond,
                label=f"{cond.label} + Jam R{r.rotor_id}",
                failed_rotor_id=r.rotor_id,
                far_section="SC-VTOL.2150+2135")
            forces = vtol_matrix._compute_rotor_forces_transition(c2, wc)
            case = vtol_matrix._condition_to_cert_case(c2, wc, forces)
            vtol_matrix.cases.append(case)
            n_added += 1
    return n_added


def model_mass_cg(model):
    """전체 모델 (질량 kg, CG_x mm) — compute_node_masses 럼핑 기준."""
    nm = compute_node_masses(model)          # tonnes
    m = sum(nm.values())
    sx = sum(mi * model.nodes[n].xyz_global[0] for n, mi in nm.items())
    return m * 1000.0, sx / m


def adjust_fuselage_masses(model, dm_kg: float, cg_shift_mm: float):
    """동체 CONM2(절점 100k대)에 m_i' = m_i(a + b x_i) 재분배를 적용해
    전체 질량 dm_kg 변화 + CG cg_shift_mm 이동을 달성. 음질량은 0으로
    클립 후 1회 재보정."""
    m0_kg, cg0 = model_mass_cg(model)
    m_t = (m0_kg + dm_kg) / 1000.0                       # target, tonnes
    s_t = m_t * (cg0 + cg_shift_mm)                      # target moment

    fus = [(eid, c) for eid, c in model.conm2s.items()
           if 100000 <= c.node_id < 300000]
    xs = np.array([model.nodes[c.node_id].xyz_global[0] for _, c in fus])
    ms = np.array([c.mass for _, c in fus])
    nm = compute_node_masses(model)
    m0 = sum(nm.values())
    s0 = sum(mi * model.nodes[n].xyz_global[0] for n, mi in nm.items())
    # 동체 그룹이 담당할 질량/모멘트
    mf_t = m_t - (m0 - ms.sum())
    sf_t = s_t - (s0 - (ms * xs).sum())

    for _ in range(3):                                    # 클립 반복 보정
        A = np.array([[ms.sum(), (ms * xs).sum()],
                      [(ms * xs).sum(), (ms * xs * xs).sum()]])
        a, b = np.linalg.solve(A, [mf_t, sf_t])
        new = ms * (a + b * xs)
        if (new >= 0).all():
            break
        keep = new > 0
        new = np.clip(new, 0.0, None)
        ms = np.where(keep, ms, 0.0)                      # 클립된 것 고정
    for (eid, c), mi in zip(fus, new):
        c.mass = float(mi)
    return model_mass_cg(model)


def build_components(model):
    """드라이버 run_ilc8_cert_analysis.py [8]과 동일한 수동 컴포넌트."""
    def _nids(lo, hi, extra=()):
        ids = [n for n in model.nodes if lo <= n <= hi]
        return ids + [n for n in extra if n in model.nodes]

    WING = dict(span_axis=1, shear_axis=2, bending_axis=0, torsion_axis=1)
    hubs_r = (990103, 990104, 990107, 990108)
    hubs_l = (990101, 990102, 990105, 990106)
    return identify_components_manual(model, [
        dict(name="Right Wing", integration_sign=1.0, color="blue",
             node_ids=_nids(400000, 499999)
             + _nids(730000, 749999, hubs_r), **WING),
        dict(name="Left Wing", integration_sign=-1.0, color="dodgerblue",
             node_ids=_nids(300000, 399999)
             + _nids(710000, 729999, hubs_l), **WING),
        dict(name="Right V-Tail", integration_sign=1.0, color="red",
             node_ids=_nids(600000, 699999), **WING),
        dict(name="Left V-Tail", integration_sign=-1.0, color="salmon",
             node_ids=_nids(500000, 599999), **WING),
        dict(name="Fuselage", integration_sign=-1.0, color="gray",
             node_ids=_nids(100000, 299999, (990201,)),
             span_axis=0, shear_axis=2, bending_axis=1, torsion_axis=0),
    ])


def run_variant(name, dm_kg, cg_shift, alts, combos=False):
    t0 = time.time()
    model = parse_bdf(os.path.join(ILC8, "ilc8.bdf"))
    mass_kg, cg_x = adjust_fuselage_masses(model, dm_kg, cg_shift)
    weight_N = mass_kg * G
    print(f"\n=== {name}: mass {mass_kg:.1f} kg, CG {cg_x:.0f} mm, "
          f"alts {alts} ===", flush=True)

    from ascent_load.aero.dlm import compute_rigid_clalpha
    clalpha = compute_rigid_clalpha(model, mach=80.0 / 340.3,
                                    ref_area=16.2e6)
    config = AircraftConfig(
        speeds=SpeedSchedule(VS1=36.0, VA=71.0, VB=0.0, VC=80.0,
                             VD=100.0, VF=50.0),
        weight_cg_conditions=[
            WeightCGCondition(label=name, weight_N=weight_N, cg_x=cg_x),
        ],
        altitudes_m=list(alts),
        wing_area_m2=16.2, CLalpha=clalpha, mean_chord_m=1.35,
        ctrl_limits=ControlSurfaceLimits(
            aileron_max_deg=20.0, rudder_max_deg=25.0,
            elevator_max_deg=25.0),
        landing_gear=LandingGearConfig(
            main_gear_node_ids=[101720, 101718],
            nose_gear_node_ids=[100419],
            main_gear_x=4700.0, nose_gear_x=1500.0,
            strut_efficiency=0.7, stroke=0.25, sink_rate_fps=10.0,
            tire_deflection=0.03, tire_efficiency=0.3),
        gust_Ude_VC_fps=50.0, gust_Ude_VD_fps=25.0,
    )
    vtol_config = make_ilc8_vtol_config()
    config.vtol_config = vtol_config

    conv_matrix = LoadCaseMatrix(config)
    conv_matrix.generate_all()
    vtol_matrix = VTOLLoadCaseMatrix(vtol_config, config)
    vtol_cases = vtol_matrix.generate_all()
    if combos:
        n_combo = add_combo_cases(vtol_matrix, vtol_config, config)
        vtol_cases = list(vtol_matrix.cases)
        print(f"  combo cases added: {n_combo}", flush=True)
    conv_only = SimpleNamespace(
        flight_cases=list(conv_matrix.flight_cases),
        landing_cases=list(conv_matrix.landing_cases),
        dynamic_cases=[], config=config)
    n_cases = (len(conv_matrix.flight_cases)
               + len(conv_matrix.landing_cases) + len(vtol_cases))
    print(f"  cases: flight {len(conv_matrix.flight_cases)} + landing "
          f"{len(conv_matrix.landing_cases)} + vtol {len(vtol_cases)} "
          f"= {n_cases}", flush=True)

    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    runner = VTOLBatchRunner(conv_only, vtol_matrix, bdf_model=model,
                             vtol_config=vtol_config, n_workers=n_cpus,
                             include_transient=False)
    batch = runner.run()
    print(f"  solved {batch.n_converged}/{batch.n_total} "
          f"({time.time()-t0:.0f}s)", flush=True)

    components = build_components(model)
    vmt = compute_vmt_for_batch(model, batch, components=components,
                                fuselage_cg_x=cg_x)
    meta = {}
    for src in (conv_matrix.flight_cases, conv_matrix.landing_cases,
                vtol_cases):
        for c in src:
            meta[c.case_id] = c
    return batch, vmt, meta


def exceedance_all(vmt_data, base_ids):
    """모든 케이스의 base 세트 헐 대비 방향 초과율 — (comp, station)별
    헐을 한 번만 만들어 전 케이스를 일괄 평가 (compare_hull_selection.
    exceedance와 동일 정의의 벡터화 버전)."""
    from scipy.spatial import ConvexHull

    worst = {}
    comps = set()
    for cd in vmt_data.values():
        comps.update(cd.keys())
    for comp in comps:
        cids = [c for c in vmt_data if comp in vmt_data[c]]
        n_sta = len(vmt_data[cids[0]][comp]["stations"])
        P = np.empty((len(cids), n_sta, 3))
        for k, cid in enumerate(cids):
            d = vmt_data[cid][comp]
            P[k, :, 0] = d["shear"]
            P[k, :, 1] = d["bending"]
            P[k, :, 2] = d["torsion"]
        base_rows = [k for k, c in enumerate(cids) if c in base_ids]
        if len(base_rows) < 4:
            continue
        for i in range(n_sta):
            pts = P[:, i, :]
            lo = pts.min(axis=0)
            span = np.ptp(pts, axis=0)
            span[span == 0] = 1.0
            q = (pts - lo) / span
            try:
                hull = ConvexHull(q[base_rows])
            except Exception:
                continue
            viol = (q @ hull.equations[:, :3].T
                    + hull.equations[:, 3]).max(axis=1)
            for k, cid in enumerate(cids):
                if viol[k] > worst.get(cid, 0.0):
                    worst[cid] = float(viol[k])
    return worst


def main() -> None:
    setup_logging("ERROR")
    os.chdir(SOLVER)

    merged_batch = BatchResult()
    merged_vmt = {}
    labels = {}
    per_variant = {}
    for vi, (name, dm, cgs, alts, combos) in enumerate(VARIANTS):
        batch, vmt, meta = run_variant(name, dm, cgs, alts, combos)
        off = (vi + 1) * 10000
        ids_here = []
        for r in batch.case_results:
            if r.case_id not in vmt:
                continue
            old = r.case_id
            r.case_id = off + old
            mc = meta.get(old)
            alt = getattr(mc, "altitude_m", 0.0) if mc else 0.0
            r.label = f"[{name}@{alt:.0f}m] {r.label}"
            merged_batch.case_results.append(r)
            merged_batch.completed_ids.add(r.case_id)
            merged_vmt[r.case_id] = vmt[old]
            labels[r.case_id] = (r.label, r.category)
            ids_here.append(r.case_id)
        per_variant[name] = ids_here
        print(f"  merged +{len(ids_here)} (total "
              f"{len(merged_batch.case_results)})", flush=True)

    print(f"\n=== merged campaign: {len(merged_batch.case_results)} cases "
          f"===", flush=True)

    # ── 변형별 자체 2D/3D 비교 (단일 조건 세트에서 5% 나오는지) ──
    variant_rows = {}
    for name, ids in per_variant.items():
        sub_batch = BatchResult()
        sub_batch.case_results = [r for r in merged_batch.case_results
                                  if r.case_id in set(ids)]
        sub_batch.completed_ids = set(ids)
        sub_vmt = {c: merged_vmt[c] for c in ids}
        _, dc2 = run_selection(sub_batch, sub_vmt, "2d")
        _, dc3 = run_selection(sub_batch, sub_vmt, "3d")
        s2 = {d.case_id for d in dc2}
        s3 = {d.case_id for d in dc3}
        rows = []
        for cid in sorted(s3 - s2):
            exc = exceedance(sub_vmt, s2, cid) * 100.0
            rows.append({"case_id": cid, "label": labels[cid][0],
                         "category": labels[cid][1],
                         "exceedance_pct": round(exc, 2)})
        rows.sort(key=lambda r: -r["exceedance_pct"])
        variant_rows[name] = {"n_2d": len(s2), "n_3d": len(s3),
                              "only_3d": rows}
        top = rows[0]["exceedance_pct"] if rows else 0.0
        print(f"  [{name:9s}] 2D {len(s2)} / 3D {len(s3)} / "
              f"3D-only {len(rows)} / max exceedance {top:.2f}%", flush=True)

    # ── 통합 캠페인 2D vs 3D ──
    _, dc2 = run_selection(merged_batch, merged_vmt, "2d")
    _, dc3 = run_selection(merged_batch, merged_vmt, "3d")
    s2 = {d.case_id for d in dc2}
    s3 = {d.case_id for d in dc3}
    only3 = sorted(s3 - s2)
    print(f"\n[merged] 2D {len(s2)} -> 3D {len(s3)}, 3D-only {len(only3)}",
          flush=True)

    exc2 = exceedance_all(merged_vmt, s2)
    exc3 = exceedance_all(merged_vmt, s3)
    rows3 = []
    for cid in only3:
        rows3.append({"case_id": cid, "label": labels[cid][0],
                      "category": labels[cid][1],
                      "exceedance_pct": round(exc2.get(cid, 0.0) * 100, 2)})
    rows3.sort(key=lambda r: -r["exceedance_pct"])
    print("\n3D-only 케이스의 2D 세트 대비 방향 초과율:")
    for r in rows3:
        flag = ("*** >=5% ***" if r["exceedance_pct"] >= 5.0 else
                "의미있음(>=1%)" if r["exceedance_pct"] >= 1.0 else "미미")
        print(f"  C{r['case_id']:6d} {r['category']:16s} "
              f"{r['exceedance_pct']:6.2f}%  {flag}  {r['label'][:52]}")

    # ── 전수 감사: 모든 비선정 케이스의 2D/3D 세트 초과율 ──
    audit2, audit3 = [], []
    all_ids = [r.case_id for r in merged_batch.case_results]
    for cid in all_ids:
        if cid not in s2:
            e2 = exc2.get(cid, 0.0) * 100.0
            if e2 >= 1.0:
                audit2.append({"case_id": cid, "label": labels[cid][0],
                               "category": labels[cid][1],
                               "exceedance_pct": round(e2, 2)})
        if cid not in s3:
            e3 = exc3.get(cid, 0.0) * 100.0
            if e3 >= 1.0:
                audit3.append({"case_id": cid, "label": labels[cid][0],
                               "category": labels[cid][1],
                               "exceedance_pct": round(e3, 2)})
    audit2.sort(key=lambda r: -r["exceedance_pct"])
    audit3.sort(key=lambda r: -r["exceedance_pct"])
    print(f"\n전수 감사 — 2D 세트 밖 >=1% 케이스 {len(audit2)}건 "
          f"(>=5%: {sum(1 for r in audit2 if r['exceedance_pct'] >= 5)}건):")
    for r in audit2[:15]:
        print(f"  C{r['case_id']:6d} {r['category']:16s} "
              f"{r['exceedance_pct']:6.2f}%  {r['label'][:52]}")
    print(f"전수 감사 — 3D 세트 밖 >=1% 케이스 {len(audit3)}건 "
          f"(>=5%: {sum(1 for r in audit3 if r['exceedance_pct'] >= 5)}건):")
    for r in audit3[:15]:
        print(f"  C{r['case_id']:6d} {r['category']:16s} "
              f"{r['exceedance_pct']:6.2f}%  {r['label'][:52]}")

    with open(os.path.join(HERE, "hull3d_severity_results.json"), "w") as f:
        json.dump({"variants": {k: v for k, v in variant_rows.items()},
                   "merged": {"n_2d": len(s2), "n_3d": len(s3),
                              "only_3d": rows3,
                              "audit_vs_2d": audit2,
                              "audit_vs_3d": audit3}},
                  f, ensure_ascii=False, indent=1)
    print("\nsaved: hull3d_severity_results.json")


if __name__ == "__main__":
    main()

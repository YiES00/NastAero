# r3 통합 분석: ILC-8 파이프라인 재실행 + 응력 보존(MC9) + 국부 6분력 비교(MC1)
"""One ILC-8 rerun feeding three r3 deliverables:

  [P] the certification pipeline itself under the new
      infeasible_policy='separate' (saturated-command accounting, MC2);
  [S] stress-preservation audit of the reduced design set vs ALL
      converged feasible cases (MC9): per-node von-Mises running max
      from a linear static re-solve of every case's exported nodal
      loads against a single reference-clamped factorization;
  [L] local 6-component vs legacy global-axis section loads on the
      governing wing / V-tail stations and on a dedicated boom
      component (MC1 before/after quantification).

Usage:  python scripts/r3_ilc8_analysis.py
Output: r3_ilc8_analysis.json next to this script + console tables.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
os.chdir(SOLVER)

from run_ilc8_cert_analysis import main as pipeline_main  # noqa: E402
from nastaero.fem.model import FEModel                    # noqa: E402
from nastaero.fem.stress_recovery import recover_stresses_to_nodes  # noqa: E402
from nastaero.loads_analysis.component_id import ComponentDef  # noqa: E402
from nastaero.loads_analysis.vmt import compute_vmt       # noqa: E402


def stress_preservation(ctx):
    """[S] full-set vs design-set per-node von-Mises preservation."""
    model = ctx["model"]
    br = ctx["batch_result"]
    sel = ctx["selection"]

    fe = FEModel(model)
    eff = model.get_effective_subcase(
        model.subcases[0] if model.subcases else model.global_case)
    K_ff, _, _, f_dofs, _ = fe.get_partitioned_system(eff)
    dof_mgr = fe.dof_mgr

    # 기준 클램프: CG에 가장 가까운 동체 절점 6자유도 — 케이스 하중은
    # 관성완화로 자체 평형이므로 반력은 ~0이고 응력장은 구속 무관
    cg_x = 4450.0
    best, best_d = None, np.inf
    for nid, node in model.nodes.items():
        if 100000 <= nid <= 299999:
            x, y, _ = node.xyz_global
            d = (x - cg_x) ** 2 + y ** 2
            if d < best_d:
                best, best_d = nid, d
    clamp = {dof_mgr.get_dof(best, c) for c in range(6)}
    keep = np.array([i for i, dof in enumerate(f_dofs) if dof not in clamp])
    K_ll = K_ff.tocsr()[keep][:, keep].tocsc()
    kdiag = np.abs(np.asarray(K_ll.diagonal()).ravel())
    eps = float(np.mean(kdiag[kdiag > 0])) * 1e-8
    lu = spla.splu(K_ll + sp.eye(K_ll.shape[0], format='csc') * eps)

    sorted_nids = dof_mgr.node_ids
    f_index = {dof: i for i, dof in enumerate(f_dofs)}

    def solve_case(nodal_forces):
        F = np.zeros(len(f_dofs))
        for nid, f6 in nodal_forces.items():
            if nid not in model.nodes:
                continue
            for c in range(6):
                dof = dof_mgr.get_dof(nid, c + 1)
                i = f_index.get(dof)
                if i is not None:
                    F[i] += f6[c]
        u = np.zeros(len(f_dofs))
        u[keep] = lu.solve(F[keep])
        S = recover_stresses_to_nodes(model, dof_mgr, f_dofs, u,
                                      sorted_nids)
        sxx, syy, szz, sxy, syz, szx = S.T
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                            + (szz - sxx) ** 2)
                     + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
        return vm

    feasible_ids = [cr.case_id for cr in br.case_results
                    if cr.converged and cr.nodal_forces
                    and getattr(cr, "rotor_command_feasible", True)]
    design_ids = {d.case_id for d in sel["design_cases"]
                  if d.rotor_command_feasible}

    n_nodes = len(sorted_nids)
    vm_all = np.zeros(n_nodes)
    vm_sel = np.zeros(n_nodes)
    t0 = time.time()
    for k, cid in enumerate(feasible_ids):
        cr = br.get_result(cid)
        vm = solve_case(cr.nodal_forces)
        np.maximum(vm_all, vm, out=vm_all)
        if cid in design_ids:
            np.maximum(vm_sel, vm, out=vm_sel)
        if (k + 1) % 25 == 0:
            print(f"    [S] {k+1}/{len(feasible_ids)} cases "
                  f"({time.time()-t0:.0f}s)")

    # 무의미한 저응력 절점 제외: 전수 최대의 1% 이상 응력을 갖는
    # 절점만 보존율 통계에 반영
    thresh = 0.01 * vm_all.max()
    sig = vm_all > thresh
    ratio = np.ones(n_nodes)
    ratio[sig] = vm_sel[sig] / vm_all[sig]
    n_sig = int(sig.sum())
    worst_i = int(np.argmin(ratio))
    out = {
        "n_cases_full": len(feasible_ids),
        "n_cases_design": len(design_ids),
        "n_nodes_significant": n_sig,
        "min_preservation_ratio": float(ratio.min()),
        "worst_node": int(sorted_nids[worst_i]),
        "n_nodes_below_0.99": int(np.sum(ratio[sig] < 0.99)),
        "n_nodes_below_0.95": int(np.sum(ratio[sig] < 0.95)),
        "pct_nodes_fully_preserved": float(
            np.sum(ratio[sig] >= 1.0 - 1e-9) / max(n_sig, 1) * 100.0),
    }
    print("\n=== [S] 응력 보존 감사 (설계 세트 vs 전수) ===")
    for k, v in out.items():
        print(f"    {k}: {v}")
    return out


def local6_comparison(ctx):
    """[L] legacy global-axis vs local 6-component on wing/V-tail/boom."""
    model = ctx["model"]
    br = ctx["batch_result"]
    sel = ctx["selection"]
    comps = {c.name: c for c in ctx["components"].components}

    # 붐 전용 구성품 (파이프라인에선 파일런 관례로 날개 귀속) — 우측
    # 전방 붐 대역 730000-749999, 스팬축 X
    boom_ids = [n for n in model.nodes if 730000 <= n <= 749999]
    boom = ComponentDef(name="Right Boom", node_ids=boom_ids,
                        span_axis=0, shear_axis=2, bending_axis=1,
                        torsion_axis=0, integration_sign=1.0)

    targets = [comps["Right Wing"], comps["Right V-Tail"], boom]
    top_ids = [d.case_id for d in sel["design_cases"][:10]
               if d.rotor_command_feasible]

    rows = []
    for comp in targets:
        for cid in top_ids:
            cr = br.get_result(cid)
            if not cr or not cr.nodal_forces:
                continue
            c = compute_vmt(model, cr.nodal_forces, comp, n_stations=20)
            if c.local_stations is None:
                continue
            # 뿌리 스테이션 비교: 전역 (V, M, T) vs 국부 6분력
            rows.append({
                "component": comp.name, "case_id": cid,
                "legacy_V": float(c.shear[0]),
                "legacy_M": float(c.bending_moment[0]),
                "legacy_T": float(c.torsion[0]),
                "local_N": float(c.local_N[0]),
                "local_Vy": float(c.local_Vy[0]),
                "local_Vz": float(c.local_Vz[0]),
                "local_Mx": float(c.local_Mx[0]),
                "local_My": float(c.local_My[0]),
                "local_Mz": float(c.local_Mz[0]),
            })

    print("\n=== [L] 전역 3성분 vs 국부 6분력 (뿌리 스테이션) ===")
    for comp_name in {r["component"] for r in rows}:
        sub = [r for r in rows if r["component"] == comp_name]
        dV = [abs(r["local_Vz"] - r["legacy_V"])
              / max(abs(r["legacy_V"]), 1.0) for r in sub]
        dT = [abs(r["local_Mx"] - r["legacy_T"])
              / max(abs(r["legacy_T"]), 1.0) for r in sub]
        aN = [abs(r["local_N"]) for r in sub]
        aVy = [abs(r["local_Vy"]) for r in sub]
        print(f"    {comp_name}: |Vz-V|/|V| max {max(dV)*100:.1f}%, "
              f"|Mx-T|/|T| max {max(dT)*100:.1f}%, "
              f"신규 |N| max {max(aN):,.0f} N, |Vy| max {max(aVy):,.0f} N")
    return rows


def main():
    ctx = pipeline_main()
    s = stress_preservation(ctx)
    l6 = local6_comparison(ctx)
    sel = ctx["selection"]
    pl = sel["propulsion_limit"]
    proc = ctx["processor"]
    br = ctx["batch_result"]

    cat_dist = proc.critical_category_distribution()

    # 우측 날개 비틀림 최소(T-min) 스테이션의 지배 카테고리 분할
    case_cat = {cr.case_id: cr.category for cr in br.case_results}
    tmin_split = {}
    env = proc.get_envelope("Right Wing")
    if env:
        for se in env.envelopes:
            cid = se.T_min_case_id
            if cid is not None:
                cat = case_cat.get(cid, "?")
                tmin_split[cat] = tmin_split.get(cat, 0) + 1
    print(f"    날개 T-min 스테이션 분할: {tmin_split}")
    sat = [(cr.case_id, cr.label, cr.category,
            float(getattr(cr, "rotor_thrust_shortfall", 0.0)))
           for cr in br.case_results
           if not getattr(cr, "rotor_command_feasible", True)]
    n_feas_design = sum(1 for d in sel["design_cases"]
                        if d.rotor_command_feasible)
    print("\n=== 집계 (논문 갱신용) ===")
    print(f"    임계 레코드 {sel['n_critical']}, 카테고리 분포:")
    for k, v in sorted(cat_dist.items(), key=lambda t: -t[1]):
        print(f"      {k:20s}: {v}")
    print(f"    포화 케이스 {len(sat)}건, 최대 부족률 "
          f"{max((t[3] for t in sat), default=0)*100:.1f}%")
    print(f"    설계 세트: 실현가능 {n_feas_design} + 추진계 한계 "
          f"{sel['n_design_cases'] - n_feas_design} = "
          f"{sel['n_design_cases']}")

    data = {
        "propulsion_limit": {
            "n_infeasible": pl["n_infeasible"],
            "n_exceedances": len(pl["exceedances"]),
            "n_appended": pl["n_appended_design_cases"],
            "exceedances": pl["exceedances"][:50],
            "saturated_cases": sat,
        },
        "n_design_cases": sel["n_design_cases"],
        "n_design_feasible": n_feas_design,
        "n_critical": sel["n_critical"],
        "critical_category_distribution": cat_dist,
        "wing_tmin_station_split": tmin_split,
        "stress_preservation": s,
        "local6_root_rows": l6,
    }
    out = os.path.join(HERE, "r3_ilc8_analysis.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

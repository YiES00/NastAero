# 국부 6분력 선정이 설계 세트를 어떻게 바꾸는지 (r4 MC3 항목 9)
"""Design-set comparison: global-axis triplet vs. global + local six.

r4 MC3 asks whether wiring the component-local six-component section
loads into the envelope and critical-case selection actually changes
the design set — i.e. whether the global-axis (V, M, T) selection was
missing cases that govern a local channel (axial force, chord shear,
in-plane bending) on canted or swept components.

Runs the ILC-8 certification pipeline once, then performs the design
selection twice on the SAME VMT data: once with include_local6=False
(the previous behavior) and once True (the new default). Reports the
two design sets, the cases each finds that the other does not, and
which local quantity/component/station each new case governs.

Usage:  python scripts/r4_local6_design_set.py
Output: r4_local6_design_set.json next to this script.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
os.chdir(SOLVER)

from run_ilc8_cert_analysis import main as pipeline_main   # noqa: E402
from ascent_load.loads_analysis.certification.envelope import (  # noqa: E402
    select_critical_design_loads, LOCAL_QUANTITIES,
)


def main():
    ctx = pipeline_main()
    model = ctx["model"]
    br = ctx["batch_result"]
    comps = ctx["components"]
    vmt = ctx["vmt_data"]

    print("\n=== 설계 세트 비교: 전역 3성분 대 전역+국부 6분력 ===")
    sel_g = select_critical_design_loads(
        model, br, fuselage_cg_x=4450.0, infeasible_policy="separate",
        components=comps, vmt_data=vmt, include_local6=False)
    sel_l = select_critical_design_loads(
        model, br, fuselage_cg_x=4450.0, infeasible_policy="separate",
        components=comps, vmt_data=vmt, include_local6=True)

    ids_g = {d.case_id for d in sel_g["design_cases"]}
    ids_l = {d.case_id for d in sel_l["design_cases"]}
    only_l = sorted(ids_l - ids_g)
    only_g = sorted(ids_g - ids_l)

    print(f"  전역 3성분  : 임계 {sel_g['n_critical']:,} → 설계 "
          f"{sel_g['n_design_cases']}건")
    print(f"  전역+국부   : 임계 {sel_l['n_critical']:,} → 설계 "
          f"{sel_l['n_design_cases']}건")
    print(f"  국부에서만 새로 들어온 케이스: {len(only_l)}건 {only_l}")
    print(f"  국부 선정에서 빠진 케이스   : {len(only_g)}건 {only_g}")

    # 새 케이스가 무엇을 지배하는지
    detail = []
    by_id = {d.case_id: d for d in sel_l["design_cases"]}
    for cid in only_l:
        d = by_id[cid]
        locg = [g for g in d.governs if g[2] in LOCAL_QUANTITIES]
        comps_hit = sorted({g[0] for g in locg})
        qtys = sorted({g[2] for g in locg})
        cr = br.get_result(cid)
        detail.append(dict(case_id=cid, label=getattr(cr, "label", ""),
                           category=getattr(cr, "category", ""),
                           n_local_govern=len(locg),
                           components=comps_hit, quantities=qtys))
        print(f"    case {cid} [{getattr(cr,'category','')}] "
              f"{getattr(cr,'label','')}: 국부 지배 {len(locg)}건, "
              f"{comps_hit} / {qtys}")

    # 국부 물리량별 지배 건수 (전 구성품)
    per_qty = {}
    per_comp = {}
    for d in sel_l["design_cases"]:
        for g in d.governs:
            if g[2] in LOCAL_QUANTITIES:
                per_qty[g[2]] = per_qty.get(g[2], 0) + 1
                per_comp[g[0]] = per_comp.get(g[0], 0) + 1
    print(f"\n  국부 지배 기록: 물리량별 {per_qty}")
    print(f"                  구성품별 {per_comp}")

    out = dict(
        global_only=dict(n_critical=sel_g["n_critical"],
                         n_design=sel_g["n_design_cases"],
                         case_ids=sorted(ids_g)),
        with_local6=dict(n_critical=sel_l["n_critical"],
                         n_design=sel_l["n_design_cases"],
                         case_ids=sorted(ids_l)),
        only_in_local6=detail,
        only_in_global=only_g,
        local_govern_by_quantity=per_qty,
        local_govern_by_component=per_comp,
    )
    path = os.path.join(HERE, "r4_local6_design_set.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()

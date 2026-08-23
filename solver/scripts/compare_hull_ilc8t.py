# ILC-8T(틸트) 전체 매트릭스의 2D vs 3D 헐 선정 비교 — 재트림 채택 케이스 포함
"""2-D vs 3-D hull selection audit on the full ILC-8T (tilt) matrix.

Runs the complete ILC-8T pipeline (fixed-wing + landing + VTOL matrix
incl. the tilt conversion corridor, corridor gust, and tilt-stuck
cases), adopts the (failure x re-trim) TILT-family cases from
RetrimScreen, then reruns the design-case selection with the 2-D
pairwise interaction hull and the 3-D convex hull and scores every
3-D-only addition with the directional-exceedance metric (per-station
axis-range-normalized distance outside the 2-D-selected set's hull).

Usage:  python scripts/compare_hull_ilc8t.py
Output: prints the comparison; writes hull_comparison_ilc8t.json here.
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)

from nastaero.bdf.parser import parse_bdf                      # noqa: E402
from nastaero.config import setup_logging                      # noqa: E402
from nastaero.loads_analysis.certification.aircraft_config import (  # noqa: E402
    AircraftConfig,
)
from nastaero.loads_analysis.certification.load_case_matrix import (  # noqa: E402
    LoadCaseMatrix,
)
from nastaero.loads_analysis.certification.retrim_events import (  # noqa: E402
    RetrimScreen,
)
from nastaero.loads_analysis.certification.vtol_batch_runner import (  # noqa: E402
    VTOLBatchRunner,
)
from nastaero.loads_analysis.certification.vtol_load_case_matrix import (  # noqa: E402
    VTOLLoadCaseMatrix,
)
from nastaero.loads_analysis.certification.vmt_bridge import (  # noqa: E402
    compute_vmt_for_batch,
)
from nastaero.models.ilc8t import make_ilc8t_vtol_config       # noqa: E402

from compare_hull_selection import exceedance, run_selection   # noqa: E402
from hull3d_severity_search import build_components            # noqa: E402

BDF = "tests/validation/ILC8T/ilc8t.bdf"
CFG = "tests/validation/ILC8T/ilc8t_cert_config.yaml"


def main() -> None:
    setup_logging("ERROR")
    os.chdir(SOLVER)
    t0 = time.time()
    model = parse_bdf(BDF)
    with open(CFG) as f:
        config = AircraftConfig.from_dict(yaml.safe_load(f))
    vtol_config = make_ilc8t_vtol_config()
    components = build_components(model)

    conv = LoadCaseMatrix(config)
    conv.generate_all(bdf_model=model, include_dynamic=False)
    vtol_matrix = VTOLLoadCaseMatrix(vtol_config, config)
    vtol_matrix.generate_all()
    conv_only = SimpleNamespace(
        flight_cases=list(conv.flight_cases),
        landing_cases=list(conv.landing_cases),
        dynamic_cases=[], config=config)
    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    runner = VTOLBatchRunner(conv_only, vtol_matrix, bdf_model=model,
                             vtol_config=vtol_config,
                             n_workers=n_cpus,
                             include_transient=False)
    batch = runner.run()
    print(f"matrix: {batch.n_converged}/{batch.n_total} converged "
          f"({time.time()-t0:.0f}s)", flush=True)

    # (고장×재트림) TILT 계열 선별 + 채택 케이스 병합
    t1 = time.time()
    screen = RetrimScreen(model, vtol_config, config,
                          components=components,
                          fuselage_cg_x=4450.0)
    events = screen.screen()
    retrim = screen.realize(events, top_n=8)
    for c in retrim:
        batch.case_results.append(c)
        batch.completed_ids.add(c.case_id)
    print(f"retrim: {len(events)} events -> {len(retrim)} adopted "
          f"({time.time()-t1:.0f}s)", flush=True)

    vmt = compute_vmt_for_batch(model, batch, components=components,
                                fuselage_cg_x=4450.0)
    labels = {c.case_id: (c.label, c.category)
              for c in batch.case_results}
    print(f"\n=== ILC-8T hull comparison ({len(vmt)} VMT cases) ===",
          flush=True)

    # 채택된 재트림 케이스의 우익 비틀림 범위 (논문 2 수치 재현)
    import numpy as _np
    _rt = {c.case_id for c in retrim}
    _lo, _hi = 0.0, 0.0
    for cid in sorted(_rt):
        cd = vmt.get(cid, {})
        if "Right Wing" not in cd:
            continue
        arr = _np.asarray(cd["Right Wing"]["torsion"], float) / 1e6
        _lo = min(_lo, float(arr.min())); _hi = max(_hi, float(arr.max()))
        print(f"  retrim C{cid} tor[{arr.min()/1:7.2f}, {arr.max()/1:6.2f}] kN*m")
    print(f"  채택 재트림 우익 비틀림 범위: {_lo:.2f} ~ {_hi:.2f} kN*m",
          flush=True)

    proc2, dc2 = run_selection(batch, vmt, "2d")
    proc3, dc3 = run_selection(batch, vmt, "3d")
    s2 = {d.case_id for d in dc2}
    s3 = {d.case_id for d in dc3}
    only3, only2 = sorted(s3 - s2), sorted(s2 - s3)

    print(f"2D: critical {len(proc2.get_critical_cases())} -> "
          f"design {len(s2)}")
    print(f"3D: critical {len(proc3.get_critical_cases())} -> "
          f"design {len(s3)}")
    print(f"2D에만: {only2}")
    print(f"3D에만: {only3}")

    rows = []
    for cid in only3:
        exc = exceedance(vmt, s2, cid) * 100.0
        lab, cat = labels.get(cid, ("?", "?"))
        rows.append({"case_id": cid, "label": lab, "category": cat,
                     "exceedance_pct": round(exc, 2)})
    rows.sort(key=lambda r: -r["exceedance_pct"])
    print("\n3D 추가 케이스 방향 초과율 (2D 세트 헐 대비, 축범위%):")
    for r in rows:
        flag = ("의미있음" if r["exceedance_pct"] >= 5.0 else
                "미미" if r["exceedance_pct"] >= 1.0 else
                "과선정 성격")
        print(f"  C{r['case_id']:5d} {r['category']:18s} "
              f"{r['exceedance_pct']:6.2f}%  {flag}  "
              f"{str(r['label'])[:42]}")

    cats = {}
    for d in dc3:
        cats[labels[d.case_id][1]] = cats.get(
            labels[d.case_id][1], 0) + 1
    print("\n3D 설계 케이스 범주 분포:")
    for cat, cnt in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:22s} {cnt}")

    with open(os.path.join(HERE, "hull_comparison_ilc8t.json"),
              "w") as f:
        json.dump({"design_2d": sorted(s2), "design_3d": sorted(s3),
                   "only_3d": rows, "only_2d": only2,
                   "retrim_adopted": [c.case_id for c in retrim],
                   "categories_3d": cats}, f,
                  ensure_ascii=False, indent=1)
    print("\nsaved: hull_comparison_ilc8t.json")


if __name__ == "__main__":
    main()

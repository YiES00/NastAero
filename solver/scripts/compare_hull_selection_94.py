# 2D vs 3D 헐 선정 비교의 94케이스(VTOL 포함) 확장 — 드라이버 파이프라인 재사용
"""Extend the 2-D vs 3-D hull comparison to the full ILC-8 94-case
matrix (55 fixed-wing incl. landing + 39 VTOL incl. rotor jam).

Reuses the run_ilc8_cert_analysis pipeline by intercepting the
EnvelopeProcessor construction (captures batch_result + vmt_data with
the driver's manual components), then reruns the selection three ways
and scores 3-D-only additions with the directional-exceedance metric
of compare_hull_selection.py.

Usage:  python scripts/compare_hull_selection_94.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)

from nastaero.loads_analysis.certification import envelope as env_mod  # noqa: E402

captured = {}
_RealProc = env_mod.EnvelopeProcessor


class _Recorder(_RealProc):
    def __init__(self, batch_result, vmt_data=None):
        super().__init__(batch_result, vmt_data)
        if vmt_data and "batch" not in captured:
            captured["batch"] = batch_result
            captured["vmt"] = vmt_data


env_mod.EnvelopeProcessor = _Recorder

spec = importlib.util.spec_from_file_location(
    "ilc8_driver", os.path.join(SOLVER, "run_ilc8_cert_analysis.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
os.chdir(SOLVER)          # 드라이버 상대경로 보호
driver.main()
env_mod.EnvelopeProcessor = _RealProc

batch, vmt = captured["batch"], captured["vmt"]
labels = {c.case_id: (c.label, c.category) for c in batch.case_results}
print(f"\n=== 94-case hull comparison (captured {len(vmt)} cases) ===",
      flush=True)

from compare_hull_selection import exceedance, run_selection  # noqa: E402

proc2, dc2 = run_selection(batch, vmt, "2d")
proc3, dc3 = run_selection(batch, vmt, "3d")
s2 = {d.case_id for d in dc2}
s3 = {d.case_id for d in dc3}
only3, only2 = sorted(s3 - s2), sorted(s2 - s3)

print(f"2D: critical {len(proc2.get_critical_cases())} -> design {len(s2)}")
print(f"3D: critical {len(proc3.get_critical_cases())} -> design {len(s3)}")
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
            "미미" if r["exceedance_pct"] >= 1.0 else "과선정 성격")
    print(f"  C{r['case_id']:4d} {r['category']:16s} "
          f"{r['exceedance_pct']:6.2f}%  {flag}  {str(r['label'])[:40]}")

with open(os.path.join(HERE, "hull_comparison_94.json"), "w") as f:
    json.dump({"design_2d": sorted(s2), "design_3d": sorted(s3),
               "only_3d": rows, "only_2d": only2}, f,
              ensure_ascii=False, indent=1)
print("\nsaved: hull_comparison_94.json")

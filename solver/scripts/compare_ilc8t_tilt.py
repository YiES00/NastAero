# ILC-8 (L+C) vs ILC-8T (틸트) 하중 비교 — 동일 기체구조·등중량에서 추진 방식만 교체
"""Compare certification loads of the lift+cruise ILC-8 and the tilt
variant ILC-8T on the same airframe at equal weight.

Both aircraft run the full pipeline (conventional flight + landing +
VTOL matrix incl. the tilt conversion corridor for the -8T) and the
wing/boom envelope extremes and governing categories are tabulated
side by side. Because the airframe, weight, and fixed-wing cases are
identical, every difference in the tables is attributable to the
propulsion architecture (pusher + fixed lift rotors vs tilting front
row).

Usage:  python scripts/compare_ilc8t_tilt.py
Output: prints the comparison; writes ilc8t_comparison.json here.
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

from ascent_load.bdf.parser import parse_bdf                      # noqa: E402
from ascent_load.config import setup_logging                      # noqa: E402
from ascent_load.loads_analysis.certification.aircraft_config import (  # noqa: E402
    AircraftConfig,
)
from ascent_load.loads_analysis.certification.load_case_matrix import (  # noqa: E402
    LoadCaseMatrix,
)
from ascent_load.loads_analysis.certification.vtol_batch_runner import (  # noqa: E402
    VTOLBatchRunner,
)
from ascent_load.loads_analysis.certification.vtol_load_case_matrix import (  # noqa: E402
    VTOLLoadCaseMatrix,
)
from ascent_load.loads_analysis.certification.vmt_bridge import (  # noqa: E402
    compute_vmt_for_batch,
)

from hull3d_severity_search import build_components            # noqa: E402

AIRCRAFT = [
    ("ILC-8 (L+C)", "tests/validation/ILC8/ilc8.bdf",
     "tests/validation/ILC8/ilc8_cert_config.yaml",
     "ascent_load.models.ilc8", "make_ilc8_vtol_config"),
    ("ILC-8T (tilt)", "tests/validation/ILC8T/ilc8t.bdf",
     "tests/validation/ILC8T/ilc8t_cert_config.yaml",
     "ascent_load.models.ilc8t", "make_ilc8t_vtol_config"),
]


def run_aircraft(name, bdf_path, cfg_path, mod, fn):
    import importlib

    t0 = time.time()
    model = parse_bdf(os.path.join(SOLVER, bdf_path))
    with open(os.path.join(SOLVER, cfg_path)) as f:
        config = AircraftConfig.from_dict(yaml.safe_load(f))
    vtol_config = getattr(importlib.import_module(mod), fn)()

    conv = LoadCaseMatrix(config)
    conv.generate_all(bdf_model=model, include_dynamic=False)
    vtol_matrix = VTOLLoadCaseMatrix(vtol_config, config)
    vtol_cases = vtol_matrix.generate_all()
    conv_only = SimpleNamespace(
        flight_cases=list(conv.flight_cases),
        landing_cases=list(conv.landing_cases),
        dynamic_cases=[], config=config)
    n_cpus = max(1, (os.cpu_count() or 2) // 2)
    runner = VTOLBatchRunner(conv_only, vtol_matrix, bdf_model=model,
                             vtol_config=vtol_config, n_workers=n_cpus,
                             include_transient=False)
    batch = runner.run()
    components = build_components(model)
    vmt = compute_vmt_for_batch(model, batch, components=components,
                                fuselage_cg_x=4450.0)
    labels = {c.case_id: (c.label, c.category)
              for c in batch.case_results}
    print(f"[{name}] {batch.n_converged}/{batch.n_total} converged, "
          f"{len(vmt)} VMT cases ({time.time()-t0:.0f}s)", flush=True)
    return vmt, labels, batch


def envelope_extremes(vmt, labels, comp, qty):
    """부재·수량의 (min, max)와 지배 케이스 (스팬 전 스테이션)."""
    lo, hi = None, None
    for cid, cd in vmt.items():
        if comp not in cd:
            continue
        arr = np.asarray(cd[comp][qty], float)
        mn, mx = float(arr.min()), float(arr.max())
        if lo is None or mn < lo[0]:
            lo = (mn, labels[cid])
        if hi is None or mx > hi[0]:
            hi = (mx, labels[cid])
    return lo, hi


def main() -> None:
    setup_logging("ERROR")
    os.chdir(SOLVER)
    results = {}
    for row in AIRCRAFT:
        results[row[0]] = run_aircraft(*row)

    out = {}
    print("\n=== 포락선 극값 비교 (동일 기체구조·등중량, 추진 방식만"
          " 상이) ===")
    for comp, qty, unit, scale in (
            ("Right Wing", "bending", "kN·m", 1e6),
            ("Right Wing", "torsion", "kN·m", 1e6),
            ("Right Wing", "shear", "kN", 1e3),
            ("Fuselage", "bending", "kN·m", 1e6),
            ("Fuselage", "torsion", "kN·m", 1e6)):
        print(f"\n[{comp} — {qty}]")
        entry = {}
        for name in results:
            vmt, labels, _ = results[name]
            lo, hi = envelope_extremes(vmt, labels, comp, qty)
            print(f"  {name:14s} min {lo[0]/scale:9.2f} {unit} "
                  f"<- {lo[1][1]}: {lo[1][0][:44]}")
            print(f"  {'':14s} max {hi[0]/scale:9.2f} {unit} "
                  f"<- {hi[1][1]}: {hi[1][0][:44]}")
            entry[name] = {"min": lo[0], "min_case": lo[1][0],
                           "min_cat": lo[1][1],
                           "max": hi[0], "max_case": hi[1][0],
                           "max_cat": hi[1][1]}
            # 범주별 극값 — 계열 단독 경계(예: 틸트 회랑만) 인용용
            by_cat = {}
            for cid, cd in vmt.items():
                if comp not in cd:
                    continue
                import numpy as _np
                arr = _np.asarray(cd[comp][qty], float)
                cat = labels.get(cid, ("", "?"))[1]
                d = by_cat.setdefault(cat, [float("inf"), float("-inf")])
                d[0] = min(d[0], float(arr.min()))
                d[1] = max(d[1], float(arr.max()))
            entry[name]["by_category"] = {
                k: {"min": v[0], "max": v[1]} for k, v in sorted(by_cat.items())}
        out[f"{comp}/{qty}"] = entry

    with open(os.path.join(HERE, "ilc8t_comparison.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nsaved: ilc8t_comparison.json")


if __name__ == "__main__":
    main()

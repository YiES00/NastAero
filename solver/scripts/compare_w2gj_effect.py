# GACOMP W2GJ(캠버/비틀림 DMI) 적용 전후 트림·날개 루트 하중 A/B 비교 재현 스크립트
"""Quantify the effect of the deck-supplied W2GJ DMI on GACOMP trim.

Runs the free-trim deck twice — with the W2GJ camber/incidence downwash
applied (as MSC Nastran does) and with it stripped (flat-plate panels,
the pre-fix ASCENT-Load behavior) — and tabulates per-subcase trim
variables (alpha, elevator), the 1-g invariant alpha*M^2, and the
right-wing root V/M/T from the combined nodal loads.

Usage:  python scripts/compare_w2gj_effect.py
Output: prints the comparison table; writes w2gj_ab_results.json
        next to this script.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)

from ascent_load.bdf.parser import parse_bdf                      # noqa: E402
from ascent_load.loads_analysis.component_id import identify_components  # noqa: E402
from ascent_load.loads_analysis.vmt import compute_vmt            # noqa: E402
from ascent_load.solvers.sol144 import solve_trim                 # noqa: E402

BDF = os.path.join(SOLVER, "tests/validation/GACOMP/p400r3-free-trim.bdf")


def run(machs, strip_w2gj: bool):
    model = parse_bdf(BDF)
    if strip_w2gj:
        model.dmis.clear()
    res = solve_trim(model, n_workers=-1)
    comps = identify_components(model)
    rwing = next(c for c in comps.components if "Right Wing" in c.name)
    rows = []
    for sc in res.subcases:
        tv = sc.trim_variables or {}
        a = tv.get("ANGLEA", 0.0)
        mach = machs.get(sc.subcase_id, 0.0)
        row = {"sc": sc.subcase_id, "mach": mach,
               "alpha_deg": math.degrees(a),
               "elev_rad": tv.get("ELEV", 0.0),
               "aM2e3": a * mach * mach * 1e3}
        if sc.nodal_combined_forces:
            c = compute_vmt(model, sc.nodal_combined_forces, rwing,
                            n_stations=50, subcase_id=sc.subcase_id)
            row["root_V"] = float(c.shear[0])
            row["root_M"] = float(c.bending_moment[0])
            row["root_T"] = float(c.torsion[0])
        rows.append(row)
    return rows


def main() -> None:
    m = parse_bdf(BDF)
    machs = {tid: t.mach for tid, t in m.trims.items()}
    del m

    print("=== RUN 1: W2GJ ON (deck DMI applied) ===", flush=True)
    on = run(machs, strip_w2gj=False)
    print("=== RUN 2: W2GJ OFF (flat plate) ===", flush=True)
    off = run(machs, strip_w2gj=True)

    hdr = (f"{'M':>4} | {'a_off':>8} {'a_on':>8} {'da':>7} (deg) | "
           f"{'de_off':>8} {'de_on':>8} (rad) | "
           f"{'aM2_off':>8} {'aM2_on':>8} | {'dV%':>7} {'dM%':>7} {'dT%':>7}")
    print(hdr)
    print("-" * len(hdr))
    table = []
    for o, n in zip(off, on):
        def pct(k):
            return ((n.get(k, 0.0) - o.get(k, 0.0)) / abs(o[k]) * 100
                    if o.get(k) else 0.0)
        dv, dm, dt = pct("root_V"), pct("root_M"), pct("root_T")
        print(f"{o['mach']:4.1f} | {o['alpha_deg']:8.3f} {n['alpha_deg']:8.3f} "
              f"{n['alpha_deg'] - o['alpha_deg']:7.3f}       | "
              f"{o['elev_rad']:8.4f} {n['elev_rad']:8.4f}       | "
              f"{o['aM2e3']:8.2f} {n['aM2e3']:8.2f} | "
              f"{dv:7.2f} {dm:7.2f} {dt:7.2f}")
        table.append({"mach": o["mach"], "off": o, "on": n,
                      "dV_pct": dv, "dM_pct": dm, "dT_pct": dt})

    out = os.path.join(HERE, "w2gj_ab_results.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=1)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

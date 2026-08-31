# MSC 대 NastAero SOL 144 변위장의 사전 기준 정량 비교 (r3 MC3)
"""Formal displacement-field comparison metrics, ILC-8 v8 deck.

Acceptance criteria — DECLARED BEFORE COMPUTATION (r3 MC3 요구):
  A1: per-subcase L2 relative error of the elastic T3 field <= 10 %
  A2: per-subcase max |dT3_elastic| / max |T3_elastic,MSC| <= 15 %
  A3: wing-tip elastic deflection difference <= 10 %

"Elastic" means each code's field after least-squares removal of the
best-fit rigid-body motion (t + theta x r, 6 parameters) — the SUPORT
reference leaves both solutions defined only up to rigid content, so
raw fields are not comparable; the elastic remainder is.

Usage:  python scripts/r3_msc_displacement_metrics.py
Data:   tests/validation/ILC8/ilc8_msc_sol144_v8_shellbend{_MSC,}.f06
Output: r3_msc_displacement_metrics.json next to this script.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)

V8 = os.path.join(SOLVER, "tests", "validation", "ILC8")
F_MSC = os.path.join(V8, "ilc8_msc_sol144_v8_shellbend_MSC.f06")
F_NA = os.path.join(V8, "ilc8_msc_sol144_v8_shellbend.f06")
DECK = os.path.join(V8, "ilc8_msc_sol144_v8_shellbend.bdf")

CRITERIA = {"A1_L2_T3_pct": 10.0, "A2_max_T3_pct": 15.0,
            "A3_tip_pct": 10.0}

_ROW = re.compile(r"^\s+(\d+)\s+G\s+([-+.\dE]+)\s+([-+.\dE]+)\s+"
                  r"([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)\s+([-+.\dE]+)")
_SC = re.compile(r"SUBCASE\s+(\d+)")


def parse_displacements(path):
    """{subcase: {nid: ndarray(6)}} from an F06."""
    out = {}
    sc = None
    in_block = False
    with open(path, errors="ignore") as f:
        for line in f:
            m = _SC.search(line)
            if m:
                sc = int(m.group(1))
            if "D I S P L A C E M E N T   V E C T O R" in line:
                in_block = True
                continue
            if in_block:
                r = _ROW.match(line)
                if r:
                    nid = int(r.group(1))
                    out.setdefault(sc, {})[nid] = np.array(
                        [float(r.group(i)) for i in range(2, 8)])
                elif line.strip() and "POINT ID" not in line \
                        and not line.startswith(" "):
                    in_block = False
    return out


def remove_rigid(xyz, u):
    """u_elastic = u - (t + theta x r) 최소제곱 강체 성분 제거 (T1-3)."""
    n = len(xyz)
    A = np.zeros((3 * n, 6))
    b = u[:, :3].reshape(-1)
    r = xyz - xyz.mean(axis=0)
    for i in range(n):
        A[3 * i:3 * i + 3, :3] = np.eye(3)
        rx, ry, rz = r[i]
        A[3 * i:3 * i + 3, 3:] = np.array(
            [[0, rz, -ry], [-rz, 0, rx], [ry, -rx, 0]])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return (b - A @ p).reshape(n, 3)


def main():
    from nastaero.bdf.parser import parse_bdf
    model = parse_bdf(DECK)
    print(f"덱 절점 {len(model.nodes)}")

    d_msc = parse_displacements(F_MSC)
    d_na = parse_displacements(F_NA)
    print(f"MSC 서브케이스 {sorted(d_msc)}, NastAero {sorted(d_na)}")

    wing_nids = [n for n in model.nodes if 300000 <= n <= 499999]
    results = {}
    for sc in sorted(set(d_msc) & set(d_na)):
        common = sorted(set(d_msc[sc]) & set(d_na[sc]) & set(model.nodes))
        xyz = np.array([model.nodes[n].xyz_global for n in common])
        um = np.array([d_msc[sc][n] for n in common])
        un = np.array([d_na[sc][n] for n in common])
        em = remove_rigid(xyz, um)
        en = remove_rigid(xyz, un)

        t3m, t3n = em[:, 2], en[:, 2]
        l2 = np.linalg.norm(t3n - t3m) / max(np.linalg.norm(t3m), 1e-12)
        mx = np.max(np.abs(t3n - t3m)) / max(np.max(np.abs(t3m)), 1e-12)

        widx = [i for i, n in enumerate(common) if n in set(wing_nids)]
        if widx:
            wi = np.array(widx)
            tipi = wi[np.argmax(np.abs(t3m[wi]))]
            tip_m, tip_n = t3m[tipi], t3n[tipi]
            tip_pct = abs(tip_n - tip_m) / max(abs(tip_m), 1e-12) * 100
        else:
            tip_m = tip_n = tip_pct = float("nan")

        results[sc] = {
            "n_common": len(common),
            "L2_T3_pct": float(l2 * 100),
            "max_T3_pct": float(mx * 100),
            "tip_msc_mm": float(tip_m), "tip_na_mm": float(tip_n),
            "tip_pct": float(tip_pct),
            "pass_A1": bool(l2 * 100 <= CRITERIA["A1_L2_T3_pct"]),
            "pass_A2": bool(mx * 100 <= CRITERIA["A2_max_T3_pct"]),
            "pass_A3": bool(tip_pct <= CRITERIA["A3_tip_pct"]),
        }
        r = results[sc]
        print(f"SC{sc}: n={r['n_common']}  L2(T3) {r['L2_T3_pct']:.2f}% "
              f"[{'PASS' if r['pass_A1'] else 'FAIL'}]  max {r['max_T3_pct']:.2f}% "
              f"[{'PASS' if r['pass_A2'] else 'FAIL'}]  tip {tip_m:.2f}/"
              f"{tip_n:.2f} mm ({r['tip_pct']:.2f}%) "
              f"[{'PASS' if r['pass_A3'] else 'FAIL'}]")

    out = {"criteria_declared": CRITERIA, "per_subcase": results}
    path = os.path.join(HERE, "r3_msc_displacement_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()

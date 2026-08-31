# 과도 로터 하중의 수치 수렴성·모델형식 민감도 연구 (r3 MC4)
"""Timestep convergence and inflow-time-constant sensitivity of the
hover-gust transient peaks (ILC-8).

Study A: integrator timestep sweep dt = 20/10/5/2.5 ms at fixed gust
parameters; convergence of the up/down peak wing-root Mx.
Study B: Pitt-Peters time-constant coefficient — uncorrected apparent
mass 4/(3*pi) (production) vs corrected L-matrix 64/(75*pi) — as a
model-form uncertainty band on the same peaks.

Usage:  python scripts/r3_rotor_convergence.py
Output: prints tables; writes r3_rotor_convergence.json next to this
script (numbers cited in Paper 1, r3 revision).
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

from nastaero.bdf.parser import parse_bdf                     # noqa: E402
from nastaero.loads_analysis.certification.aircraft_config import (  # noqa: E402
    AircraftConfig,
)
from nastaero.loads_analysis.certification.vtol_transient_loads import (  # noqa: E402
    VTOLTransientLoadsRunner,
)
from nastaero.models.ilc8 import build_ilc8, make_ilc8_vtol_config  # noqa: E402


def _make_runner(tmpdir, tau_coeff=None):
    build_ilc8(tmpdir)
    model = parse_bdf(os.path.join(tmpdir, "ilc8.bdf"))
    cfg_path = os.path.join(SOLVER, "tests", "validation", "ILC8",
                            "ilc8_cert_config.yaml")
    with open(cfg_path) as f:
        acfg = AircraftConfig.from_dict(yaml.safe_load(f))
    return VTOLTransientLoadsRunner(model, make_ilc8_vtol_config(), acfg,
                                    inflow_tau_coeff=tau_coeff)


def _peaks(results):
    out = {}
    for r in results:
        tag = "up" if "up" in r.failed_rotor_label else "down"
        out[tag] = {"peak_wing_Mx": float(r.peak_wing_Mx),
                    "qs_wing_Mx": float(r.qs_wing_Mx),
                    "daf": float(r.daf_wing_Mx)}
    return out


def main():
    import tempfile
    data = {"study_A_dt": {}, "study_B_tau": {}}

    with tempfile.TemporaryDirectory() as td:
        # ---- Study A: dt sweep ----
        for dt in (0.02, 0.01, 0.005, 0.0025):
            runner = _make_runner(td)
            res = runner.run_all_hover_gust(t_sim=3.0, dt=dt,
                                            dt_loads=0.02)
            data["study_A_dt"][f"{dt*1000:g}ms"] = _peaks(res)
            print(f"dt={dt*1000:g} ms 완료")

        # ---- Study B: tau coefficient ----
        for name, coeff in (("4/(3pi)", 4.0 / (3.0 * np.pi)),
                            ("64/(75pi)", 64.0 / (75.0 * np.pi))):
            runner = _make_runner(td, tau_coeff=coeff)
            res = runner.run_all_hover_gust(t_sim=3.0, dt=0.005,
                                            dt_loads=0.02)
            data["study_B_tau"][name] = _peaks(res)
            print(f"tau_coeff={name} 완료")

    # ---- Report ----
    print("\n=== Study A: 시간간격 수렴성 (peak wing Mx, N*m) ===")
    ref = data["study_A_dt"]["2.5ms"]
    for k, v in data["study_A_dt"].items():
        row = []
        for d in ("up", "down"):
            p = v[d]["peak_wing_Mx"]
            r0 = ref[d]["peak_wing_Mx"]
            row.append(f"{d}: {p:,.1f} ({(p/r0-1)*100:+.3f}% vs 2.5ms)")
        print(f"  dt={k:>6}  " + " | ".join(row))

    print("\n=== Study B: 시정수 계수 민감도 (dt=5ms) ===")
    a = data["study_B_tau"]["4/(3pi)"]
    b = data["study_B_tau"]["64/(75pi)"]
    for d in ("up", "down"):
        pa, pb = a[d]["peak_wing_Mx"], b[d]["peak_wing_Mx"]
        print(f"  {d:>4}: 4/(3pi) {pa:,.1f} vs 64/(75pi) {pb:,.1f} "
              f"N*m  (delta {(pb/pa-1)*100:+.2f}%)")

    out_path = os.path.join(HERE, "r3_rotor_convergence.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

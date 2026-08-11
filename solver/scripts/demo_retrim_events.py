# (고장×재트림) 확장 사건 선별 데모 — ILC-8에서 P×C 순위와 지배 패턴 출력
"""Demonstrate the (failure x re-trim) event-space extension on ILC-8.

Runs RetrimScreen: 8-rotor linear VMT basis, exhaustive command-band
sweep per failure event, consequence C = directional exceedance beyond
the discrete-taxonomy hull, adoption by P x C.

Usage:  python scripts/demo_retrim_events.py
"""
from __future__ import annotations

import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
ILC8 = os.path.join(SOLVER, "tests/validation/ILC8")
sys.path.insert(0, SOLVER)

from nastaero.bdf.parser import parse_bdf                      # noqa: E402
from nastaero.config import setup_logging                      # noqa: E402
from nastaero.loads_analysis.certification.aircraft_config import (  # noqa: E402
    AircraftConfig,
)
from nastaero.loads_analysis.certification.retrim_events import (  # noqa: E402
    RetrimScreen,
)
from nastaero.loads_analysis.component_id import (             # noqa: E402
    identify_components_manual,
)
from nastaero.models.ilc8 import make_ilc8_vtol_config         # noqa: E402


def build_components(model):
    def _nids(lo, hi, extra=()):
        ids = [n for n in model.nodes if lo <= n <= hi]
        return ids + [n for n in extra if n in model.nodes]

    WING = dict(span_axis=1, shear_axis=2, bending_axis=0, torsion_axis=1)
    return identify_components_manual(model, [
        dict(name="Right Wing", integration_sign=1.0, color="blue",
             node_ids=_nids(400000, 499999)
             + _nids(730000, 749999,
                     (990103, 990104, 990107, 990108)), **WING),
        dict(name="Left Wing", integration_sign=-1.0, color="dodgerblue",
             node_ids=_nids(300000, 399999)
             + _nids(710000, 729999,
                     (990101, 990102, 990105, 990106)), **WING),
        dict(name="Right V-Tail", integration_sign=1.0, color="red",
             node_ids=_nids(600000, 699999), **WING),
        dict(name="Left V-Tail", integration_sign=-1.0, color="salmon",
             node_ids=_nids(500000, 599999), **WING),
        dict(name="Fuselage", integration_sign=-1.0, color="gray",
             node_ids=_nids(100000, 299999, (990201,)),
             span_axis=0, shear_axis=2, bending_axis=1, torsion_axis=0),
    ])


def main() -> None:
    setup_logging("ERROR")
    t0 = time.time()
    model = parse_bdf(os.path.join(ILC8, "ilc8.bdf"))
    with open(os.path.join(ILC8, "ilc8_cert_config.yaml")) as f:
        config = AircraftConfig.from_dict(yaml.safe_load(f))
    vtol_config = make_ilc8_vtol_config()
    components = build_components(model)

    screen = RetrimScreen(model, vtol_config, config,
                          components=components, fuselage_cg_x=4450.0)
    events = screen.screen()
    print(f"screened {len(events)} (failure x re-trim) events, "
          f"{sum(e.n_patterns for e in events)} patterns "
          f"({time.time()-t0:.1f}s)\n")

    print(f"{'rank':>4s} {'mode':4s} {'rotor':22s} {'P':>8s} "
          f"{'C':>7s} {'P*C':>9s}  {'worst @':24s} pattern")
    for k, e in enumerate(events, 1):
        print(f"{k:4d} {e.mode:4s} {e.rotor_label:22s} "
              f"{e.P:8.5f} {e.consequence*100:6.2f}% "
              f"{e.risk*100:8.4f}  "
              f"{e.worst_component:12s} y={e.worst_station:7.0f}  "
              f"{'/'.join(f'{v:g}' for v in e.pattern)}")

    cases = screen.realize(events, threshold_pct=1.0, top_n=8)
    print(f"\nadopted as cases (C >= 1%, top 8): {len(cases)}")
    for c in cases:
        print(f"  C{c.case_id} [{c.far_section}] {c.label}")


if __name__ == "__main__":
    main()

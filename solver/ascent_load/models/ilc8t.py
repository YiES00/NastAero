# ILC-8T 틸트 변형 기체 — 전열 4기 틸트 나셀, 푸셔 제거 (L+C 대비 비교용 별도 기체)
"""ILC-8T: tilt-conversion variant of the ILC-8 airframe.

Same airframe (fuselage/wing/V-tail/booms/gear) as ILC-8 so that a
lift+cruise vs tilt comparison isolates the propulsion architecture:

  - the tail pusher (45 kg) is REMOVED;
  - the four FRONT rotors (x = 3 m row) become TILT rotors with
    nacelle/actuator mass (+10 kg each) at the hubs; they provide
    cruise thrust after conversion (tilt angle sigma: 0 deg = vertical
    hover, 90 deg = horizontal cruise, thrust +x);
  - the four AFT rotors remain fixed lift rotors that spin down as
    the wing takes over.

The two-pass ballast solve targets the same MTOW/CG as ILC-8, so the
comparison runs at equal weight. Deck is written separately
(ilc8t.bdf) and never touches the ILC-8 artifacts.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .ilc8 import (
    BOOM_YS, CG_X_TARGET, HUB_BASE, HUB_Z, MTOW_T, ROTOR_X_A, ROTOR_X_F,
    generate,
)

logger = logging.getLogger(__name__)

TILT_NACELLE_T = 0.010          # 틸트 나셀+액추에이터 10 kg/기


def build_ilc8t(out_dir: str) -> str:
    """2-패스 생성 (ILC-8와 동일 절차, 푸셔 제거 + 틸트 나셀 질량)."""
    from ..bdf.parser import parse_bdf
    from ..loads_analysis.trim_loads import compute_node_masses

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bdf = out / "ilc8t.bdf"

    kw = dict(pusher=False, tilt_nacelle_t=TILT_NACELLE_T,
              title="ILC-8T TILT VARIANT - FREE TRIM")
    generate(str(bdf), **kw)
    m = parse_bdf(str(bdf))
    nm = compute_node_masses(m)
    m_tot = sum(nm.values())
    cg_x = sum(mass * m.nodes[n].xyz_global[0]
               for n, mass in nm.items() if n in m.nodes) / m_tot
    dm = MTOW_T - m_tot
    # 푸셔(테일 45 kg) 제거는 CG를 전방으로 크게 옮기므로 순수 첨가
    # 밸러스트만으로 MTOW·CG를 동시에 복원할 수 없다. 전방 해는
    # 노즈 항전 그룹(x 100~1300, 도심 ~700)의 경량화(음수 질량)로
    # 구현한다 — 틸트 변형의 장비 재배치라는 물리적 해석.
    x_a, x_b = 700.0, 9400.0
    rhs = MTOW_T * CG_X_TARGET - m_tot * cg_x
    m_b = (rhs - dm * x_a) / (x_b - x_a)
    m_a = dm - m_b
    logger.info("ILC-8T pass1: m=%.4f t, cg_x=%.1f -> fwd=%.4f t "
                "(nose trim if <0), aft=%.4f t", m_tot, cg_x, m_a, m_b)
    generate(str(bdf), ballast=(max(m_a, 0.0), max(m_b, 0.0)),
             ballast_x=(x_a, x_b), nose_trim_t=min(m_a, 0.0), **kw)
    return str(bdf)


def make_ilc8t_vtol_config():
    """ILC-8T 로터 구성 — 전열 4기 TILT(순항 담당), 후열 4기 LIFT."""
    from ..rotor.airfoil import RotorAirfoil
    from ..rotor.blade import BladeDef
    from ..rotor.rotor_config import (
        RotationDir, RotorDef, RotorType, VTOLConfig,
    )

    lift_blade = BladeDef(radius=0.75, root_cutout=0.15, n_elements=20,
                          mean_chord=0.11,
                          twist_root=math.radians(12.0),
                          twist_tip=math.radians(4.0),
                          airfoil=RotorAirfoil.naca0012())
    rotors = []
    rid = 0
    for x, row in ((ROTOR_X_F, "F"), (ROTOR_X_A, "A")):
        tilt_row = (row == "F")
        for bi, y in enumerate(BOOM_YS):
            rid += 1
            rotors.append(RotorDef(
                rotor_id=rid,
                label=(f"{'Tilt' if tilt_row else 'Lift'} Rotor "
                       f"{row}{bi + 1} (y={y / 1000:+.0f}m)"),
                rotor_type=(RotorType.TILT if tilt_row
                            else RotorType.LIFT),
                hub_position=np.array([x, y, HUB_Z]),
                shaft_axis=np.array([0.0, 0.0, 1.0]),
                blade=lift_blade, n_blades=4,
                rotation_dir=(RotationDir.CW if (rid % 2)
                              else RotationDir.CCW),
                rpm_hover=2400.0,
                rpm_cruise=(2400.0 if tilt_row else 0.0),
                hub_node_id=(HUB_BASE + (1 if row == "F" else 5) + bi),
                mass_kg=25.0 + (10.0 if tilt_row else 0.0),
            ))
    return VTOLConfig(config_type="ilc8t_tilt", rotors=rotors)

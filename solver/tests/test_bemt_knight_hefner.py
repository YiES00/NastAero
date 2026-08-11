# BEMT 호버 솔버를 Knight-Hefner(NACA TN-626, 1937) 실측 데이터에 고정하는 회귀 시험
"""Regression tests pinning the BEMT hover solver to experiment.

Reference: Knight & Hefner, "Static Thrust Analysis of the Lifting
Airscrew", NACA TN-626 (1937). Untwisted constant-chord NACA 0015
rotors, R = 2.5 ft, chord 2 in, 2-5 blades, tip Re = 242,000. Data
transcribed from the report tables in
tests/validation/knight_hefner/kh_tn626_rotor_CT_CQ.csv; the report's
CT convention is twice the modern helicopter convention and is halved
on load.

Tolerances reflect the measured agreement at integration time
(working thrust range theta >= 6 deg: mean |err| 3.8%, max 13.5%
across all four rotors) with margin for platform arithmetic. Points
below 6 deg are excluded: CT is tiny there and the low-Reynolds root
fairing of the test rig is not modeled.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from nastaero.rotor.blade import BladeDef
from nastaero.rotor.airfoil import RotorAirfoil
from nastaero.rotor.bemt_solver import BEMTSolver

KH_CSV = os.path.join(os.path.dirname(__file__),
                      "validation", "knight_hefner",
                      "kh_tn626_rotor_CT_CQ.csv")

R = 0.762
CHORD = 0.0508
CUTOUT = 0.167
RPM = 960.0
RHO = 1.225


def _load(blades):
    rows = []
    with open(KH_CSV) as f:
        for r in csv.DictReader(x for x in f if not x.startswith("#")):
            if int(r["blades"]) == blades and float(r["theta_deg"]) >= 6.0:
                rows.append((float(r["theta_deg"]), float(r["CT"]) / 2.0))
    return rows


def _solver(blades):
    airfoil = RotorAirfoil(Cl_alpha=5.75, alpha_0=0.0,
                           Cd_0=0.0113, Cd_1=0.0, Cd_2=0.75)
    blade = BladeDef(radius=R, root_cutout=CUTOUT, n_elements=40,
                     mean_chord=CHORD, twist_root=0.0, twist_tip=0.0,
                     airfoil=airfoil)
    return BEMTSolver(blade, n_blades=blades)


def _ct(solver, theta_deg):
    omega = RPM * 2.0 * np.pi / 60.0
    loads = solver.solve(rpm=RPM, V_inf=0.0, rho=RHO,
                         collective_rad=np.radians(theta_deg))
    return loads.thrust / (RHO * np.pi * R ** 2 * (omega * R) ** 2)


class TestKnightHefnerValidation:
    """BEMT hover CT vs NACA TN-626 measured static thrust."""

    @pytest.mark.parametrize("blades,tol_each,tol_mean", [
        (2, 0.08, 0.05),
        (3, 0.08, 0.05),
        (4, 0.12, 0.06),
        (5, 0.17, 0.09),
    ])
    def test_ct_matches_experiment(self, blades, tol_each, tol_mean):
        rows = _load(blades)
        assert len(rows) >= 4, "expected at least 4 working-range points"
        s = _solver(blades)
        errs = []
        for theta, ct_exp in rows:
            ct = _ct(s, theta)
            err = abs(ct - ct_exp) / ct_exp
            assert err < tol_each, (
                f"B={blades} theta={theta}: CT={ct:.6f} vs "
                f"experiment {ct_exp:.6f} ({err * 100:.1f}%)")
            errs.append(err)
        assert np.mean(errs) < tol_mean

    def test_thrust_scales_with_solidity(self):
        """At fixed collective, measured CT ordering across blade count
        (solidity) must be reproduced."""
        cts = [_ct(_solver(b), 8.0) for b in (2, 3, 4, 5)]
        assert all(c2 > c1 for c1, c2 in zip(cts, cts[1:]))

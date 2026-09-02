# 패치시험 맹점 명제의 회귀 고정: 두 워시 구성의 강체·아핀 정확성과 보존 항등식
"""T3 blind-spot propositions, pinned as regression tests.

(3a) surface construction: augmented-IPS displacement AND its analytic
     slope reproduce affine fields exactly.
(3b) rotation construction: with kinematically consistent nodal data
     (theta_y = -dz/dx), wash and lever-arm displacement transfer are
     exact on affine fields too.
Corollary: G_d^T conserves total force and pitching moment for BOTH
     constructions (rigid reproduction of the displacement spline).

These are exactly the checks a standard spline V&V battery runs — and
both constructions pass them identically, while their divergence
pressures on the comparison model differ by two orders of magnitude.
The propositions therefore pin the *insensitivity* of patch-type tests,
not the health of either construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from ascent_load.aero.spline import (build_ips_spline,
                                     build_ips_spline_slope)
from ascent_load.solvers.sol144 import _fill_geff


class _MiniDofMgr:
    def __init__(self, nids):
        self.node_ids = list(nids)
        self._index = {nid: i for i, nid in enumerate(self.node_ids)}

    def get_dof(self, nid, comp):
        return self._index[nid] * 6 + (comp - 1)


def _patch(seed=42, ns=18, nb=10):
    rng = np.random.default_rng(seed)
    struct_xyz = np.column_stack([rng.uniform(0.0, 4.0, ns),
                                  rng.uniform(0.0, 7.0, ns),
                                  np.zeros(ns)])
    wash_pts = np.column_stack([rng.uniform(0.5, 3.5, nb),
                                rng.uniform(0.5, 6.5, nb),
                                np.zeros(nb)])
    force_pts = wash_pts.copy()
    force_pts[:, 0] -= 0.4
    return struct_xyz, wash_pts, force_pts


def _build_pair(struct_xyz, wash_pts, force_pts, method):
    nids = list(range(1, len(struct_xyz) + 1))
    dof_mgr = _MiniDofMgr(nids)
    n_free = 6 * len(nids)
    f_dof_index = {d: d for d in range(n_free)}
    nb = len(wash_pts)
    G_w = np.zeros((nb, n_free))
    G_d = np.zeros((nb, n_free))
    slope = (build_ips_spline_slope(struct_xyz, wash_pts)
             if method == "surface" else None)
    _fill_geff(G_w, G_d,
               build_ips_spline(struct_xyz, wash_pts),
               build_ips_spline(struct_xyz, force_pts),
               list(range(nb)), nids, force_pts, struct_xyz,
               dof_mgr, f_dof_index, G_ka_slope=slope)
    return G_w, G_d


def _u(struct_xyz, a0, a1, a2):
    n = len(struct_xyz)
    u = np.zeros(6 * n)
    u[2::6] = a0 + a1 * struct_xyz[:, 0] + a2 * struct_xyz[:, 1]
    u[4::6] = -a1          # consistent rotation: theta_y = -dz/dx
    return u


FIELDS = [(1.7, 0.0, 0.0),       # rigid translation
          (0.0, -0.02, 0.0),     # rigid pitch
          (0.0, 0.0, 0.015),     # rigid roll
          (0.9, -0.013, 0.008)]  # combined affine


@pytest.mark.parametrize("method", ["surface", "rotation"])
@pytest.mark.parametrize("a0,a1,a2", FIELDS)
def test_affine_patch_exact(method, a0, a1, a2):
    sx, wp, fp = _patch()
    G_w, G_d = _build_pair(sx, wp, fp, method)
    u = _u(sx, a0, a1, a2)
    assert np.max(np.abs(G_w @ u - a1)) < 1e-10
    disp_ref = a0 + a1 * fp[:, 0] + a2 * fp[:, 1]
    assert np.max(np.abs(G_d @ u - disp_ref)) < 1e-10


@pytest.mark.parametrize("method", ["surface", "rotation"])
def test_force_and_moment_conservation(method):
    sx, wp, fp = _patch()
    _, G_d = _build_pair(sx, wp, fp, method)
    f = np.random.default_rng(7).normal(size=len(fp))
    F = G_d.T @ f
    assert abs(F[2::6].sum() - f.sum()) < 1e-10
    m_struct = -(F[2::6] * sx[:, 0]).sum() + F[4::6].sum()
    m_aero = -(f * fp[:, 0]).sum()
    assert abs(m_struct - m_aero) < 1e-9


def test_blindspot_documented():
    """The two constructions are IDENTICAL under every check above —
    the assertion of the proposition is precisely that this battery
    carries no information about the divergence spectrum."""
    sx, wp, fp = _patch()
    errs = {}
    for method in ("surface", "rotation"):
        G_w, G_d = _build_pair(sx, wp, fp, method)
        u = _u(sx, 0.9, -0.013, 0.008)
        errs[method] = max(np.max(np.abs(G_w @ u - (-0.013))),
                           np.max(np.abs(G_d @ u - (0.9 - 0.013 * fp[:, 0]
                                                    + 0.008 * fp[:, 1]))))
    assert errs["surface"] < 1e-10 and errs["rotation"] < 1e-10

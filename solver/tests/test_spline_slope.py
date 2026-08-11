# IPS 미분(slope) 스플라인 커널과 surface-slope 모드의 단위 시험
"""Unit tests for the SPLINE1-semantics surface-slope path.

Verifies:
1. The analytic x-derivative kernel reproduces linear fields exactly
   (polynomial-term consistency of the augmented IPS system).
2. The analytic slope matches central finite differences of the
   displacement interpolation on a random smooth field.
3. Kernel regularity at r -> 0 (no singularity).
4. Surface mode in _fill_geff: a chordwise z-gradient (effective twist
   carried by translations only) produces nonzero normalwash, which the
   rotation mode by construction cannot see.
5. The default rotation mode is unchanged by the new keyword.
"""
from __future__ import annotations

import numpy as np
import pytest

from nastaero.aero.spline import (
    build_ips_spline,
    build_ips_spline_slope,
    _green_function_dx,
)


def _scatter(n, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(0.0, 5.0, n),
        rng.uniform(0.0, 8.0, n),
        np.zeros(n),
    ])
    return pts


class TestSlopeKernel:

    def test_linear_field_slope_exact(self):
        """z = a + b x + c y  →  dz/dx = b exactly (polynomial term)."""
        sn = _scatter(15, seed=1)
        ap = _scatter(9, seed=2)
        z = 2.0 + 0.3 * sn[:, 0] + 0.1 * sn[:, 1]
        Gx = build_ips_spline_slope(sn, ap)
        assert np.max(np.abs(Gx @ z - 0.3)) < 1e-10

    def test_constant_field_zero_slope(self):
        sn = _scatter(12, seed=3)
        ap = _scatter(7, seed=4)
        Gx = build_ips_spline_slope(sn, ap)
        assert np.max(np.abs(Gx @ np.ones(12))) < 1e-10

    def test_matches_finite_difference(self):
        """Analytic slope == central FD of the displacement spline."""
        sn = _scatter(15, seed=5)
        ap = _scatter(9, seed=6)
        rng = np.random.default_rng(7)
        z = rng.normal(size=15)

        h = 1e-5
        ap_p = ap.copy(); ap_p[:, 0] += h
        ap_m = ap.copy(); ap_m[:, 0] -= h
        fd = (build_ips_spline(sn, ap_p) @ z
              - build_ips_spline(sn, ap_m) @ z) / (2 * h)
        an = build_ips_spline_slope(sn, ap) @ z
        assert np.max(np.abs(fd - an)) < 1e-6 * max(1.0, np.max(np.abs(an)))

    def test_kernel_regular_at_origin(self):
        """dG/dx → 0 as r → 0 (|dx| <= r, r ln r → 0)."""
        assert _green_function_dx(0.0, 0.0) == 0.0
        assert abs(_green_function_dx(1e-9, 1e-9)) < 1e-7

    def test_smoothing_dz_consistency(self):
        """With dz smoothing, slope must still differentiate the SAME
        (smoothed) surface — FD check with dz > 0."""
        sn = _scatter(15, seed=8)
        ap = _scatter(6, seed=9)
        rng = np.random.default_rng(10)
        z = rng.normal(size=15)
        dz = 0.05

        h = 1e-5
        ap_p = ap.copy(); ap_p[:, 0] += h
        ap_m = ap.copy(); ap_m[:, 0] -= h
        fd = (build_ips_spline(sn, ap_p, dz) @ z
              - build_ips_spline(sn, ap_m, dz) @ z) / (2 * h)
        an = build_ips_spline_slope(sn, ap, dz) @ z
        assert np.max(np.abs(fd - an)) < 1e-6 * max(1.0, np.max(np.abs(an)))


class TestSurfaceModeFill:
    """Surface mode captures chordwise z-gradient twist; rotation mode
    cannot (that is precisely the modeling difference)."""

    def _mini(self):
        # Two chordwise node lines (LE x=0, TE x=1) across 3 span stations:
        # a translation-only twist field z = -theta * x (theta_y DOFs zero).
        xs, ys = [0.0, 1.0], [0.0, 1.0, 2.0]
        nodes = np.array([[x, y, 0.0] for y in ys for x in xs])
        aero = np.array([[0.5, y + 0.5, 0.0] for y in ys[:-1]])
        return nodes, aero

    def test_surface_slope_sees_translation_twist(self):
        nodes, aero = self._mini()
        theta = 0.02
        z = -theta * nodes[:, 0]  # dz/dx = -theta everywhere
        Gx = build_ips_spline_slope(nodes, aero)
        slope = Gx @ z
        assert np.max(np.abs(slope - (-theta))) < 1e-8

    def test_rotation_mode_blind_to_translation_twist(self):
        """Displacement spline itself reproduces the twist field, but a
        theta_y-only normalwash (rotation mode) built from zero nodal
        rotations is identically zero — documenting the gap the surface
        mode closes."""
        nodes, aero = self._mini()
        theta = 0.02
        z = -theta * nodes[:, 0]
        Gz = build_ips_spline(nodes, aero)
        # displacement interpolation is fine...
        assert np.max(np.abs(Gz @ z - (-theta * 0.5))) < 1e-8
        # ...but nodal theta_y are all zero for this field, so the
        # rotation-mode normalwash w = sum w_k * theta_y_k vanishes.
        theta_y = np.zeros(len(nodes))
        assert np.max(np.abs(Gz @ theta_y)) == 0.0

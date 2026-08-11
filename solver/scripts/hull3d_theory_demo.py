# 3D 헐 전용 케이스의 이론적 최대 초과율 — 순수 메커니즘(팔면체) + 대각 조합 케이스 해석 데모
"""Theoretical demonstration: how severe can a 3-D-hull-only case be?

Model: each certification case loads the (V, M, T) triple at one wing
station. Three independent load mechanisms define the axes:

  e_V  (shear-dominant,   e.g. landing / point rotor loads)
  e_M  (bending-dominant, e.g. nz maneuver at max q)
  e_T  (torsion-dominant, e.g. rotor jam / aileron)

Pure-mechanism extremes +/-e_i are always caught by 2-D selection
(axis extremes). A COMBINED case p = a*(1,1,1) — a failure occurring
during a maneuver/gust — projects to (a, a) in every coordinate
plane, which stays strictly INSIDE the projected hulls (diamonds
|x|+|y| <= 1) as long as a < 0.5: 2-D selection cannot see it.
In 3-D it leaves the octahedron already at a > 1/3.

This script quantifies the directional exceedance (same normalized
metric as compare_hull_selection.py) as a function of a, and the
rank-1-correlation limit that explains why single-parameter case sets
(one weight, one q-sweep) show near-zero exceedance.

Usage:  python scripts/hull3d_theory_demo.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def norm_exceedance(base_pts: np.ndarray, probe: np.ndarray) -> float:
    """compare_hull_selection.exceedance와 동일 정의 (단일 스테이션)."""
    all_pts = np.vstack([base_pts, probe])
    lo = all_pts.min(axis=0)
    span = np.ptp(all_pts, axis=0)
    span[span == 0] = 1.0
    b = (base_pts - lo) / span
    p = (probe - lo) / span
    hull = ConvexHull(b)
    return float(np.max(hull.equations[:, :3] @ p + hull.equations[:, 3]))


def main() -> None:
    # ── 1. 팔면체(순수 메커니즘 6극값) + 대각 조합 케이스 ──
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                     [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    print("팔면체(순수 메커니즘 극값 6개, 2D 선정이 항상 포착) 대비")
    print("대각 조합 케이스 p = a*(1,1,1)의 방향 초과율:")
    print(f"  {'a':>6s} {'2D 가시성':14s} {'초과율':>8s}")
    for a in (0.30, 1 / 3, 0.35, 0.391, 0.45, 0.49):
        p = np.array([a, a, a])
        vis = "2D 꼭짓점" if 2 * a >= 1.0 else "2D 비가시"
        exc = norm_exceedance(octa, p) * 100
        print(f"  {a:6.3f} {vis:14s} {max(exc, 0):7.2f}%")
    print("  → a<0.5 내내 세 평면 투영에서는 내부점(2D 선정 불가),")
    print("    a=0.391에서 이미 5%, a→0.5 극한에서 14.4% 초과.")

    # 5% 경계 해석해: (3a-1)/(2*sqrt(3)) = 0.05  →  a = 0.391
    a5 = (2 * np.sqrt(3) * 0.05 + 1) / 3
    print(f"  해석해: 5% 경계 a* = {a5:.4f} "
          "(각 순수 극값의 39%만 동시에 걸리면 충분)")

    # ── 2. 랭크-1 상관 극한 — 단일 파라미터 지배 케이스 집합 ──
    rng = np.random.default_rng(7)
    t = rng.uniform(-1, 1, 200)
    for eps in (0.02, 0.05, 0.3):
        pts = np.outer(t, [1.0, 0.9, 0.8])
        pts += eps * rng.standard_normal(pts.shape)
        # 2D 선정 흉내: 세 좌표평면 헐 꼭짓점의 합집합
        sel = set()
        for i, j in ((0, 1), (0, 2), (1, 2)):
            h = ConvexHull(pts[:, [i, j]])
            sel.update(h.vertices.tolist())
        base = pts[sorted(sel)]
        worst = 0.0
        for k in range(len(pts)):
            if k in sel:
                continue
            worst = max(worst, norm_exceedance(base, pts[k]))
        print(f"  랭크-1 + 노이즈 eps={eps:4.2f}: 2D 선정 "
              f"{len(sel):3d}/200, 비선정 최대 초과 {worst*100:5.2f}%")
    print("  → V/M/T가 한 파라미터(q·nz)로 함께 움직이면 2D 선정이")
    print("    3D 헐을 사실상 재현 — 기준 매트릭스의 ≤0.8%가 이 극한.")


if __name__ == "__main__":
    main()

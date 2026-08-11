# 트림 공력 박스 힘 → 외피 요소 압력(PLOAD4) 내보내기 — 외피 패널 국부 설계용 보조 산출물
"""Map trim aero box forces onto structural skin elements as PLOAD4.

Role separation (deliberate): the FORCE-card path remains the loads
deliverable for section loads and design cases (virtual-work
conjugate, resultant-preserving); this module provides the auxiliary
*pressure* product used for skin-panel local design, following the
established pressure-delivery practice of CFD-based loads processes.

The DLM carries one net pressure per box (constant delta-Cp), so the
mapping deliberately paints each box's NET normal force over the skin
elements beneath it -- it does not invent chordwise or upper/lower
resolution the aero data does not have. Conservation is enforced per
box: the summed mapped normal force equals the box force exactly;
boxes with no skin beneath them are reported as uncovered and their
share stays in the FORCE-card deliverable, which remains the
complete loads product. The offset band is sized per CAERO (15% of
its mean chord) so thick inboard skins are not dropped, and
continuous-curvature surfaces (boom tubes -- the fold-chain rule
shared with the spline helper) are excluded so the wider band cannot
paint lifting-surface pressure onto them. ILC-8 with defaults:
248/248 boxes covered, 748 skin elements, resultant residual 0.19%.

Mapping rules per box:
- candidate skin elements: centroid within the box outline (in-plane)
  and within ``offset_tol`` of the box plane;
- alignment filter |n_elem . n_box| >= ``min_align`` excludes spar/rib
  webs automatically;
- pressure p_e = sign(n_e . n_box) * F_box / sum(A_proj), where
  A_proj = A_e |n_e . n_box| -- upper and lower skins each carry a
  share and the resultant is preserved.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SHELLS = ("CQUAD4", "CQUAD8", "CTRIA3", "CTRIA6")


def _shell_geometry(model) -> Tuple[np.ndarray, ...]:
    """모든 쉘 요소의 (eid, 도심, 법선, 면적) 배열."""
    eids: List[int] = []
    cents: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    areas: List[float] = []
    for eid, e in model.elements.items():
        if e.type not in _SHELLS:
            continue
        nids = list(e.node_ids)[:4 if e.type.startswith("CQUAD") else 3]
        pts = []
        ok = True
        for n in nids:
            node = model.nodes.get(n)
            if node is None:
                ok = False
                break
            pts.append(np.asarray(getattr(node, "xyz_global", node.xyz),
                                  dtype=float))
        if not ok or len(pts) < 3:
            continue
        pts = np.array(pts)
        if len(pts) == 3:
            nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            area = 0.5 * np.linalg.norm(nrm)
        else:
            d1 = pts[2] - pts[0]
            d2 = pts[3] - pts[1]
            nrm = np.cross(d1, d2)
            area = 0.5 * np.linalg.norm(nrm)
        mag = np.linalg.norm(nrm)
        if mag < 1e-12 or area < 1e-12:
            continue
        eids.append(eid)
        cents.append(pts.mean(axis=0))
        normals.append(nrm / mag)
        areas.append(float(area))
    return (np.array(eids, dtype=np.int64), np.array(cents),
            np.array(normals), np.array(areas))


def _box_offset_tols(model, boxes, frac: float = 0.15) -> np.ndarray:
    """박스별 면외 허용치 — 각 박스가 속한 CAERO의 평균 시위 기준.

    전 CAERO 평균을 쓰면 V-테일의 작은 시위가 밴드를 끌어내려 두꺼운
    날개 안쪽 구간의 윗면 외피가 탈락한다(ILC-8에서 확인). 박스 ID
    범위로 소속 CAERO를 찾아 시위별 밴드를 준다.
    """
    caeros = getattr(model, "caero_panels", {}) or {}
    ranges = []
    for eid in sorted(caeros):
        c = caeros[eid]
        n = max(c.nspan, 1) * max(c.nchord, 1)
        ranges.append((eid, eid + n, frac * (c.chord1 + c.chord4) / 2.0))
    default = frac * (np.mean([(c.chord1 + c.chord4) / 2.0
                               for c in caeros.values()]) if caeros else 1.0)
    tols = np.full(len(boxes), default)
    for i, b in enumerate(boxes):
        for lo, hi, t in ranges:
            if lo <= b.box_id < hi:
                tols[i] = t
                break
    return tols


def map_box_forces_to_skin(model, boxes, box_forces: np.ndarray,
                           offset_tol: float = 0.0,
                           margin_frac: float = 0.02,
                           min_align: float = 0.3,
                           exclude_curved: bool = True,
                           pids=None,
                           ) -> Tuple[Dict[int, float], Dict]:
    """박스 법선 힘을 외피 요소 압력으로 분배 (박스별 보존).

    Parameters
    ----------
    offset_tol : float
        면외 허용치 (>0이면 전 박스 공통 강제, 0이면 박스별 자동 =
        소속 CAERO 평균 시위의 15%).
    exclude_curved : bool
        연속 곡면(붐 튜브 등) 요소를 제외 (스플라인 도우미와 같은
        곡률 체인 판정). 밴드 안에 들어온 튜브 페이싯에 양력면
        압력이 칠해지는 것을 막는다.
    pids : set of int, optional
        지정 시 해당 속성의 요소만 후보 (모델러 직접 제어, 휴리스틱
        보다 우선).

    Returns
    -------
    pressures : {eid: p}  (양수 = 요소 법선 방향, N/단위면적)
    report : dict — n_boxes, n_covered, force_in(3,), force_mapped(3,),
        residual_pct(법선 합력 기준), uncovered(box_id 목록)
    """
    eids, cents, normals, areas = _shell_geometry(model)
    if len(eids) == 0:
        return {}, {"n_boxes": len(boxes), "n_covered": 0,
                    "uncovered": [b.box_id for b in boxes]}

    keep = np.ones(len(eids), dtype=bool)
    if pids:
        pid_set = {int(p) for p in pids}
        keep &= np.array([getattr(model.elements[int(e)], "pid", None)
                          in pid_set for e in eids])
    elif exclude_curved:
        from ..aero.panel_authoring import curved_surface_node_ids

        curved = curved_surface_node_ids(model)
        if curved:
            keep &= np.array([
                not all(n in curved for n in
                        (getattr(model.elements[int(e)], "node_ids", [])
                         or [])[:4])
                for e in eids])
    if not np.all(keep):
        eids, cents = eids[keep], cents[keep]
        normals, areas = normals[keep], areas[keep]
    if len(eids) == 0:
        return {}, {"n_boxes": len(boxes), "n_covered": 0,
                    "uncovered": [b.box_id for b in boxes]}

    if offset_tol > 0.0:
        box_tols = np.full(len(boxes), float(offset_tol))
    else:
        box_tols = _box_offset_tols(model, boxes)

    pressures: Dict[int, float] = {}
    uncovered: List[int] = []
    force_mapped = np.zeros(3)
    force_in = np.zeros(3)

    for i, box in enumerate(boxes):
        F = np.asarray(box_forces[i], dtype=float)
        force_in += F
        n_box = box.normal
        w = float(np.dot(F, n_box))          # 법선 방향 순힘 (부호 포함)
        c = box.corners                      # (4,3)
        origin = c[0]
        # 면내 2D 기저 (u: 첫 변, v: 법선×u)
        u = c[1] - c[0]
        umag = np.linalg.norm(u)
        if umag < 1e-12:
            uncovered.append(box.box_id)
            continue
        u = u / umag
        v = np.cross(n_box, u)

        rel = cents - origin
        off = rel @ n_box
        cand = np.abs(off) <= box_tols[i]
        if not np.any(cand):
            uncovered.append(box.box_id)
            continue
        # 정렬도 필터 (웹 제외)
        align = normals[cand] @ n_box
        cand_idx = np.where(cand)[0][np.abs(align) >= min_align]
        if len(cand_idx) == 0:
            uncovered.append(box.box_id)
            continue
        # 볼록 사각형 내부 판정 (여유 margin_frac 확장)
        quad2d = np.array([[np.dot(p - origin, u), np.dot(p - origin, v)]
                           for p in c])
        centroid2d = quad2d.mean(axis=0)
        quad2d = centroid2d + (quad2d - centroid2d) * (1.0 + margin_frac)
        pts2d = np.column_stack([rel[cand_idx] @ u, rel[cand_idx] @ v])
        # 기준 부호는 사각형 감김 방향에서 (후보점과 무관하게) 결정
        e1 = quad2d[1] - quad2d[0]
        e2 = quad2d[2] - quad2d[1]
        ref_sign = np.sign(e1[0] * e2[1] - e1[1] * e2[0]) or 1.0
        inside = np.ones(len(cand_idx), dtype=bool)
        for k in range(4):
            a, b = quad2d[k], quad2d[(k + 1) % 4]
            cr = ((b[0] - a[0]) * (pts2d[:, 1] - a[1])
                  - (b[1] - a[1]) * (pts2d[:, 0] - a[0]))
            inside &= (cr * ref_sign) >= -1e-12
        sel = cand_idx[inside]
        if len(sel) == 0:
            uncovered.append(box.box_id)
            continue

        a_dot = normals[sel] @ n_box
        A_proj = areas[sel] * np.abs(a_dot)
        total_A = float(A_proj.sum())
        if total_A < 1e-12:
            uncovered.append(box.box_id)
            continue
        p_base = w / total_A
        for j, gid in enumerate(sel):
            p_e = p_base * np.sign(a_dot[j])
            eid = int(eids[gid])
            pressures[eid] = pressures.get(eid, 0.0) + float(p_e)
            force_mapped += p_e * areas[gid] * normals[gid]

    w_in = float(np.linalg.norm(force_in))
    resid = (float(np.linalg.norm(force_in - force_mapped)) / w_in * 100.0
             if w_in > 1e-12 else 0.0)
    report = {
        "n_boxes": len(boxes),
        "n_covered": len(boxes) - len(uncovered),
        "uncovered": uncovered,
        "force_in": force_in,
        "force_mapped": force_mapped,
        "residual_pct": resid,
    }
    logger.info("PLOAD4 mapping: %d/%d boxes covered, %d skin elements, "
                "resultant residual %.2f%%",
                report["n_covered"], len(boxes), len(pressures), resid)
    return pressures, report


def write_pload4_cards(pressures: Dict[int, float], filepath: str,
                       load_sid: int, label: str = "",
                       report: Dict = None, append: bool = False) -> None:
    """PLOAD4 큰필드 카드 작성 (요소당 균일 압력, 요소 법선 방향 양)."""
    mode = "a" if append else "w"
    with open(filepath, mode) as f:
        f.write(f"$ SKIN PRESSURES (PLOAD4) — {label}\n")
        f.write("$ Auxiliary product for skin-panel local design; the\n")
        f.write("$ FORCE decks remain the section-loads deliverable.\n")
        if report:
            fi, fm = report["force_in"], report["force_mapped"]
            f.write(f"$ boxes covered: {report['n_covered']}"
                    f"/{report['n_boxes']}  resultant residual: "
                    f"{report['residual_pct']:.2f}%\n")
            f.write(f"$ Fz in / mapped: {fi[2]:.4E} / {fm[2]:.4E}\n")
        for eid in sorted(pressures):
            p = pressures[eid]
            if abs(p) < 1e-20:
                continue
            f.write("PLOAD4* %16d%16d%16.8E%16s\n*\n"
                    % (load_sid, eid, p, ""))
    logger.info("  PLOAD4 cards written to: %s", filepath)


def export_pload4(model, boxes, results, filepath: str,
                  offset_tol: float = 0.0) -> Dict:
    """전 서브케이스의 트림 박스 힘을 PLOAD4 덱 하나로 내보낸다.

    서브케이스별 SID = subcase_id. 반환값은 subcase_id → report.
    """
    reports = {}
    first = True
    for sc in results.subcases:
        if sc.aero_forces is None:
            continue
        pressures, report = map_box_forces_to_skin(
            model, boxes, sc.aero_forces, offset_tol=offset_tol)
        write_pload4_cards(pressures, filepath, load_sid=sc.subcase_id,
                           label=f"SUBCASE {sc.subcase_id}",
                           report=report, append=not first)
        reports[sc.subcase_id] = report
        first = False
    return reports

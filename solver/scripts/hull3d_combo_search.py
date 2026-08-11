# 로터 추력 패턴 공간 전수탐색 — 2D 선정이 못 보는 3D 헐 전용 ≥5% 케이스의 구성적 탐색
"""Constructive search for a >= 5 % 3-D-hull-only case, per the
maneuver-sweep -> VMT-sensitivity -> in-envelope-combination protocol.

Key physical fact: on ILC-8 the wing has no aileron AESURF; every
independent wing V/M/T mechanism is a per-rotor thrust pattern
(command differentials, gust response, OEI/jam/runaway). Hover-family
cases need no stiffness solve (nodal forces -> VMT integration), and
the whole map T-pattern -> VMT is LINEAR (incl. inertia and
inertia-relief closure). So:

  1. build 8 basis VMT curve sets (one per rotor: hub 6-vec at hover
     thrust + its 1/8 inertia share + relief), validated against the
     pipeline-solved hover/jam cases;
  2. enumerate the full thrust-pattern grid l_i in {0, 0.7, 1.0,
     1.3, 1.5} x T_hover (failure states 0/1.5, command band
     0.7..1.3), keep patterns with nz = mean(l) in [0.6, 1.5];
  3. score each pattern with the directional-exceedance metric vs the
     3-D hull of the 2-D-selected set, REQUIRING the pattern to stay
     pairwise-interior (inside all three coordinate-plane hulls of
     the full matrix at every station — else 2-D selection would
     catch it and it would not be a 3-D-only case);
  4. realize the top patterns as real pipeline cases (exact force
     assembly + relief + VMT integrator), append to the matrix,
     rerun 2-D vs 3-D selection end-to-end, and report the confirmed
     3-D-only exceedances.

Usage:  python scripts/hull3d_combo_search.py [MTOW|LIGHT_AFT]
Output: prints the search; writes hull3d_combo_results_<ctx>.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, SOLVER)
sys.path.insert(0, HERE)

from nastaero.config import setup_logging                      # noqa: E402
from nastaero.loads_analysis.certification.batch_runner import (  # noqa: E402
    CaseResult,
)
from nastaero.loads_analysis.certification.vmt_bridge import (  # noqa: E402
    compute_vmt_for_batch,
)
from nastaero.loads_analysis.trim_loads import (               # noqa: E402
    apply_inertia_relief, compute_nodal_inertial_forces,
    compute_node_masses,
)

from compare_hull_selection import run_selection               # noqa: E402
from hull3d_severity_search import (                           # noqa: E402
    BatchResult, build_components, exceedance_all, run_variant,
)

LEVELS = np.array([0.0, 0.7, 1.0, 1.3, 1.5])
NZ_MIN, NZ_MAX = 0.6, 1.5
TOP_K = 12


def classify(pattern: np.ndarray) -> str:
    """패턴의 방어 가능성 등급 — 비정상(0 또는 1.5) 로터 수 기준."""
    abnormal = int(np.sum((pattern == 0.0) | (pattern == 1.5)))
    if abnormal == 0:
        return "command"          # 정상 지령 대역(0.7~1.3)만의 조합
    if abnormal == 1:
        return "single-failure"
    if abnormal == 2:
        return "dual-failure"
    return f"multi-failure({abnormal})"


def hover_nodal_forces(model, hub_vecs, pattern, node_masses, cg, g):
    """추력 패턴 -> 절점하중 (로터 6분력 스케일 + 관성 + relief)."""
    nz = float(np.mean(pattern))          # 모든 축이 +z 호버 로터
    forces = {}
    for (nid, vec), li in zip(hub_vecs, pattern):
        if abs(li) < 1e-12:
            continue
        forces[nid] = forces.get(nid, np.zeros(6)) + li * vec
    inertial = compute_nodal_inertial_forces(model, nz, g)
    for nid, f in inertial.items():
        forces[nid] = forces.get(nid, np.zeros(6)) + f
    apply_inertia_relief(model, {}, forces, cg=cg, g=g)
    return forces, nz


def vmt_to_mat(vmt_case, comps):
    """케이스 VMT dict -> (n_comp*n_sta, 3) 행렬."""
    rows = []
    for comp in comps:
        d = vmt_case[comp]
        rows.append(np.column_stack([d["shear"], d["bending"],
                                     d["torsion"]]))
    return np.vstack(rows)


def main() -> None:
    setup_logging("ERROR")
    os.chdir(SOLVER)
    ctx = sys.argv[1] if len(sys.argv) > 1 else "MTOW"
    dm, cgs = (0.0, 0.0) if ctx == "MTOW" else (-400.0, 150.0)

    # ── 1. 기준 매트릭스 (고장·조합 포함 246케이스) ──
    t0 = time.time()
    batch, vmt, meta = run_variant(ctx, dm, cgs, [0.0], combos=True)
    model = None            # run_variant 내부 모델 재사용 위해 재파싱
    from nastaero.bdf.parser import parse_bdf
    from hull3d_severity_search import adjust_fuselage_masses, ILC8
    model = parse_bdf(os.path.join(ILC8, "ilc8.bdf"))
    mass_kg, cg_x = adjust_fuselage_masses(model, dm, cgs)
    components = build_components(model)

    labels = {c.case_id: (c.label, c.category) for c in batch.case_results}
    proc2, dc2 = run_selection(batch, vmt, "2d")
    s2 = {d.case_id for d in dc2}
    print(f"[{ctx}] base matrix {len(vmt)} cases, 2D-selected {len(s2)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── 2. 로터 기저 (호버 1.0g 케이스의 허브 6분력) ──
    from nastaero.models.ilc8 import make_ilc8_vtol_config
    vtol_config = make_ilc8_vtol_config()
    hover_case = next(c for c in meta.values()
                      if getattr(c, "rotor_forces", None)
                      and "Hover 1.0g" in (c.trim_condition.label
                                           if c.trim_condition else ""))
    rotors = list(vtol_config.hover_rotors)
    hub_vecs = [(r.hub_node_id,
                 np.array(hover_case.rotor_forces[r.hub_node_id], float))
                for r in rotors]
    node_masses = compute_node_masses(model)
    total_mass = sum(node_masses.values())
    cg = np.zeros(3)
    for nid, m in node_masses.items():
        cg += m * model.nodes[nid].xyz_global
    cg /= total_mass
    g = 9806.65                                  # mm/s^2
    W = total_mass * g

    basis_batch = BatchResult()
    for i in range(len(rotors)):
        pat = np.zeros(len(rotors))
        pat[i] = 1.0
        forces, nz = hover_nodal_forces(model, hub_vecs, pat,
                                        node_masses, cg, g)
        basis_batch.case_results.append(CaseResult(
            case_id=900000 + i, category="basis", converged=True,
            nodal_forces=forces, nz=nz, label=f"basis R{i+1}"))
        basis_batch.completed_ids.add(900000 + i)
    basis_vmt = compute_vmt_for_batch(model, basis_batch,
                                      components=components,
                                      fuselage_cg_x=cg_x)
    comps = sorted(next(iter(basis_vmt.values())).keys())
    B = np.stack([vmt_to_mat(basis_vmt[900000 + i], comps)
                  for i in range(len(rotors))])      # (8, S, 3)
    n_sta_total = B.shape[1]
    print(f"basis built: {B.shape} comps={comps}", flush=True)

    # 검증: l=1 전체 패턴 == 파이프라인 Hover 1.0g VMT
    hover_id = next(cid for cid, (lab, cat) in labels.items()
                    if "Hover 1.0g" in lab)
    ref = vmt_to_mat(vmt[hover_id], comps)
    rec = B.sum(axis=0)
    denom = np.abs(ref).max()
    err = np.abs(rec - ref).max() / denom
    print(f"linearity check vs solved Hover 1.0g: max rel err "
          f"{err*100:.3f}%", flush=True)

    # ── 3. 패턴 전수 그리드 ──
    grids = np.meshgrid(*([LEVELS] * len(rotors)), indexing="ij")
    P = np.stack([gg.ravel() for gg in grids], axis=1)     # (N, 8)
    nz_all = P.mean(axis=1)
    P = P[(nz_all >= NZ_MIN) & (nz_all <= NZ_MAX)]
    print(f"patterns after nz filter: {len(P)}", flush=True)

    # 기준 매트릭스 전 케이스의 VMT 행렬 (S, 3) 스택
    base_ids = [cid for cid in vmt]
    base_mat = np.stack([vmt_to_mat(vmt[cid], comps) for cid in base_ids])
    sel_rows = [k for k, cid in enumerate(base_ids) if cid in s2]

    # ── 4. 스테이션별 평가: 짝지어 내부 & 3D 초과 ──
    # (V_pat 전체 저장은 수백 MB — 스테이션별 P@B로 온더플라이 계산.
    #  정규화 스팬은 기준 매트릭스 기준: 최종 검증 지표와 일치.)
    from scipy.spatial import ConvexHull

    N = len(P)
    interior = np.ones(N, bool)
    score = np.zeros(N)
    score_sta = np.zeros(N, int)
    for si in range(n_sta_total):
        pts = base_mat[:, si, :]
        pr = P @ B[:, si, :]                               # (N, 3)
        lo = pts.min(axis=0)
        span = np.ptp(pts, axis=0)
        span[span == 0] = 1.0
        q = (pts - lo) / span
        qp = (pr - lo) / span
        # 짝지어(2D) 내부성 — 전 매트릭스 헐 기준
        for i, j in ((0, 1), (0, 2), (1, 2)):
            try:
                h = ConvexHull(q[:, [i, j]])
            except Exception:
                continue
            viol = (qp[:, [i, j]] @ h.equations[:, :2].T
                    + h.equations[:, 2]).max(axis=1)
            interior &= viol <= 1e-3
        # 3D 초과 — 2D 선정 세트 헐 기준
        try:
            h3 = ConvexHull(q[sel_rows])
        except Exception:
            continue
        viol3 = (qp @ h3.equations[:, :3].T + h3.equations[:, 3]).max(axis=1)
        upd = viol3 > score
        score_sta[upd] = si
        score = np.maximum(score, viol3)
    score_masked = np.where(interior, score, -1.0)
    order = np.argsort(-score_masked)
    print(f"pairwise-interior patterns: {int(interior.sum())}/{N}",
          flush=True)
    print("\n상위 후보 (짝지어 내부 & 3D 초과):", flush=True)
    shown = 0
    for k in order:
        if score_masked[k] <= 0 or shown >= 20:
            break
        pat = P[k]
        print(f"  exc={score[k]*100:6.2f}%  nz={pat.mean():.2f}  "
              f"cls={classify(pat):18s}  l={pat.tolist()}")
        shown += 1

    # ── 5. 후보별 개별 검증 — "매트릭스 + 이 케이스 1개"가 올바른
    #       질문: 함께 추가하면 후보끼리 서로를 2D 꼭짓점으로 밀어
    #       올려 상호 커버함(1차 실험에서 확인). 등급별 최고 후보 포함.
    chosen, seen_sig = [], set()
    best_by_class = {}
    for k in order:
        if score_masked[k] <= 0:
            break
        cls = classify(P[k])
        if cls not in best_by_class:
            best_by_class[cls] = k
    for k in order:
        if score_masked[k] <= 0 or len(chosen) >= TOP_K:
            break
        sig = tuple((P[k] == 0.0) | (P[k] == 1.5))
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        chosen.append(k)
    for cls, k in best_by_class.items():
        if k not in chosen:
            chosen.append(k)
    print(f"\nvalidating {len(chosen)} patterns individually "
          f"(matrix+1 each)...", flush=True)

    # 스테이션 좌표 매핑 (플랫 인덱스 -> comp, station)
    sta_map = []
    for comp in comps:
        st = basis_vmt[900000][comp]["stations"]
        sta_map.extend((comp, float(s)) for s in st)

    results = []
    for m, k in enumerate(chosen):
        pat = P[k]
        forces, nz = hover_nodal_forces(model, hub_vecs, pat,
                                        node_masses, cg, g)
        cid = 950000 + m
        lab = ("RotorPattern " + "/".join(f"{v:g}" for v in pat)
               + f" nz={nz:.2f}")
        cr = CaseResult(case_id=cid, category="vtol_combined_cmd",
                        far_section="SC-VTOL.2135/2150",
                        converged=True, nodal_forces=forces, nz=nz,
                        label=lab)
        one = BatchResult()
        one.case_results = list(batch.case_results) + [cr]
        one.completed_ids = set(batch.completed_ids) | {cid}
        mini = BatchResult()
        mini.case_results = [cr]
        mini.completed_ids = {cid}
        vmt_one = dict(vmt)
        vmt_one.update(compute_vmt_for_batch(
            model, mini, components=components, fuselage_cg_x=cg_x))
        _, dc2f = run_selection(one, vmt_one, "2d")
        _, dc3f = run_selection(one, vmt_one, "3d")
        s2f = {d.case_id for d in dc2f}
        s3f = {d.case_id for d in dc3f}
        in2 = cid in s2f
        in3 = cid in s3f
        exc = exceedance_all(vmt_one, s2f).get(cid, 0.0) * 100
        comp_w, sta_w = sta_map[score_sta[k]]
        status = ("3D-only" if (in3 and not in2) else
                  "2D-caught" if in2 else "not-selected")
        flag = "*** >=5% ***" if (exc >= 5.0 and in3 and not in2) else ""
        print(f"  [{m:2d}] exc={exc:6.2f}%  {status:12s} "
              f"cls={classify(pat):18s} @{comp_w} y={sta_w:.0f} "
              f"{flag} l={pat.tolist()}", flush=True)
        results.append({
            "pattern": pat.tolist(), "nz": round(float(nz), 3),
            "class": classify(pat), "search_pct":
                round(float(score[k]) * 100, 2),
            "validated_pct": round(float(exc), 2),
            "status": status, "worst_component": comp_w,
            "worst_station": sta_w})
    rows = results

    with open(os.path.join(HERE,
              f"hull3d_combo_results_{ctx}.json"), "w") as f:
        json.dump({"context": ctx, "n_patterns": int(N),
                   "n_interior": int(interior.sum()),
                   "top_search": [
                       {"pattern": P[k].tolist(),
                        "exceedance_pct": round(float(score[k]) * 100, 2),
                        "class": classify(P[k])}
                       for k in order[:20] if score_masked[k] > 0],
                   "validated": rows}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved: hull3d_combo_results_{ctx}.json")


if __name__ == "__main__":
    main()

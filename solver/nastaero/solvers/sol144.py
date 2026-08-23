"""SOL 144 - Static Aeroelastic Trim Analysis.

Solves the coupled structural-aerodynamic system for trim equilibrium.

The VLM (Vortex-Lattice Method, steady k=0) approach:
1. Build structural stiffness matrix K
2. Generate aerodynamic panels from CAERO1 cards
3. Build AIC matrix D such that {w/V} = [D]{gamma}  (gamma = Gamma/V)
4. Build spline + DOF coupling matrix G_eff (per-spline mapping)
5. Form aero-structural coupling Q_aa in structural DOF space
6. Set up trim equations with force/moment balance constraints
7. Solve coupled system for displacements and trim variables

Key normalization: gamma = Gamma/V (normalized circulation), so
  delta_Cp_j = 2 * gamma_j / chord_j
  F_j = q * delta_Cp_j * area_j

Full aircraft support:
- Multiple CAERO1 panels (wing, tail, etc.)
- Per-spline structural-aero coupling (SPLINE1/SPLINE2 per surface)
- AESURF/AELIST control surface definition
- AELINK control surface coupling (V-tail, differential aileron)
- Control surface normalwash applied only to specific boxes
- Z-force AND pitch moment trim constraints
- CG-based moment reference point

Parallel support:
- TrimSharedData extracts Mach-independent pre-computations (1x)
- Each subcase solves independently (embarrassingly parallel)
- multiprocessing.Pool for CPU-parallel execution
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from ..fem.model import FEModel
from ..fem.dof_manager import DOFManager
from ..bdf.model import BDFModel, Subcase
from ..aero.panel import generate_all_panels, get_box_index_map, AeroBox
from ..aero.dlm import build_aic_matrix, compute_aero_forces, circulation_to_delta_cp
from ..aero.spline import (build_ips_spline, build_ips_spline_slope,
                            build_beam_spline)
from ..output.result_data import ResultData, SubcaseResult
from ..config import logger
from typing import List, Dict, Tuple, Optional, Set, Any


# ---------------------------------------------------------------------------
# TrimSharedData: Mach-independent data shared across subcases
# ---------------------------------------------------------------------------

@dataclass
class TrimSharedData:
    """Mach-independent pre-computed data shared across all subcases.

    These quantities depend only on the structural model and spline geometry,
    NOT on the Mach number or dynamic pressure. Computing them once saves
    significant time when solving many subcases (e.g. 7 Mach numbers).

    Spline coupling uses TWO matrices:
    - G_sp (normalwash): maps structural DOFs to normalwash (slope dz/dx).
      Only theta_y (DOF 5) contributes because slope = theta_y for beam spline.
      Used in: w_deform = G_sp @ u_f
    - G_disp (displacement): maps structural DOFs to z-displacement at aero points.
      Both z (DOF 3) and theta_y (DOF 5) contribute.
      Used in: F_struct = G_disp^T @ f_aero (force distribution)

    The aero stiffness is: Q_aa = G_disp^T @ A_jj @ G_sp
    (NOT G^T @ A_jj @ G with a single G!)
    """
    K_ff: Any = None
    F_f: Any = None
    f_dofs: Any = None
    s_dofs: Any = None
    G_sp: Any = None              # normalwash coupling (n_boxes x n_free)
    G_disp: Any = None            # displacement coupling (n_boxes x n_free)
    G_force_node: Any = None      # 물리공간 힘 분배 (n_boxes x n_nodes)
    boxes: Any = None
    box_id_to_index: Any = None
    total_weight: float = 0.0
    cg_x: float = 0.0
    cg_y: float = 0.0
    cg_z: float = 0.0
    refc: float = 0.0
    refb: float = 0.0
    refs: float = 0.0
    velocity: float = 0.0
    g: float = 9810.0
    dof_mgr: Any = None
    bdf_model: Any = None
    all_trim_labels: Any = None

    # Slave DOF dependencies from RBE2/MPC elimination
    slave_deps: Any = None        # {slave_dof: [(master_dof, coeff), ...]}

    # Pre-computed for iterative solver (large models)
    active_cols: Any = None
    G_w_active: Any = None        # G_sp active columns (normalwash)
    G_d_active: Any = None        # G_disp active columns (displacement)

    # 자유-자유 덱의 가상 확정 마운트(3-2-1) f-set 인덱스 (구속 덱이면 None).
    # 이전의 K+eps·I 인공 기초는 전 자유도 접지 스프링이 되어 유연 변형을
    # 지배했다 (ILC-8 날개 끝 3.9 mm vs MSC 283 mm) — MSC RESTRAINED
    # 정식화와 같은 결정적 마운트로 대체 (MSC SOL 144 대조로 확정).
    mount_idx: Any = None
    # f-DOF별 T3 질량 벡터 — 구조 해에 관성 하중(-nz·g·m)을 포함시켜
    # 자중 처짐/워시아웃이 공탄성 피드백에 반영되게 한다 (MSC INERTIAL 열).
    mass_t3: Any = None

    # K_eff factorization cache by (mach, q)
    # Key: (round(mach, 8), round(q, 8))
    # Value: {'D_inv': ndarray, 'A_jj': ndarray,
    #         'solve_fn': callable, 'K_eff_csc': csc_matrix}
    _solver_cache: Any = None     # initialized to {} in _build_shared_data

    # Camber normalwash correction (from airfoil profiles)
    w_camber: Any = None          # ndarray (n_boxes,) or None


# ---------------------------------------------------------------------------
# Virtual mount (free-free structural solve)
# ---------------------------------------------------------------------------

def _build_rigid_modes(bdf_model: BDFModel, dof_mgr,
                       f_dof_index: Dict[int, int], n_free: int,
                       cg: np.ndarray) -> np.ndarray:
    """f-set 위 강체모드 6열 (CG 기준 병진 3 + 회전 3)."""
    D = np.zeros((n_free, 6))
    for nid, node in bdf_model.nodes.items():
        d = np.asarray(node.xyz_global, dtype=float) - cg
        for c in (1, 2, 3):
            j = f_dof_index.get(dof_mgr.get_dof(nid, c))
            if j is None:
                continue
            D[j, c - 1] = 1.0
            # 회전 모드의 병진 성분: u = omega x (r - cg)
            if c == 1:    # ux: +wy*dz - wz*dy
                D[j, 4] = d[2]; D[j, 5] = -d[1]
            elif c == 2:  # uy: +wz*dx - wx*dz
                D[j, 5] = d[0]; D[j, 3] = -d[2]
            else:         # uz: +wx*dy - wy*dx
                D[j, 3] = d[1]; D[j, 4] = -d[0]
        for c in (4, 5, 6):
            j = f_dof_index.get(dof_mgr.get_dof(nid, c))
            if j is not None:
                D[j, c - 1] = 1.0
    return D


def _has_free_rigid_modes(K_ff, D: np.ndarray) -> bool:
    """강체모드 잔차 검사: 어느 모드든 ||K·d||가 미소하면 자유-자유.

    덱 SPC 개수 휴리스틱은 국소 더미 구속(예: GACOMP의 계기 절점)이 많은
    산업 덱에서 오판하므로, K가 실제로 강체모드를 지지하는지 직접 잰다.
    """
    K = K_ff if sp.issparse(K_ff) else sp.csc_matrix(K_ff)
    diag = np.abs(K.diagonal())
    scale = np.mean(diag[diag > 0]) if np.any(diag > 0) else 1.0
    for k in range(6):
        d = D[:, k]
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            continue
        r = np.linalg.norm(K @ d) / (scale * nd)
        if r < 1e-6:
            return True
    return False


def _suport_mount_idx(bdf_model: BDFModel, dof_mgr,
                      f_dof_index: Dict[int, int]) -> Optional[np.ndarray]:
    """덱의 SUPORT 자유도를 자유-자유 기준(마운트)으로 변환한다.

    변위는 마운트 기준 상대 변형이므로, 기준점이 다르면 강체 자세와
    탄성 변형의 분해가 달라진다. 다른 솔버와 대조할 때 덱이 SUPORT로
    기준을 명시했다면 그것을 그대로 써야 비교가 성립한다.

    f-set에 없는 자유도(이미 SPC로 구속된 경우 등)는 건너뛴다.
    """
    if not bdf_model.suports:
        return None
    idx: list = []
    nodes: list = []
    skipped: list = []
    for sup in bdf_model.suports:
        for nid, comp in sup.entries:
            for c in comp:
                if c not in "123456":
                    continue
                d = dof_mgr.get_dof(nid, int(c))
                j = f_dof_index.get(d)
                if j is None:
                    skipped.append((nid, int(c)))
                elif j not in idx:
                    idx.append(j)
            nodes.append(nid)
    if not idx:
        logger.warning("  SUPORT DOFs are not in the free set — "
                       "falling back to the automatic virtual mount")
        return None
    if skipped:
        logger.warning("  SUPORT: %d DOF(s) already constrained, skipped: %s",
                       len(skipped), skipped[:6])
    if len(idx) != 6:
        logger.warning("  SUPORT defines %d DOF(s), not the 6 needed to fix "
                       "rigid-body motion; solving with them as given",
                       len(idx))
    logger.info("  Deck SUPORT mount: nodes %s (%d DOFs)",
                "/".join(str(n) for n in nodes), len(idx))
    return np.asarray(idx, dtype=int)


def _select_virtual_mount(bdf_model: BDFModel, dof_mgr,
                          f_dof_index: Dict[int, int],
                          cg: np.ndarray) -> Optional[np.ndarray]:
    """자유-자유 덱에 대한 가상 확정 마운트(3-2-1) f-set 인덱스 선정.

    n1(CG 최근접): T1+T2+T3, n2(n1에서 x-거리 최대): T2+T3,
    n3(n1-n2 축에서 수직거리 최대): T3 — 강체 6자유도가 비공선 3점의
    병진으로 지지된다. 하중이 평형(트림+릴리프)이면 마운트 반력은 ~0이고
    변위는 마운트 기준 상대 변형(MSC SUPORT-상대 변위와 동렬)이다.
    """
    cand = []
    for nid, node in bdf_model.nodes.items():
        dofs = [dof_mgr.get_dof(nid, c) for c in (1, 2, 3)]
        if all(d in f_dof_index for d in dofs):
            cand.append((nid, np.asarray(node.xyz_global, dtype=float), dofs))
    if len(cand) < 3:
        return None

    # 후보를 CG 근방(주 구조부)으로 제한 — 기하 극값을 그대로 좇으면
    # 날개/스텁 끝단이 뽑혀 유연 부재를 핀으로 누르게 된다. MSC 덱의
    # 검증된 패턴(CG 인접 동체 프레임 3점 클러스터)을 따른다.
    all_xyz = np.array([c[1] for c in cand])
    order = np.argsort(np.linalg.norm(all_xyz - cg, axis=1))
    # CG 최근접 소수(주 구조부)만 후보로 — 풀이 크면 소형 모델에서
    # 날개/부속 끝단이 뽑혀 유연 부재를 핀으로 누른다
    pool_n = min(max(12, len(cand) // 20), len(cand))
    while True:
        pool = [cand[int(k)] for k in order[:pool_n]]
        xyz = np.array([c[1] for c in pool])
        i1 = 0  # CG 최근접
        i2 = int(np.argmax(np.abs(xyz[:, 0] - xyz[i1, 0])))
        axis = xyz[i2] - xyz[i1]
        axis_n = axis / max(np.linalg.norm(axis), 1e-12)
        rel = xyz - xyz[i1]
        perp = np.linalg.norm(rel - np.outer(rel @ axis_n, axis_n), axis=1)
        i3 = int(np.argmax(perp))
        if (np.linalg.norm(axis) > 1e-6 and perp[i3] > 1e-6):
            break
        if pool_n >= len(cand):
            return None
        pool_n = min(pool_n * 2, len(cand))
    cand = pool

    n1, n2, n3 = cand[i1], cand[i2], cand[i3]
    idx = [f_dof_index[d] for d in n1[2]]          # T1 T2 T3
    idx += [f_dof_index[n2[2][1]], f_dof_index[n2[2][2]]]  # T2 T3
    idx += [f_dof_index[n3[2][2]]]                 # T3
    logger.info("  Virtual mount (3-2-1): nodes %d/%d/%d",
                n1[0], n2[0], n3[0])
    return np.asarray(idx, dtype=int)


def _build_mass_t3(bdf_model: BDFModel, dof_mgr,
                   f_dof_index: Dict[int, int], n_free: int) -> np.ndarray:
    """f-DOF 벡터로 표현한 절점 T3 질량 (관성 하중 -nz·g·m 용)."""
    from ..loads_analysis.trim_loads import compute_node_masses
    m = np.zeros(n_free)
    for nid, mass in compute_node_masses(bdf_model).items():
        d = dof_mgr.get_dof(nid, 3)
        j = f_dof_index.get(d)
        if j is not None:
            m[j] = mass
    return m


def _apply_virtual_mount(K_csc, mount_idx: np.ndarray):
    """K(희소)에 마운트 자유도 행/열 소거 + 단위 대각(평균 스케일) 적용."""
    diag = np.abs(K_csc.diagonal())
    scale = np.mean(diag[diag > 0]) if np.any(diag > 0) else 1.0
    K = K_csc.tolil()
    for i in mount_idx:
        K.rows[i] = [i]
        K.data[i] = [scale]
    K = K.tocsc()
    # 열 소거 (대칭 위치): 마운트 열의 비대각 성분 제거
    mask = np.ones(K.shape[0], dtype=bool)
    mask[mount_idx] = False
    K_t = K.T.tolil()
    for i in mount_idx:
        K_t.rows[i] = [i]
        K_t.data[i] = [scale]
    return K_t.T.tocsc()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_trim(bdf_model: BDFModel, n_workers: int = 0,
               blas_threads: int = 1,
               airfoil_config=None,
               spline_slope_method: str = 'surface',
               apply_w2gj: bool = True) -> ResultData:
    """Run SOL 144 static aeroelastic trim analysis.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model.
    n_workers : int
        Number of parallel workers.
        0 = sequential (default), -1 = auto, >0 = explicit.
    blas_threads : int
        BLAS threads per worker (default 1 to avoid oversubscription).
    airfoil_config : PanelAirfoilConfig, optional
        Airfoil camber assignment for DLM panels.  When provided,
        camber normalwash corrections (dz_c/dx) are applied to each
        panel's boundary condition, equivalent to NASTRAN's W2GJ.
    spline_slope_method : str
        How the panel normalwash slope is built for surface (SPLINE1)
        splines. 'surface' (default): analytic x-derivative of the IPS
        displacement surface built from z-translations (SPLINE1
        semantics; captures chordwise z-gradient twist on shell
        models). 'rotation': nodal theta_y carries the slope
        (SPLINE2-like treatment; kept for reproducing results archived
        before the default change). Beam/collinear splines always use
        the rotation treatment.
    apply_w2gj : bool
        Apply the deck's W2GJ DMI (camber/twist initial downwash) to
        the trim boundary condition, matching MSC Nastran (default
        True). False reproduces flat-plate results archived before
        W2GJ support.

    Returns
    -------
    ResultData with SubcaseResult per trim subcase.
    """
    t0 = time.perf_counter()

    # ---------------------------------------------------------------
    # Phase 1: Build shared (Mach-independent) data — computed ONCE
    # ---------------------------------------------------------------
    shared = _build_shared_data(bdf_model, airfoil_config=airfoil_config,
                                spline_slope_method=spline_slope_method,
                                apply_w2gj=apply_w2gj)
    if shared is None:
        return ResultData(title="NastAero SOL 144 - Static Aeroelastic Trim")

    t_shared = time.perf_counter() - t0
    logger.info("Shared data built in %.2f s", t_shared)

    # Collect subcases with TRIM cards
    subcases = bdf_model.subcases if bdf_model.subcases else [bdf_model.global_case]
    trim_list = []  # (trim_obj, subcase_id)
    for subcase in subcases:
        effective = bdf_model.get_effective_subcase(subcase)
        trim_id = effective.trim_id
        if trim_id == 0:
            trim_id = bdf_model.global_case.trim_id
        if trim_id == 0:
            if bdf_model.trims:
                trim_id = next(iter(bdf_model.trims.keys()))
        if trim_id not in bdf_model.trims:
            logger.warning("Subcase %d: no TRIM card found", subcase.id)
            continue
        trim = bdf_model.trims[trim_id]
        trim_list.append((trim, subcase.id))

    if not trim_list:
        logger.error("No subcases with TRIM cards found.")
        return ResultData(title="NastAero SOL 144 - Static Aeroelastic Trim")

    # ---------------------------------------------------------------
    # Phase 2: Solve each subcase (sequential or parallel)
    # ---------------------------------------------------------------
    n_sc = len(trim_list)

    if n_workers == -1:
        import os
        n_workers = min(n_sc, max(1, (os.cpu_count() or 1) - 1))

    if n_workers > 1 and n_sc > 1:
        logger.info("Solving %d subcases with %d parallel workers...", n_sc, n_workers)
        sc_results = _solve_subcases_parallel(
            shared, trim_list, n_workers, blas_threads)
    else:
        logger.info("Solving %d subcases sequentially...", n_sc)
        sc_results = []
        for trim, sc_id in trim_list:
            logger.info("Solving subcase %d, TRIM %d (M=%.3f, q=%.6g)...",
                        sc_id, trim.tid, trim.mach, trim.q)
            sc_result = _solve_trim_subcase_from_shared(shared, trim, sc_id)
            sc_results.append(sc_result)

    t_total = time.perf_counter() - t0
    logger.info("All %d subcases done in %.1f s (shared=%.1f s, solve=%.1f s)",
                n_sc, t_total, t_shared, t_total - t_shared)

    results = ResultData(title="NastAero SOL 144 - Static Aeroelastic Trim")
    results.subcases = sc_results
    return results


# ---------------------------------------------------------------------------
# Phase 1: Shared data construction
# ---------------------------------------------------------------------------

def _build_shared_data(bdf_model: BDFModel,
                       airfoil_config=None,
                               spline_slope_method: str = 'surface',
                       apply_w2gj: bool = True
                       ) -> Optional[TrimSharedData]:
    """Build all Mach-independent data (computed once).

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model.
    airfoil_config : PanelAirfoilConfig, optional
        Airfoil camber assignment for DLM panels.  If provided, camber
        normalwash correction (dz_c/dx at each panel 3/4-chord) is
        added to the VLM boundary condition.  Equivalent to NASTRAN W2GJ.
    """
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)

    # Generate aerodynamic panels
    logger.info("Generating aerodynamic panel mesh...")
    boxes = generate_all_panels(bdf_model, use_nastran_eid=True)
    n_boxes = len(boxes)
    logger.info("  %d aerodynamic boxes generated", n_boxes)

    if n_boxes == 0:
        logger.error("No aerodynamic boxes. Check CAERO1 cards.")
        return None

    box_id_to_index = get_box_index_map(boxes)

    # Aero reference data
    aeros = bdf_model.aeros
    aero = bdf_model.aero
    if aeros is None and aero is None:
        logger.error("Missing AERO/AEROS card.")
        return None

    refc = aeros.refc if aeros else aero.refc
    refb = aeros.refb if aeros else 1.0
    refs = aeros.refs if aeros else 1.0
    velocity = aero.velocity if aero else 50.0

    # Partition K (use first subcase's SPC — typically shared across all).
    # 인수분해를 서브케이스 간에 재사용하는 구조라 f-set은 하나뿐이다.
    # 서브케이스마다 SPC가 다르면 그 가정이 깨지므로 조용히 넘기지
    # 않고 경고한다.
    subcases = bdf_model.subcases if bdf_model.subcases else [bdf_model.global_case]
    effective_sc = bdf_model.get_effective_subcase(subcases[0])
    _spc_ids = {getattr(bdf_model.get_effective_subcase(sc), 'spc_id', 0)
                for sc in subcases}
    if len(_spc_ids) > 1:
        logger.warning(
            "  서브케이스마다 SPC 집합이 다르다 (%s). 현재 구현은 첫 "
            "서브케이스의 구속으로 한 번만 분할하므로 나머지 서브케이스의 "
            "경계조건이 반영되지 않는다 — 덱을 SPC별로 나누어 실행할 것.",
            sorted(_spc_ids))
    K_ff, M_ff, F_f, f_dofs, s_dofs = fe_model.get_partitioned_system(effective_sc)
    n_free = len(f_dofs)
    logger.info("  Structural system: %d free DOFs, %d constrained", n_free, len(s_dofs))

    # Build spline coupling matrices (Mach-independent)
    # G_w: structural DOFs → normalwash (slope dz/dx), theta_y only
    # G_d: structural DOFs → z-displacement, z + theta_y*dx
    logger.info("  Building spline coupling matrices G_w and G_d...")
    G_w_dense, G_d_dense, G_force_node = _build_geff_per_spline(
        bdf_model, boxes, box_id_to_index, fe_model.dof_mgr, f_dofs,
        slope_method=spline_slope_method, collect_node_force=True)
    logger.info("  G_w (normalwash): max = %.4f, nonzeros = %d / %d",
                np.max(np.abs(G_w_dense)) if G_w_dense.size > 0 else 0,
                np.count_nonzero(G_w_dense), G_w_dense.size)
    logger.info("  G_d (displacement): max = %.4f, nonzeros = %d / %d",
                np.max(np.abs(G_d_dense)) if G_d_dense.size > 0 else 0,
                np.count_nonzero(G_d_dense), G_d_dense.size)
    G_sp = sp.csr_matrix(G_w_dense)
    G_disp = sp.csr_matrix(G_d_dense)
    del G_w_dense, G_d_dense

    # Weight and CG (Mach-independent)
    total_weight = _compute_total_weight(bdf_model)
    cg = _compute_cg(bdf_model)
    cg_x = float(cg[0])
    g = _detect_gravity(bdf_model)

    # All trim labels (exclude AELINK-dependent surfaces)
    aelink_deps = set()
    for aelink in getattr(bdf_model, 'aelinks', []):
        aelink_deps.add(aelink.dependent)
    all_trim_labels = []
    for aid, aestat in bdf_model.aestats.items():
        all_trim_labels.append(aestat.label)
    for aid, aesurf in bdf_model.aesurfs.items():
        if aesurf.label not in aelink_deps:
            all_trim_labels.append(aesurf.label)

    # Pre-compute iterative solver helpers for large models
    active_cols = None
    G_w_active_arr = None
    G_d_active_arr = None

    if n_free > 10000:
        # Active columns = union of nonzero columns in G_sp and G_disp
        G_w_csc = G_sp.tocsc()
        G_d_csc = G_disp.tocsc()
        col_nnz_w = np.diff(G_w_csc.indptr) > 0
        col_nnz_d = np.diff(G_d_csc.indptr) > 0
        active_cols = np.where(col_nnz_w | col_nnz_d)[0]
        n_active = len(active_cols)
        logger.info("  G has %d active columns out of %d (pre-computed)", n_active, n_free)
        G_w_active_arr = G_w_csc[:, active_cols].toarray()
        G_d_active_arr = G_d_csc[:, active_cols].toarray()

    # 강체모드 지지 판정: 덱 SPC < 6 자유도면 가상 마운트(3-2-1) 적용
    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}
    mount_idx = None
    D_rigid = _build_rigid_modes(bdf_model, fe_model.dof_mgr, f_dof_index,
                                 n_free, cg)
    if _has_free_rigid_modes(K_ff, D_rigid):
        # 덱이 SUPORT로 기준을 명시했으면 그것을 쓰고, 없을 때만 자동 선정
        mount_idx = _suport_mount_idx(bdf_model, fe_model.dof_mgr,
                                      f_dof_index)
        if mount_idx is None:
            mount_idx = _select_virtual_mount(bdf_model, fe_model.dof_mgr,
                                              f_dof_index, cg)
        if mount_idx is None:
            logger.warning("  Virtual mount selection failed — "
                           "falling back to unconstrained solve")
    mass_t3 = _build_mass_t3(bdf_model, fe_model.dof_mgr, f_dof_index,
                             n_free)

    # Camber normalwash correction (equivalent to NASTRAN W2GJ)
    w_camber = None
    if airfoil_config is not None and airfoil_config.panel_airfoils:
        from ..aero.airfoil_camber import compute_camber_normalwash
        w_camber = compute_camber_normalwash(boxes, bdf_model.caero_panels,
                                              airfoil_config)
        n_nonzero = np.count_nonzero(w_camber)
        if n_nonzero > 0:
            logger.info("  Camber normalwash: %d/%d boxes with correction "
                        "(max=%.4f, min=%.4f)",
                        n_nonzero, n_boxes,
                        np.max(w_camber), np.min(w_camber))
        else:
            w_camber = None  # all symmetric, no correction needed

    # Deck-supplied W2GJ DMI (initial incidence/camber downwash per box).
    # Rows follow the j-set = boxes in ascending box-id order, which is
    # exactly the generate_all_panels() ordering. MSC expresses W2GJ in
    # each panel's own geometric-normal sense (from CAERO corner
    # ordering); NastAero normalizes normals to +z/+y, so values on
    # flipped panels change sign (GACOMP: the left-side mirror is
    # sign-inverted in the raw file for this reason). Overall sign:
    # W2GJ acts like a local incidence and NastAero's unit-ANGLEA wash
    # is -1, so the wash contribution is -W2GJ (no n_z projection --
    # the value is already per-panel).
    w2gj = getattr(bdf_model, "dmis", {}).get("W2GJ") if apply_w2gj else None
    if w2gj is not None:
        if w2gj.m == n_boxes:
            flip = np.array([-1.0 if b.normal_flipped else 1.0
                             for b in boxes])
            incidence = flip * w2gj.matrix[:, 0]
            wash = -incidence
            w_camber = wash if w_camber is None else w_camber + wash
            logger.info("  W2GJ DMI applied: %d boxes, %d normal-flipped "
                        "(incidence max=%.4f, min=%.4f rad)",
                        n_boxes, int(np.sum(flip < 0)),
                        np.max(incidence), np.min(incidence))
        else:
            logger.warning("  W2GJ DMI has %d rows but model has %d boxes "
                           "- ignored", w2gj.m, n_boxes)

    shared = TrimSharedData(
        K_ff=K_ff, F_f=F_f, f_dofs=f_dofs, s_dofs=s_dofs,
        G_sp=G_sp, G_disp=G_disp, G_force_node=G_force_node,
        boxes=boxes, box_id_to_index=box_id_to_index,
        total_weight=total_weight, cg_x=cg_x,
        cg_y=float(cg[1]), cg_z=float(cg[2]),
        refc=refc, refb=refb, refs=refs, velocity=velocity, g=g,
        dof_mgr=fe_model.dof_mgr, bdf_model=bdf_model,
        all_trim_labels=all_trim_labels,
        slave_deps=fe_model.slave_deps,
        active_cols=active_cols, G_w_active=G_w_active_arr,
        G_d_active=G_d_active_arr,
        mount_idx=mount_idx, mass_t3=mass_t3,
        _solver_cache={},
        w_camber=w_camber,
    )
    return shared


# ---------------------------------------------------------------------------
# Phase 2: Per-subcase solver (uses TrimSharedData)
# ---------------------------------------------------------------------------

def _solve_trim_subcase_from_shared(shared: TrimSharedData,
                                     trim, subcase_id: int,
                                     nz_override: float = None) -> SubcaseResult:
    """Solve a single trim subcase using pre-computed shared data.

    This function is designed to be pickle-safe for multiprocessing.
    All Mach-independent data comes from `shared`.

    Parameters
    ----------
    shared : TrimSharedData
        Pre-computed Mach-independent data.
    trim : TRIM
        Trim card for this subcase.
    subcase_id : int
        Subcase identifier.
    nz_override : float, optional
        If provided, use this load factor instead of extracting from
        the TRIM card's URDD3 variable.  Default: extract from URDD3
        or 1.0 if URDD3 is not specified.
    """
    t_start = time.perf_counter()

    boxes = shared.boxes
    n_boxes = len(boxes)
    n_free = len(shared.f_dofs)
    q = trim.q
    mach = trim.mach

    # Detect XZ symmetry from AEROS card (needed for AIC and constraints)
    sym_xz = 0
    if shared.bdf_model.aeros and hasattr(shared.bdf_model.aeros, 'symxz'):
        sym_xz = shared.bdf_model.aeros.symxz
    elif shared.bdf_model.aero and hasattr(shared.bdf_model.aero, 'symxz'):
        sym_xz = shared.bdf_model.aero.symxz

    # Cache key for (Mach, q)-dependent quantities
    cache_key = (round(float(mach), 8), round(float(q), 8))
    cache = shared._solver_cache if shared._solver_cache is not None else {}

    cached = cache.get(cache_key)
    if cached is not None and 'A_jj' in cached:
        # ---- Cache hit: reuse D_inv and A_jj ----
        D_inv = cached['D_inv']
        A_jj = cached['A_jj']
        logger.info("  [SC%d] Reusing cached AIC/A_jj (M=%.3f, q=%.4e)",
                    subcase_id, mach, q)
    else:
        # ---- Cache miss: build AIC and A_jj ----
        # 1. Build AIC matrix (Mach-dependent)
        logger.info("  [SC%d] Building AIC matrix (%d x %d, M=%.3f, symxz=%d)...",
                    subcase_id, n_boxes, n_boxes, mach, sym_xz)
        D = build_aic_matrix(boxes, mach, reduced_freq=0.0, sym_xz=sym_xz)

        try:
            D_inv = np.linalg.inv(D)
        except np.linalg.LinAlgError:
            logger.warning("  [SC%d] AIC matrix singular, adding regularization",
                           subcase_id)
            D_inv = np.linalg.inv(D + np.eye(n_boxes) * 1e-10)

        # 2. Build force diagonal (Mach + q dependent)
        f_diag_vec = np.zeros(n_boxes)
        for j in range(n_boxes):
            chord_j = boxes[j].chord
            if chord_j > 1e-12:
                f_diag_vec[j] = 2.0 * q * boxes[j].area / chord_j

        A_jj = np.diag(f_diag_vec) @ D_inv

        # Store in cache (K_eff solver will be added by iterative solver)
        cache[cache_key] = {'D_inv': D_inv, 'A_jj': A_jj}
        if shared._solver_cache is not None:
            shared._solver_cache = cache

    G_sp = shared.G_sp        # normalwash coupling (slope)
    G_disp = shared.G_disp    # displacement coupling (force distribution)

    # 3. Parse trim variables (fixed vs free)
    if getattr(trim, "aeqr", 1.0) == 0.0:
        logger.warning("  [SC%d] TRIM AEQR=0.0 (rigid trim) requested — "
                       "NastAero solves flexible trim only; this deck is "
                       "for MSC comparison", subcase_id)
    trim_vars = {}
    for label, val in trim.variables:
        trim_vars[label] = val

    fixed_labels = {}
    free_labels = []
    for label in shared.all_trim_labels:
        if label in trim_vars:
            fixed_labels[label] = trim_vars[label]
        else:
            free_labels.append(label)

    # 자유 URDD*(강체 가속도)는 공력 미지수가 아니라 관성 릴리프 폐합이
    # 푸는 양이다 (MSC 자유비행 트림 덱은 SUPORT 자유도 수를 맞추기 위해
    # URDD1/2/4/6을 자유로 두는데, 그 해가 곧 릴리프 가속도와 동일).
    relief_urdds = [l for l in free_labels if l.startswith("URDD")]
    if relief_urdds:
        free_labels = [l for l in free_labels if not l.startswith("URDD")]
        logger.info("  [SC%d] Free URDD vars delegated to inertia relief: %s",
                    subcase_id, relief_urdds)

    n_trim_free = len(free_labels)

    # Determine load factor nz
    # Priority: nz_override > URDD3 in trim card > default 1.0
    if nz_override is not None:
        nz = nz_override
    else:
        # 명시적 URDD3=0.0은 0g 푸시오버다. 부재(기본 1g)와 구분해야
        # 한다 -- 덱 생성기도 "nz=0.00" 케이스를 그렇게 쓴다.
        nz = trim_vars["URDD3"] if "URDD3" in trim_vars else 1.0

    logger.info("  [SC%d] Trim: %d fixed %s, %d free %s, nz=%.3f",
                subcase_id, len(fixed_labels), list(fixed_labels.keys()),
                n_trim_free, free_labels, nz)

    # 4. Build normalwash vectors for trim variables
    # Forces on structure = G_disp^T @ A_jj @ w  (displacement G for force distribution)
    w_fixed = np.zeros(n_boxes)

    # Add camber normalwash correction (equivalent to NASTRAN W2GJ)
    if shared.w_camber is not None:
        w_fixed += shared.w_camber

    for label, value in fixed_labels.items():
        w_contrib = _trim_variable_normalwash(label, boxes, shared.box_id_to_index,
                                              shared.bdf_model)
        w_fixed += value * w_contrib
    F_trim_fixed = G_disp.T @ (A_jj @ w_fixed)

    Q_ax = np.zeros((n_free, n_trim_free))
    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, shared.box_id_to_index,
                                              shared.bdf_model)
        Q_ax[:, k] = G_disp.T @ (A_jj @ w_contrib)

    # 5. Build trim constraint equations
    # Trim constraints: total aero force = weight, total aero moment = 0
    # These are about the AERO forces, which depend on normalwash.
    # Normalwash from deformation = G_sp @ u_f (slope coupling)
    # D_r: contribution of structural deformation to aero force constraint
    # D_r[i,:] @ u_f = (constraint_vector @ A_jj @ G_sp) @ u_f
    #
    # SYMXZ symmetry: half-model panels produce half the full-aircraft forces.
    # The AIC already includes image vortex interference, so the panel forces
    # are correct for each half. The trim constraints must account for the
    # full aircraft: 2 * sum(half_forces) = nz * W_full.
    # We apply a sym_aero_factor to the constraint equations.
    sym_aero_factor = 1.0
    if sym_xz == 1 or sym_xz == -1:
        sym_aero_factor = 2.0
        logger.info("  [SC%d] SYMXZ=%d: constraint aero force factor = %.1f",
                    subcase_id, sym_xz, sym_aero_factor)

    # 제약식 구성 — 물리 패널 힘은 계수 x 법선 벡터이므로
    # (compute_aero_forces) 각 제약의 박스 팔(arm)도 법선 성분으로
    # 투영해야 실제 ΣF/ΣM과 일치한다. 투영 없이 계수 합을 쓰면 상반각
    # (1-n_z)만큼의 힘과 VTP 기여가 새어 나가 combined 잔차로 남는다
    # (GACOMP: Fz 42 N, My 135 kN-mm).
    #
    # 기본 2개(Fz, My)에 더해, 해당 축에 유효한 자유변수가 있을 때만
    # 롤(Mx)/요(Mz) 평형을 추가한다 — 없는데 추가하면 특이 시스템이 된다.
    # 모멘트 기준점은 질량 CG(관성력 질량 분포와 동일)라 관성 모멘트는
    # 정의상 0이고, aero 모멘트 = 0이 곧 combined 평형이다.
    nz_proj = np.array([boxes[i].normal[2] for i in range(n_boxes)])
    ny_proj = np.array([boxes[i].normal[1] for i in range(n_boxes)])
    nx_proj = np.array([boxes[i].normal[0] for i in range(n_boxes)])
    # 모멘트 팔은 힘 작용점(1/4-코드 더블릿 라인) 기준 — 다운워시
    # 콜로케이션 점(3/4-코드)을 쓰면 x_ac가 박스 코드의 절반만큼 뒤로
    # 밀려 중립점 근처에서 조종면 부호가 뒤집힌다 (MSC SOL 144 대조로 확인).
    cp_pts = np.array([boxes[i].doublet_point for i in range(n_boxes)]) \
        if n_boxes else np.zeros((0, 3))

    _ROLL_LABELS = ("ARON", "AILERON", "ROLL")
    _YAW_LABELS = ("RUD", "RUDDER")  # YAW(요레이트)는 워시 미구현이라 제외
    _SIDE_LABELS = ("SIDES",)
    has_roll_var = any(lbl.upper() in _ROLL_LABELS for lbl in free_labels)
    has_yaw_var = any(lbl.upper() in _YAW_LABELS for lbl in free_labels)
    has_side_var = any(lbl.upper() in _SIDE_LABELS for lbl in free_labels)

    # (이름, 박스 팔 벡터, rhs) — rhs는 목표값에서 고정변수 기여를 빼기 전
    constraint_rows = []
    if n_trim_free >= 1:
        constraint_rows.append(("Fz", nz_proj, nz * shared.total_weight))
    if n_trim_free >= 2:
        my_arm = (cp_pts[:, 0] - shared.cg_x) * nz_proj
        constraint_rows.append(("My", my_arm, 0.0))
        logger.info("  [SC%d] Moment ref (CG_x) = %.4f", subcase_id,
                    shared.cg_x)
    if has_roll_var and n_trim_free > len(constraint_rows):
        # Mx = Σ c_i [(y_i - cg_y) n_z,i - (z_i - cg_z) n_y,i]
        mx_arm = ((cp_pts[:, 1] - shared.cg_y) * nz_proj
                  - (cp_pts[:, 2] - shared.cg_z) * ny_proj)
        constraint_rows.append(("Mx", mx_arm, 0.0))
        logger.info("  [SC%d] Roll trim active (CG_y=%.2f, CG_z=%.2f)",
                    subcase_id, shared.cg_y, shared.cg_z)
    if has_yaw_var and n_trim_free > len(constraint_rows):
        # Mz = Σ c_i [(x_i - cg_x) n_y,i - (y_i - cg_y) n_x,i]
        mz_arm = ((cp_pts[:, 0] - shared.cg_x) * ny_proj
                  - (cp_pts[:, 1] - shared.cg_y) * nx_proj)
        constraint_rows.append(("Mz", mz_arm, 0.0))
        logger.info("  [SC%d] Yaw trim active", subcase_id)
    if has_side_var and n_trim_free > len(constraint_rows):
        # 측력 평형: ΣFy(aero) = ny * W (URDD2를 nz의 URDD3처럼 하중배수로 해석)
        ny_target = float(trim_vars.get("URDD2", 0.0))
        constraint_rows.append(("Fy", ny_proj, ny_target * shared.total_weight))
        logger.info("  [SC%d] Side-force trim active (ny=%.3f)",
                    subcase_id, ny_target)

    n_constraints = len(constraint_rows)

    D_r = np.zeros((n_constraints, n_free))
    D_x = np.zeros((n_constraints, n_trim_free))
    rhs_trim = np.zeros(n_constraints)

    w_free = [_trim_variable_normalwash(free_labels[k], boxes,
                                        shared.box_id_to_index,
                                        shared.bdf_model)
              for k in range(n_trim_free)]
    for j, (name, arm, target) in enumerate(constraint_rows):
        arm_A = arm @ A_jj
        D_r[j, :] = sym_aero_factor * (arm_A @ G_sp).ravel()
        for k in range(n_trim_free):
            D_x[j, k] = sym_aero_factor * (arm_A @ w_free[k])
        fixed_contrib = sym_aero_factor * (arm_A @ w_fixed)
        rhs_trim[j] = target - fixed_contrib

    # 6. Solve
    use_iterative = n_free > 10000
    logger.info("  [SC%d] Solver mode: %s (%d free DOFs, %d trim vars)",
                subcase_id, "iterative" if use_iterative else "dense",
                n_free, n_trim_free)

    # 관성 하중(-nz·g·m, T3)을 구조 해에 포함 — 자중 처짐이 스플라인을
    # 통해 워시로 되먹임되는 실제 공탄성 경로 (MSC INERTIAL 열과 동렬).
    # 자유비행(가상 마운트) 정식화에만 적용한다: 구속(SPC) 덱은 MSC의
    # 구속 트림처럼 공력 하중만으로 응답을 계산한다.
    F_static = F_trim_fixed
    if shared.mount_idx is not None and shared.mass_t3 is not None:
        F_static = F_trim_fixed - nz * shared.g * shared.mass_t3

    if use_iterative:
        u_f, x_trim = _solve_iterative_from_shared(
            shared, A_jj, Q_ax, F_static,
            D_r, D_x, rhs_trim, n_trim_free, n_constraints, subcase_id,
            cache_key=cache_key)
    else:
        u_f, x_trim = _solve_dense(shared.K_ff, G_sp, G_disp, A_jj, Q_ax,
                                    shared.F_f, F_static, D_r, D_x, rhs_trim,
                                    n_free, n_trim_free, n_constraints,
                                    mount_idx=shared.mount_idx)

    # 7. Post-process
    dof_mgr = shared.dof_mgr
    ndof = dof_mgr.total_dof
    u_full = np.zeros(ndof)
    for i, dof in enumerate(shared.f_dofs):
        u_full[dof] = u_f[i]

    # Recover slave DOFs from master DOFs (RBE2/MPC elimination)
    for s_dof, terms in shared.slave_deps.items():
        u_full[s_dof] = sum(coeff * u_full[m_dof]
                            for m_dof, coeff in terms)

    # Total normalwash
    w_total = G_sp @ u_f + w_fixed
    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, shared.box_id_to_index,
                                              shared.bdf_model)
        w_total += x_trim[k] * w_contrib

    gamma = D_inv @ w_total
    delta_cp = circulation_to_delta_cp(boxes, gamma)
    aero_forces = compute_aero_forces(boxes, delta_cp, q)

    # --- Force-balance scaling for fully-constrained cases ---
    # When n_trim_free == 0 (dynamic cases from 6-DOF simulation), the trim
    # solver has no free variables to adjust and enforces no force/moment
    # balance constraints. The VLM-computed total lift may differ from
    # nz × W because the VLM panel solution uses a different aero model
    # than the stability-derivative model in the 6-DOF simulator.
    #
    # Fix: scale all aero panel forces uniformly so that the total
    # z-force equals the target lift (nz × W). This preserves the
    # spanwise and chordwise load distribution shape from VLM while
    # ensuring global force equilibrium with the inertial loads.
    if n_constraints == 0 and n_trim_free == 0:
        total_aero_fz = float(np.sum(aero_forces[:, 2]))
        target_lift = nz * shared.total_weight
        if abs(total_aero_fz) > 1.0:
            lift_scale = target_lift / total_aero_fz
            aero_forces = aero_forces * lift_scale
            logger.info("  [SC%d] Force-balance scaling: %.4f "
                        "(VLM Fz=%.0f → nz*W=%.0f)",
                        subcase_id, lift_scale, total_aero_fz, target_lift)
        else:
            logger.warning("  [SC%d] Near-zero VLM lift (%.2e) — "
                           "cannot apply force-balance scaling",
                           subcase_id, total_aero_fz)

    # Store results
    sc_result = SubcaseResult(subcase_id=subcase_id)
    for nid in dof_mgr.node_ids:
        nd = dof_mgr.get_node_dofs(nid)
        sc_result.displacements[nid] = u_full[nd[0]:nd[5]+1]

    sc_result.trim_variables = {}
    for label, val in fixed_labels.items():
        sc_result.trim_variables[label] = val
    for k, label in enumerate(free_labels):
        sc_result.trim_variables[label] = x_trim[k]

    sc_result.aero_pressures = delta_cp
    sc_result.aero_forces = aero_forces
    sc_result.aero_boxes = boxes

    # Log results
    logger.info("  [SC%d] Trim results:", subcase_id)
    for label, val in sc_result.trim_variables.items():
        if label == "ANGLEA":
            logger.info("    %s = %.4f rad (%.2f deg)", label, val, np.degrees(val))
        else:
            logger.info("    %s = %.6f", label, val)
    logger.info("  [SC%d] Max displacement = %.6e", subcase_id, np.max(np.abs(u_full)))
    total_lift = np.sum(aero_forces[:, 2])
    logger.info("  [SC%d] Total aero Fz = %.2f (nz*W = %.2f, nz=%.3f)",
                subcase_id, total_lift, nz * shared.total_weight, nz)

    # Pitch moment about CG
    if n_boxes > 0:
        my_total = sum(aero_forces[i, 2] * (boxes[i].doublet_point[0] - shared.cg_x)
                       for i in range(n_boxes))
        logger.info("  [SC%d] Pitch moment about CG = %.2f", subcase_id, my_total)

    # 8. Compute nodal trim loads
    try:
        from ..loads_analysis.trim_loads import (
            compute_trim_nodal_loads, verify_trim_balance)

        # Lateral load factor: URDD2 (nz의 URDD3와 동일 규약), 동적 케이스는
        # aero 측력에서 유도
        ny_val = float(trim_vars.get("URDD2", 0.0))
        if n_constraints == 0 and n_trim_free == 0:
            total_aero_fy = float(np.sum(aero_forces[:, 1]))
            if abs(shared.total_weight) > 1.0:
                ny_val = total_aero_fy / shared.total_weight
                if abs(ny_val) > 0.01:
                    logger.info("  [SC%d] Lateral load factor ny=%.3f",
                                subcase_id, ny_val)

        aero_nodal, inertial_nodal, combined_nodal = compute_trim_nodal_loads(
            shared.bdf_model, boxes, aero_forces, G_disp, shared.f_dofs, dof_mgr,
            nz=nz, g=shared.g, ny=ny_val,
            G_force_node=shared.G_force_node)

        cg_pt = np.array([shared.cg_x, shared.cg_y, shared.cg_z])

        # 준정적 관성 밸런싱 (inertia relief closure) — 트림이 강제하지
        # 못한 잔여 합력/합모멘트(예: §23.349 롤 기동의 에일러론 모멘트)를
        # 강체 가속도로 환산해 질량 비례 분포 하중으로 정확히 닫는다.
        # 대상: 자유-자유 및 기준점 지지 모델 — 구속 DOF ≤ 9이면(3-2-1
        # ~3-2-3류 기준 마운트) SPC는 강체모드 제거용 수치 장치이므로
        # 잔차를 지지점에 남기면 안 된다. 그보다 크면(캔틸레버 루트 등
        # 실제 경계) 잔차가 지지 반력으로 잡히는 것이 물리이므로 미적용.
        spc_id = 0
        for _sc in shared.bdf_model.subcases:
            if _sc.id == subcase_id:
                spc_id = getattr(_sc, "spc_id", 0) or getattr(
                    shared.bdf_model.global_case, "spc_id", 0)
                break
        n_spc_dofs = 0
        if spc_id and spc_id in shared.bdf_model.spcs:
            try:
                cdofs, _enf = shared.dof_mgr.get_constrained_dofs(
                    shared.bdf_model.spcs[spc_id], shared.bdf_model.nodes)
                n_spc_dofs = len(cdofs)
            except Exception:
                n_spc_dofs = 999  # 판정 실패 시 보수적으로 미적용
        if n_spc_dofs <= 9:
            from ..loads_analysis.trim_loads import apply_inertia_relief

            relief = apply_inertia_relief(
                shared.bdf_model, inertial_nodal, combined_nodal,
                cg=cg_pt, g=shared.g)
            if relief and max(abs(v) for v in relief.values()) > 1e-6:
                logger.info(
                    "  [SC%d] Inertia relief: dny=%.4g dnz=%.4g "
                    "p_dot=%.4g q_dot=%.4g r_dot=%.4g rad/s^2",
                    subcase_id, relief.get("relief_ny", 0.0),
                    relief.get("relief_nz", 0.0), relief.get("p_dot", 0.0),
                    relief.get("q_dot", 0.0), relief.get("r_dot", 0.0))

        sc_result.nodal_aero_forces = aero_nodal
        sc_result.nodal_inertial_forces = inertial_nodal
        sc_result.nodal_combined_forces = combined_nodal

        balance = verify_trim_balance(shared.bdf_model, combined_nodal, ref_point=cg_pt)
        if n_spc_dofs <= 9:
            balance.update(relief)
        sc_result.trim_balance = balance
    except Exception as e:
        logger.warning("  [SC%d] Trim loads computation failed: %s", subcase_id, e)

    t_elapsed = time.perf_counter() - t_start
    logger.info("  [SC%d] Subcase done in %.2f s", subcase_id, t_elapsed)

    return sc_result


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

def _solve_subcases_parallel(shared: TrimSharedData,
                              trim_list: list,
                              n_workers: int,
                              blas_threads: int) -> list:
    """Solve subcases in parallel, grouped by (Mach, q) for cache reuse.

    Subcases sharing the same (Mach, q) are dispatched to the same worker
    so that the expensive K_eff factorization (~80 s for 135 k DOFs) is
    computed once and reused for all subcases in the group.

    Typically there are 4-10 unique (Mach, q) groups across 100+ subcases,
    giving ~10-25× speedup from factorization reuse alone.
    """
    import os
    from collections import defaultdict

    # Set BLAS threads BEFORE forking
    os.environ['OPENBLAS_NUM_THREADS'] = str(blas_threads)
    os.environ['MKL_NUM_THREADS'] = str(blas_threads)
    os.environ['OMP_NUM_THREADS'] = str(blas_threads)

    # Group subcases by (Mach, q) so workers can reuse K_eff factorization
    groups = defaultdict(list)       # key → [(trim, sc_id, original_index)]
    for idx, (trim, sc_id) in enumerate(trim_list):
        key = (round(float(trim.mach), 8), round(float(trim.q), 8))
        groups[key].append((trim, sc_id, idx))

    n_groups = len(groups)
    n_workers = min(n_workers, n_groups)

    logger.info("Parallel dispatch: %d subcases in %d (Mach,q) groups, "
                "%d workers", len(trim_list), n_groups, n_workers)
    for key, items in sorted(groups.items()):
        logger.info("  Group M=%.3f q=%.4e: %d subcases", key[0], key[1], len(items))

    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = [None] * len(trim_list)

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(blas_threads,)
        ) as executor:
            future_to_indices = {}
            for key, items in groups.items():
                batch = [(trim, sc_id) for trim, sc_id, _ in items]
                orig_indices = [idx for _, _, idx in items]
                future = executor.submit(_solve_batch_worker, shared, batch)
                future_to_indices[future] = orig_indices

            for future in as_completed(future_to_indices):
                orig_indices = future_to_indices[future]
                try:
                    batch_results = future.result()
                    for i, idx in enumerate(orig_indices):
                        results[idx] = batch_results[i]
                except Exception as e:
                    logger.error("Batch failed: %s", e)
                    for idx in orig_indices:
                        results[idx] = SubcaseResult(
                            subcase_id=trim_list[idx][1])

    except Exception as e:
        logger.warning("Parallel execution failed (%s), "
                       "falling back to sequential", e)
        results = []
        for trim, sc_id in trim_list:
            logger.info("Solving subcase %d sequentially (fallback)...", sc_id)
            sc_result = _solve_trim_subcase_from_shared(shared, trim, sc_id)
            results.append(sc_result)

    return results


def _worker_init(blas_threads: int):
    """Initialize worker process: set BLAS thread count."""
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = str(blas_threads)
    os.environ['MKL_NUM_THREADS'] = str(blas_threads)
    os.environ['OMP_NUM_THREADS'] = str(blas_threads)


def _solve_batch_worker(shared: TrimSharedData,
                         batch: list) -> list:
    """Process a batch of subcases that share the same (Mach, q).

    The first subcase builds and caches the AIC inverse, A_jj, and
    K_eff factorization.  Subsequent subcases reuse everything via
    shared._solver_cache, turning an ~80 s factorization into a ~2 s
    back-substitution.
    """
    results = []
    for trim, sc_id in batch:
        r = _solve_trim_subcase_from_shared(shared, trim, sc_id)
        results.append(r)
    return results


def _solve_worker(shared: TrimSharedData, trim, sc_id: int) -> SubcaseResult:
    """Top-level worker function for parallel execution (single subcase)."""
    return _solve_trim_subcase_from_shared(shared, trim, sc_id)


# ---------------------------------------------------------------------------
# Legacy API: original solve_trim_subcase (for backward compatibility)
# ---------------------------------------------------------------------------

def _solve_trim_subcase(bdf_model: BDFModel, fe_model: FEModel,
                        boxes: List[AeroBox], box_id_to_index: Dict[int, int],
                        trim, subcase: Subcase,
                        refc: float, refb: float, refs: float,
                        velocity: float) -> SubcaseResult:
    """Solve a single trim subcase (legacy API, builds shared data internally).

    This is kept for backward compatibility. New code should use
    _solve_trim_subcase_from_shared() with pre-built TrimSharedData.
    """
    # Build minimal shared data for this single call
    aelink_deps = set()
    for aelink in getattr(bdf_model, 'aelinks', []):
        aelink_deps.add(aelink.dependent)
    all_trim_labels = []
    for aid, aestat in bdf_model.aestats.items():
        all_trim_labels.append(aestat.label)
    for aid, aesurf in bdf_model.aesurfs.items():
        if aesurf.label not in aelink_deps:
            all_trim_labels.append(aesurf.label)

    K_ff, M_ff, F_f, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)
    n_free = len(f_dofs)

    G_w_dense, G_d_dense, G_force_node = _build_geff_per_spline(
        bdf_model, boxes, box_id_to_index, fe_model.dof_mgr, f_dofs,
        collect_node_force=True)
    G_sp = sp.csr_matrix(G_w_dense)
    G_disp = sp.csr_matrix(G_d_dense)
    del G_w_dense, G_d_dense

    total_weight = _compute_total_weight(bdf_model)
    cg = _compute_cg(bdf_model)
    cg_x = float(cg[0])
    g = _detect_gravity(bdf_model)

    # Pre-compute iterative helpers if large
    active_cols = None
    G_w_active_arr = None
    G_d_active_arr = None
    if n_free > 10000:
        G_w_csc = G_sp.tocsc()
        G_d_csc = G_disp.tocsc()
        col_nnz_w = np.diff(G_w_csc.indptr) > 0
        col_nnz_d = np.diff(G_d_csc.indptr) > 0
        active_cols = np.where(col_nnz_w | col_nnz_d)[0]
        G_w_active_arr = G_w_csc[:, active_cols].toarray()
        G_d_active_arr = G_d_csc[:, active_cols].toarray()

    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}
    mount_idx = None
    D_rigid = _build_rigid_modes(bdf_model, fe_model.dof_mgr, f_dof_index,
                                 n_free, cg)
    if _has_free_rigid_modes(K_ff, D_rigid):
        # 덱이 SUPORT로 기준을 명시했으면 그것을 쓰고, 없을 때만 자동 선정
        mount_idx = _suport_mount_idx(bdf_model, fe_model.dof_mgr,
                                      f_dof_index)
        if mount_idx is None:
            mount_idx = _select_virtual_mount(bdf_model, fe_model.dof_mgr,
                                              f_dof_index, cg)
        if mount_idx is None:
            logger.warning("  Virtual mount selection failed — "
                           "falling back to unconstrained solve")
    mass_t3 = _build_mass_t3(bdf_model, fe_model.dof_mgr, f_dof_index,
                             n_free)

    shared = TrimSharedData(
        K_ff=K_ff, F_f=F_f, f_dofs=f_dofs, s_dofs=s_dofs,
        G_sp=G_sp, G_disp=G_disp, G_force_node=G_force_node,
        boxes=boxes, box_id_to_index=box_id_to_index,
        total_weight=total_weight, cg_x=cg_x,
        cg_y=float(cg[1]), cg_z=float(cg[2]),
        refc=refc, refb=refb, refs=refs, velocity=velocity, g=g,
        dof_mgr=fe_model.dof_mgr, bdf_model=bdf_model,
        all_trim_labels=all_trim_labels,
        slave_deps=fe_model.slave_deps,
        active_cols=active_cols, G_w_active=G_w_active_arr,
        G_d_active=G_d_active_arr,
        mount_idx=mount_idx, mass_t3=mass_t3,
        _solver_cache={},
    )

    return _solve_trim_subcase_from_shared(shared, trim, subcase.id)


# ---------------------------------------------------------------------------
# Dense solver (small models, n_free <= 10000)
# ---------------------------------------------------------------------------

def _solve_dense(K_ff, G_sp, G_disp, A_jj, Q_ax, F_f, F_trim_fixed,
                 D_r, D_x, rhs_trim, n_free, n_trim_free, n_constraints,
                 mount_idx=None):
    """Solve trim using dense matrices (small models).

    Q_aa = G_disp^T @ A_jj @ G_sp (asymmetric: displacement for forces,
    normalwash for coupling).

    구조 평형은 K u = F_ext + Q_aa u 이므로 계에는 (K - Q_aa)가
    들어간다. 부호가 반대면 동압에 따라 인위적으로 강해져(반발산)
    유연 증분이 과소평가되고 발산이 나타나지 않는다.
    """
    G_w_dense = G_sp.toarray() if sp.issparse(G_sp) else G_sp
    G_d_dense = G_disp.toarray() if sp.issparse(G_disp) else G_disp
    Q_aa_free = G_d_dense.T @ A_jj @ G_w_dense

    n_total = n_free + n_trim_free
    K_dense = K_ff.toarray() if sp.issparse(K_ff) else K_ff

    if mount_idx is not None:
        # 자유-자유 소형 모델: 가상 마운트 행/열 소거 (u[mount]=0)
        K_dense = K_dense.copy()
        Q_aa_free = Q_aa_free.copy()
        diag = np.abs(np.diag(K_dense))
        scale = np.mean(diag[diag > 0]) if np.any(diag > 0) else 1.0
        for i in mount_idx:
            K_dense[i, :] = 0.0; K_dense[:, i] = 0.0; K_dense[i, i] = scale
            Q_aa_free[i, :] = 0.0; Q_aa_free[:, i] = 0.0
        F_f = np.asarray(F_f).copy(); F_f[mount_idx] = 0.0
        F_trim_fixed = np.asarray(F_trim_fixed).copy()
        F_trim_fixed[mount_idx] = 0.0
        Q_ax = np.asarray(Q_ax).copy()
        if Q_ax.size:
            Q_ax[mount_idx, :] = 0.0

    if n_trim_free > 0:
        A_sys = np.zeros((n_total + n_constraints, n_total))
        rhs_sys = np.zeros(n_total + n_constraints)

        A_sys[:n_free, :n_free] = K_dense - Q_aa_free
        A_sys[:n_free, n_free:n_total] = -Q_ax
        rhs_sys[:n_free] = F_f + F_trim_fixed

        A_sys[n_total:n_total+n_constraints, :n_free] = D_r
        A_sys[n_total:n_total+n_constraints, n_free:n_total] = D_x
        rhs_sys[n_total:n_total+n_constraints] = rhs_trim

        logger.info("  Solving dense coupled system (%d x %d)...",
                     A_sys.shape[0], A_sys.shape[1])
        sol, _, _, _ = np.linalg.lstsq(A_sys, rhs_sys, rcond=None)
        u_f = sol[:n_free]
        x_trim = sol[n_free:n_total]
    else:
        K_eff = K_dense - Q_aa_free
        rhs_eff = F_f + F_trim_fixed
        u_f = np.linalg.solve(K_eff, rhs_eff)
        x_trim = np.array([])

    return u_f, x_trim


# ---------------------------------------------------------------------------
# Iterative solver (large models) using pre-computed shared data
# ---------------------------------------------------------------------------

def _solve_iterative_from_shared(shared: TrimSharedData,
                                  A_jj: np.ndarray,
                                  Q_ax: np.ndarray,
                                  F_trim_fixed: np.ndarray,
                                  D_r: np.ndarray, D_x: np.ndarray,
                                  rhs_trim: np.ndarray,
                                  n_trim_free: int, n_constraints: int,
                                  subcase_id: int,
                                  cache_key=None):
    """Sparse solver using shared pre-computed structural data.

    Uses pre-computed G_w_active, G_d_active, active_cols from
    TrimSharedData to avoid redundant work across subcases.

    Q_aa = G_d^T @ A_jj @ G_w  (asymmetric: displacement forces × normalwash)

    K_eff factorization is cached by (mach, q) via cache_key so that
    subcases sharing the same flight condition reuse the expensive LU
    decomposition (~80 s for 135k DOFs).
    """
    t_start = time.perf_counter()
    n_free = len(shared.f_dofs)

    # ---- Check K_eff factorization cache ----
    cache = shared._solver_cache if shared._solver_cache is not None else {}
    cached = cache.get(cache_key) if cache_key else None
    cached_solver = cached.get('solve_fn') if cached else None

    if cached_solver is not None:
        # Cache hit — skip Q_aa build + factorization entirely
        _solve_system = cached_solver
        logger.info("  [SC%d] Reusing cached K_eff factorization (%.2f s saved)",
                    subcase_id, cached.get('factor_time', 0))
    else:
        # ---- Cache miss — build Q_aa and factorize K_eff ----
        # Use pre-computed active columns
        active_cols = shared.active_cols
        G_w_active = shared.G_w_active   # normalwash coupling active cols
        G_d_active = shared.G_d_active   # displacement coupling active cols
        n_active = len(active_cols) if active_cols is not None else 0

        # Build Q_aa (Mach-dependent via A_jj)
        # Q_aa = G_d^T @ A_jj @ G_w  (asymmetric)
        logger.info("  [SC%d] Building Q_aa (%d active cols)...", subcase_id, n_active)
        t_q = time.perf_counter()

        B_active = A_jj @ G_w_active   # (n_boxes x n_active) — Mach-dependent
        Q_active = G_d_active.T @ B_active  # (n_active x n_active) — asymmetric!
        logger.info("  [SC%d] Q_active computed in %.2f s",
                    subcase_id, time.perf_counter() - t_q)

        # Build K_eff = K - Q_aa (인공 정규화 없음 — 자유-자유는 가상
        # 마운트가, 구속 덱은 실제 경계조건이 강체모드를 처리한다)
        K_eff = (shared.K_ff if sp.issparse(shared.K_ff)
                 else sp.csc_matrix(shared.K_ff)).tocsc(copy=True)

        if n_active > 0:
            # 큰 n_active에서도 결합을 반드시 조립한다. 누락하면 구조
            # 방정식에는 공탄성 되먹임이 없는데 트림 구속행(D_r)은
            # 변형 워시를 계속 반영해 해가 내부 모순에 빠진다.
            CHUNK = 2000
            q_blocks = []
            for s in range(0, n_active, CHUNK):
                e = min(s + CHUNK, n_active)
                blk = Q_active[s:e, :]
                row_idx = np.repeat(active_cols[s:e], n_active)
                col_idx = np.tile(active_cols, e - s)
                q_vals = blk.ravel()
                mask = np.abs(q_vals) > 1e-30
                if mask.any():
                    q_blocks.append((q_vals[mask], row_idx[mask], col_idx[mask]))
            if q_blocks:
                vals = np.concatenate([b[0] for b in q_blocks])
                rows = np.concatenate([b[1] for b in q_blocks])
                cols = np.concatenate([b[2] for b in q_blocks])
                Q_sp = sp.coo_matrix((vals, (rows, cols)),
                                     shape=(n_free, n_free)).tocsc()
                K_eff = K_eff - Q_sp

        # 가상 마운트: Q까지 조립된 뒤 행/열 소거 (마운트 자유도 = 0)
        if shared.mount_idx is not None:
            K_eff = _apply_virtual_mount(K_eff.tocsc(), shared.mount_idx)

        # 미세 백스톱: 전기체 강체모드(마운트/경계가 처리) 외의 잔여
        # 특이성(고립 부재/근사기구)만 정칙화한다. avg_diag*1e-12는 날개급
        # 유연 모드 강성의 0.1% 미만이라 변형을 왜곡하지 않는다
        # (이전의 1e-8 스케일은 인공 탄성 기초로 작용 — 제거됨).
        diag = np.abs(K_eff.diagonal())
        avg_diag = np.mean(diag[diag > 0]) if np.any(diag > 0) else 1.0
        K_eff = K_eff.tocsc() + sp.eye(n_free, format='csc') * (avg_diag * 1e-12)

        # Factorize K_eff
        logger.info("  [SC%d] Factorizing K_eff (%d x %d)...",
                    subcase_id, n_free, n_free)
        t_lu = time.perf_counter()

        K_eff_csc = K_eff.tocsc()

        # For free-trim (multiple RHS solves), prefer direct factorization
        # since we reuse the factorization across trim iterations.
        use_direct = (n_trim_free > 0)

        _solver_mode = None
        _pardiso_solve = None
        K_lu = None

        # 1. Try pypardiso (MKL PARDISO) — always preferred
        try:
            from pypardiso import spsolve as pardiso_solve
            _solver_mode = 'pardiso'
            _pardiso_solve = pardiso_solve
            logger.info("  [SC%d] Using PyPardiso solver", subcase_id)
        except ImportError:
            pass

        if _solver_mode is None:
            # Use direct LU for both free-trim and all-fixed cases.
            # CG with ILU preconditioner fails for all-fixed cases because
            # K_eff = K - Q_aa is asymmetric (Q_aa = G_d^T @ A_jj @ G_sp,
            # where G_d ≠ G_sp). Direct LU is more reliable and the
            # factorization is cached for reuse across subcases with the
            # same (Mach, q).
            try:
                K_lu = spla.splu(K_eff_csc, permc_spec='COLAMD')
                _solver_mode = 'splu'
                logger.info("  [SC%d] Using direct LU", subcase_id)
            except Exception as e:
                logger.error("  [SC%d] LU factorization failed: %s",
                            subcase_id, e)
                return np.zeros(n_free), np.zeros(n_trim_free)

        factor_time = time.perf_counter() - t_lu
        logger.info("  [SC%d] Factorization done in %.2f s",
                    subcase_id, factor_time)

        # Build solve function (마운트 자유도의 RHS는 0으로 강제 —
        # 행/열 소거된 K_eff와 함께 u[mount] = 0 을 보장한다)
        _mount = shared.mount_idx

        if _solver_mode == 'pardiso':
            _K_eff_csc_ref = K_eff_csc  # prevent GC

            def _solve_system(rhs_vec, _csc=K_eff_csc, _ps=_pardiso_solve,
                              _m=_mount):
                if _m is not None:
                    rhs_vec = rhs_vec.copy()
                    rhs_vec[_m] = 0.0
                return _ps(_csc, rhs_vec)
        else:
            def _solve_system(rhs_vec, _lu=K_lu, _m=_mount):
                if _m is not None:
                    rhs_vec = rhs_vec.copy()
                    rhs_vec[_m] = 0.0
                return _lu.solve(rhs_vec)

        # ---- Store in cache ----
        if cache_key is not None and cached is not None:
            cached['solve_fn'] = _solve_system
            cached['factor_time'] = factor_time
            logger.info("  [SC%d] K_eff factorization cached (key=%s)",
                        subcase_id, cache_key)

    # Solve the coupled trim system using Schur complement
    # The full system is:
    #   [K_eff  | -Q_ax] [u_f   ]   [F_base    ]
    #   [D_r    |  D_x ] [x_trim] = [rhs_trim  ]
    #
    # Where F_base = F_f + F_trim_fixed (forces from fixed trim vars)
    #
    # From the top block: u_f = K_eff^{-1} @ (F_base + Q_ax @ x_trim)
    # Substituting into bottom block:
    #   D_r @ K_eff^{-1} @ (F_base + Q_ax @ x) + D_x @ x = rhs_trim
    # Let: u0 = K_eff^{-1} @ F_base
    #      U_k = K_eff^{-1} @ Q_ax  (sensitivity matrix)
    # Then: (D_r @ U_k + D_x) @ x = rhs_trim - D_r @ u0
    # This is the Schur complement — solved directly, no iteration needed.

    F_base = shared.F_f + F_trim_fixed

    if n_constraints == 0 or n_trim_free == 0:
        # No free trim variables — single solve
        u_f = _solve_system(F_base)
        x_trim = np.array([])
    else:
        # Step 1: Base displacement (no free trim var contribution)
        u0 = _solve_system(F_base)

        # Step 2: Displacement sensitivity to each trim variable
        U_k = np.zeros((n_free, n_trim_free))
        for k in range(n_trim_free):
            U_k[:, k] = _solve_system(Q_ax[:, k])

        # Step 3: Schur complement
        S = D_r @ U_k + D_x  # (n_constraints x n_trim_free)
        rhs_schur = rhs_trim - D_r @ u0

        logger.info("  [SC%d] Schur complement: cond(S) = %.2e",
                    subcase_id, np.linalg.cond(S))

        if n_trim_free == n_constraints:
            x_trim = np.linalg.solve(S, rhs_schur)
        else:
            x_trim, _, _, _ = np.linalg.lstsq(S, rhs_schur, rcond=None)

        # Step 4: Total displacement
        u_f = u0 + U_k @ x_trim

        logger.info("  [SC%d] Trim solved (direct Schur complement)", subcase_id)
        for k in range(n_trim_free):
            logger.info("  [SC%d]   x_trim[%d] = %.6f", subcase_id, k, x_trim[k])

    t_elapsed = time.perf_counter() - t_start
    logger.info("  [SC%d] Iterative solve done in %.2f s", subcase_id, t_elapsed)

    return u_f, x_trim


# Legacy _solve_iterative removed — use _solve_iterative_from_shared instead.


# ---------------------------------------------------------------------------
# Spline coupling matrix builder
# ---------------------------------------------------------------------------

def _build_geff_per_spline(bdf_model: BDFModel, boxes: List[AeroBox],
                            box_id_to_index: Dict[int, int],
                            dof_mgr: DOFManager,
                            f_dofs: List[int],
                            slope_method: str = 'surface',
                            collect_node_force: bool = False):
    """Build spline coupling matrices using per-spline mapping.

    Returns TWO matrices:
    - G_w: normalwash coupling (structural DOFs → panel normalwash/slope)
      Only theta_y (DOF 5) contributes.
    - G_d: displacement coupling (structural DOFs → panel z-displacement)
      Both z (DOF 3) and theta_y × dx (DOF 5) contribute.

    The aero stiffness is: Q_aa = G_d^T @ A_jj @ G_w
    Force distribution uses G_d^T, normalwash uses G_w.

    Parameters
    ----------
    bdf_model : BDFModel
    boxes : list of AeroBox
    box_id_to_index : dict
    dof_mgr : DOFManager
    f_dofs : list of int

    Returns
    -------
    G_w : ndarray (n_boxes, n_free)
        Normalwash coupling matrix.
    G_d : ndarray (n_boxes, n_free)
        Displacement coupling matrix.
    G_fn : sparse (n_boxes, n_nodes), optional
        ``collect_node_force``일 때만 반환. 물리공간 힘 분배 가중치로,
        자유도 소거와 무관하게 모든 스플라인 SET 절점을 담는다.
        열 순서는 ``dof_mgr.node_ids``.
    """
    n_boxes = len(boxes)
    n_free = len(f_dofs)
    G_w = np.zeros((n_boxes, n_free))
    G_d = np.zeros((n_boxes, n_free))

    # Build f_dofs lookup for fast index finding
    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}

    node_force = None
    if collect_node_force:
        node_ids = list(dof_mgr.node_ids)
        node_force = ([], [], [], {nid: i for i, nid in enumerate(node_ids)})

    def _finish(G_w, G_d):
        if not collect_node_force:
            return G_w, G_d
        rows, cols, vals, node_index = node_force
        G_fn = sp.coo_matrix((vals, (rows, cols)),
                             shape=(n_boxes, len(node_index))).tocsr()
        return G_w, G_d, G_fn

    if not bdf_model.splines:
        # No splines defined, fall back to global beam spline
        all_nids = sorted(bdf_model.nodes.keys())
        struct_xyz = np.array([bdf_model.nodes[nid].xyz_global
                               for nid in all_nids])
        # 워시(G_w)는 3/4-코드 콜로케이션 점, 힘 전달(G_d)은 1/4-코드
        # 더블릿 라인(힘 작용점)에서 평가한다 (MSC SOL 144 대조로 확인).
        wash_pts = np.array([box.control_point for box in boxes])
        force_pts = np.array([box.doublet_point for box in boxes])
        G_ka_wash = build_beam_spline(struct_xyz, wash_pts, axis=1)
        G_ka_force = build_beam_spline(struct_xyz, force_pts, axis=1)
        _fill_geff(G_w, G_d, G_ka_wash, G_ka_force, range(n_boxes), all_nids,
                   force_pts, struct_xyz, dof_mgr, f_dof_index,
                   node_force=node_force)
        return _finish(G_w, G_d)

    # Process each spline independently
    for sid, spline in bdf_model.splines.items():
        # Get structural nodes for this spline
        setg = spline.setg
        if setg not in bdf_model.sets:
            logger.warning("  Spline %d: SET %d not found", sid, setg)
            continue
        spline_nids = bdf_model.sets[setg].ids
        if not spline_nids:
            continue

        struct_xyz = np.array([bdf_model.nodes[nid].xyz_global
                               for nid in spline_nids])

        # Get aero boxes for this spline
        caero_eid = spline.caero
        if caero_eid not in bdf_model.caero_panels:
            logger.warning("  Spline %d: CAERO1 %d not found", sid, caero_eid)
            continue

        caero = bdf_model.caero_panels[caero_eid]
        n_caero_boxes = caero.nspan * caero.nchord

        # Determine box range: use BOX1/BOX2 if available, else full CAERO1
        if hasattr(spline, 'box1') and hasattr(spline, 'box2'):
            raw_box1 = spline.box1
            raw_box2 = spline.box2
        elif hasattr(spline, 'id1') and hasattr(spline, 'id2'):
            raw_box1 = spline.id1
            raw_box2 = spline.id2
        else:
            raw_box1 = 0
            raw_box2 = 0

        # Convert to Nastran EID-based box IDs
        if raw_box1 <= 0 or raw_box1 < caero_eid:
            box1 = caero_eid
        else:
            box1 = raw_box1

        if raw_box2 <= 0:
            box2 = caero_eid + n_caero_boxes - 1
        elif raw_box2 < caero_eid:
            box2 = caero_eid + min(raw_box2, n_caero_boxes - 1)
        else:
            box2 = raw_box2

        # Find sequential indices for these boxes
        spline_box_indices = []
        for box_eid in range(box1, box2 + 1):
            if box_eid in box_id_to_index:
                spline_box_indices.append(box_id_to_index[box_eid])

        if not spline_box_indices:
            logger.warning("  Spline %d: no matching boxes found (%d-%d)",
                           sid, box1, box2)
            continue

        # 워시는 3/4-코드 콜로케이션 점, 힘 전달은 1/4-코드 더블릿 라인
        wash_pts = np.array([boxes[idx].control_point
                             for idx in spline_box_indices])
        force_pts = np.array([boxes[idx].doublet_point
                              for idx in spline_box_indices])

        # Determine spline axis from structural nodes
        span_range = np.ptp(struct_xyz, axis=0)
        span_axis = int(np.argmax(span_range))

        # Build interpolation matrix. 퇴화 판정은 IPS가 실제로 쓰는
        # 패널 국부 면내 투영에서 해야 한다(전역 x-y가 아니라).
        is_spline2 = hasattr(spline, 'dtor')
        e1, e2 = _spline_plane_frame(boxes, spline_box_indices)
        struct_2d = _project_plane(struct_xyz, e1, e2)
        is_collinear = _nodes_are_collinear(struct_xyz, plane_xy=struct_2d)

        G_ka_slope = None
        if is_spline2 or is_collinear:
            G_ka_wash = build_beam_spline(struct_xyz, wash_pts,
                                          axis=span_axis)
            G_ka_force = build_beam_spline(struct_xyz, force_pts,
                                           axis=span_axis)
            if is_collinear and not is_spline2:
                logger.info("    Spline %d: collinear nodes, using beam spline "
                           "instead of IPS", sid)
        else:
            dz = getattr(spline, 'dz', 0.0)
            # 패널 국부 면내 좌표로 투영해 스플라인한다 (수직/경사면
            # 에서 전역 x-y 투영이 퇴화하거나 면외 위치를 잃는 문제).
            wash_2d = _project_plane(wash_pts, e1, e2)
            force_2d = _project_plane(force_pts, e1, e2)
            G_ka_wash = build_ips_spline(struct_2d, wash_2d, dz)
            G_ka_force = build_ips_spline(struct_2d, force_2d, dz)
            if slope_method == 'surface':
                # SPLINE1 semantics: slope from the analytic x-derivative
                # of the interpolated displacement surface (z-translations
                # only; nodal rotations do not participate).
                # 기울기는 유선방향(e1) 도함수 — 투영 1축과 일치한다
                G_ka_slope = build_ips_spline_slope(struct_2d, wash_2d, dz)

        logger.info("  Spline %d: CAERO %d, boxes %d-%d (%d boxes), "
                     "%d struct nodes, axis=%d",
                     sid, caero_eid, box1, box2, len(spline_box_indices),
                     len(spline_nids), span_axis)

        # Fill both G matrices for this spline's boxes
        _fill_geff(G_w, G_d, G_ka_wash, G_ka_force, spline_box_indices,
                   spline_nids, force_pts, struct_xyz, dof_mgr, f_dof_index,
                   G_ka_slope=G_ka_slope, node_force=node_force)

    return _finish(G_w, G_d)


def _fill_geff(G_w: np.ndarray, G_d: np.ndarray, G_ka_wash: np.ndarray,
               G_ka_force: np.ndarray, box_indices: list, struct_nids: list,
               force_pts: np.ndarray, struct_xyz: np.ndarray,
               dof_mgr: DOFManager, f_dof_index: dict,
               G_ka_slope: np.ndarray = None,
               node_force: tuple = None) -> None:
    """Fill both normalwash (G_w) and displacement (G_d) coupling matrices.

    G_w: normalwash matrix — maps structural DOFs to normalwash (slope dz/dx),
        evaluated at the 3/4-chord collocation points (G_ka_wash).
        Only theta_y (DOF 5) contributes because slope = theta_y for beam spline.
        w_j = sum_k w_k * theta_y_k

    G_d: displacement matrix — maps structural DOFs to z-displacement at the
        1/4-chord doublet points where the box forces act (G_ka_force), so
        F_struct = G_d^T f_aero applies each force at its physical location.
        Both z (DOF 3) and theta_y (DOF 5) with lever arm dx contribute.
        z_j = sum_k w_k * (z_k + theta_y_k * dx_k)

    The aero stiffness Q_aa = G_d^T @ A_jj @ G_w uses G_d for force distribution
    and G_w for normalwash computation.

    When ``G_ka_slope`` is given (surface / SPLINE1 slope mode), the
    normalwash comes from the analytic x-derivative of the interpolated
    displacement surface: G_w takes the slope weights on the
    z-translation DOFs and nodal rotations do not participate at all
    (neither in G_w nor in G_d), matching SPLINE1 semantics and keeping
    the force-transfer path the virtual-work conjugate of the
    displacement interpolation.
    """
    n_local = len(box_indices)
    n_spline = len(struct_nids)

    # 물리공간 힘 분배 가중치: 자유도 소거와 무관하게 모든 SET 절점에
    # 채운다. 여기서 f-set 필터를 걸면 RBE2 종속/SPC 절점의 가중치가
    # 통째로 사라져 박스별 가중치 합(=1)이 깨지고, 뒤따르는 보존
    # 재스케일이 그 결손을 기체 전체에 균등하게 문질러 버린다.
    if node_force is not None:
        rows, cols, vals, node_index = node_force
        for i_local, i_box in enumerate(box_indices):
            for j_node in range(n_spline):
                w = G_ka_force[i_local, j_node]
                if abs(w) <= 1e-15:
                    continue
                j = node_index.get(struct_nids[j_node])
                if j is None:
                    continue
                rows.append(i_box); cols.append(j); vals.append(w)

    if G_ka_slope is not None:
        # ── surface (SPLINE1) mode: translations only ──
        for i_local, i_box in enumerate(box_indices):
            for j_node in range(n_spline):
                nid = struct_nids[j_node]
                z_dof = dof_mgr.get_dof(nid, 3)
                if z_dof not in f_dof_index:
                    continue
                w_disp = G_ka_force[i_local, j_node]
                w_slope = G_ka_slope[i_local, j_node]
                if abs(w_disp) > 1e-15:
                    G_d[i_box, f_dof_index[z_dof]] += w_disp
                if abs(w_slope) > 1e-15:
                    G_w[i_box, f_dof_index[z_dof]] += w_slope
        return

    for i_local, i_box in enumerate(box_indices):
        for j_node in range(n_spline):
            nid = struct_nids[j_node]
            w_wash = G_ka_wash[i_local, j_node]
            w_force = G_ka_force[i_local, j_node]
            if abs(w_wash) < 1e-15 and abs(w_force) < 1e-15:
                continue

            # DOF 3 (z-translation): contributes to displacement only
            z_dof = dof_mgr.get_dof(nid, 3)
            if z_dof in f_dof_index and abs(w_force) > 1e-15:
                G_d[i_box, f_dof_index[z_dof]] += w_force

            # DOF 5 (theta_y): contributes to BOTH normalwash and displacement
            # Normalwash: w_j = theta_y (direct slope contribution)
            # Displacement: z_j = theta_y * dx (lever arm effect)
            ry_dof = dof_mgr.get_dof(nid, 5)  # theta_y = pitch/torsion
            if ry_dof in f_dof_index:
                if abs(w_wash) > 1e-15:
                    G_w[i_box, f_dof_index[ry_dof]] += w_wash  # normalwash

                dx = force_pts[i_local, 0] - struct_xyz[j_node, 0]
                if abs(w_force) > 1e-15 and abs(dx) > 1e-12:
                    G_d[i_box, f_dof_index[ry_dof]] += w_force * dx


# ---------------------------------------------------------------------------
# Aerodynamic helpers
# ---------------------------------------------------------------------------

def _hinge_axis_sign(cid: int, bdf_model: BDFModel) -> float:
    """AESURF 힌지 좌표계(CID)의 회전축(y축) 전역 Y 부호.

    Nastran 규약상 조종면은 자기 힌지 좌표계 y축 둘레로 회전한다.
    에일러론 쌍은 좌/우 힌지 y축이 반대 방향(차동), 승강타 좌/우 반쪽은
    같은 방향(대칭)이라 이 부호가 면별 워시 방향을 결정한다.
    """
    coord = getattr(bdf_model, "coords", {}).get(cid)
    transform = getattr(coord, "transform", None) if coord else None
    if transform is None:
        return 1.0
    hinge_y = np.asarray(transform)[:, 1]
    return 1.0 if hinge_y[1] >= 0.0 else -1.0


def _get_control_surface_boxes(label: str, bdf_model: BDFModel,
                                box_id_to_index: Dict[int, int]
                                ) -> Tuple[Dict[int, float], float]:
    """Get {sequential box index: hinge sign} for a control surface.

    두 면(ALID1/ALID2)을 가진 AESURF는 각 면의 힌지 좌표계 방향에 따라
    부호를 부여한다 — 에일러론은 차동(+/-), 승강타 반쪽 쌍은 대칭(+/+).
    """
    for aid, aesurf in bdf_model.aesurfs.items():
        if aesurf.label == label:
            eff = aesurf.eff
            cs_signs: Dict[int, float] = {}

            for cid, alid in ((aesurf.cid1, aesurf.alid1),
                              (aesurf.cid2, aesurf.alid2)):
                if alid > 0 and alid in bdf_model.aelists:
                    sign = _hinge_axis_sign(cid, bdf_model)
                    aelist = bdf_model.aelists[alid]
                    for box_eid in aelist.elements:
                        if box_eid in box_id_to_index:
                            cs_signs[box_id_to_index[box_eid]] = sign

            if cs_signs:
                signs = sorted(set(cs_signs.values()))
                if len(signs) > 1:
                    logger.info("  Control surface '%s': differential "
                                "(hinge signs %s)", label, signs)
                return cs_signs, eff

            logger.warning("  Control surface '%s': no AELIST found, "
                           "applying to all boxes", label)
            return {i: 1.0 for i in range(len(box_id_to_index))}, eff

    return {}, 1.0


def _trim_variable_normalwash(label: str, boxes: List[AeroBox],
                               box_id_to_index: Dict[int, int],
                               bdf_model: BDFModel) -> np.ndarray:
    """Compute normalwash vector for a unit value of a trim variable.

    Vertical-velocity washes (ANGLEA, PITCH, ROLL, URDD5) are projected
    onto each panel normal via its z-component: horizontal panels see the
    full wash, vertical panels (VTP, normal ~ +/-Y) see none. Without the
    projection a symmetric trim puts spurious side load on the VTP
    (GACOMP M0.1: 1.36 kN VTP Fy from alpha alone).
    """
    n = len(boxes)
    w = np.zeros(n)

    if label == "ANGLEA":
        w[:] = -1.0

    elif label == "SIDES":
        # 옆미끄럼각: 측방 유동이 패널 법선 y성분을 통과한다 (ANGLEA의
        # -n_z에 대응하는 측방 워시 -n_y). 수평면 0, VTP 전량.
        for i in range(n):
            w[i] = -boxes[i].normal[1]

    elif label == "ROLL":
        # Roll rate (pb/2V): normalwash w_i = -2*y_i / b_ref
        # Rolling creates z-velocity proportional to y-position: v_z = -p*y
        # In nondimensional form: w = -2*y / b_ref  (for unit ROLL = pb/2V = 1)
        refb = bdf_model.aeros.refb if bdf_model.aeros else 1.0
        if refb > 1e-12:
            for i in range(n):
                w[i] = -2.0 * boxes[i].control_point[1] / refb

    elif label == "PITCH":
        # Pitch rate (qc/2V): rotation about CG creates normalwash
        # proportional to distance from CG.
        # v_z = q * (x - x_cg), normalwash w/V = q*(x-x_cg)/V
        # For unit PITCH = qc/(2V): q = 2V/c, so w = 2*(x-x_cg)/c
        # With sign (nose-up positive): w = -2*(x - x_cg) / c_ref
        refc = bdf_model.aeros.refc if bdf_model.aeros else 1.0
        x_cg = _compute_cg_x(bdf_model)
        if refc > 1e-12:
            for i in range(n):
                w[i] = -2.0 * (boxes[i].control_point[0] - x_cg) / refc

    elif label == "YAW":
        pass  # yaw rate (rb/2V): no direct z-normalwash for planar wings

    elif label.startswith("URDD"):
        if label == "URDD5":
            for i in range(n):
                w[i] = -boxes[i].control_point[0]

    else:
        # Try direct AESURF match first
        cs_signs, eff = _get_control_surface_boxes(label, bdf_model,
                                                    box_id_to_index)
        if cs_signs:
            for i, sign in cs_signs.items():
                w[i] = -eff * sign
            logger.info("    Control surface '%s': %d boxes, eff=%.3f",
                         label, len(cs_signs), eff)
        else:
            # Resolve via AELINK: label is an independent variable linked
            # to dependent physical surfaces
            if not _resolve_aelink_normalwash(label, w, bdf_model,
                                              box_id_to_index):
                logger.warning("    Control surface '%s' not found", label)

    # z-속도 기반 워시는 패널 법선 z성분으로 투영 (조종면 워시는
    # 자기 패널 법선 방향 처짐이므로 투영하지 않음)
    if label in ("ANGLEA", "PITCH", "ROLL", "URDD5"):
        nz = np.array([box.normal[2] for box in boxes])
        w *= nz

    return w


def _resolve_aelink_normalwash(label: str, w: np.ndarray,
                                bdf_model: BDFModel,
                                box_id_to_index: Dict[int, int]) -> bool:
    """Resolve AELINK: compute normalwash contribution for an independent
    variable via its linked dependent physical surfaces.

    For V-tail coupling (ELEVR = ELEV + RUD, ELEVL = ELEV - RUD):
    - normalwash("ELEV") = 1.0 * w(ELEVR) + 1.0 * w(ELEVL)
    - normalwash("RUD")  = 1.0 * w(ELEVR) - 1.0 * w(ELEVL)
    """
    aelinks = getattr(bdf_model, 'aelinks', [])
    if not aelinks:
        return False

    found = False
    for aelink in aelinks:
        for ind_label, coeff in aelink.links:
            if ind_label == label:
                dep_signs, dep_eff = _get_control_surface_boxes(
                    aelink.dependent, bdf_model, box_id_to_index)
                if dep_signs:
                    for i, sign in dep_signs.items():
                        w[i] += -dep_eff * coeff * sign
                    found = True
    if found:
        logger.info("    AELINK: '%s' resolved via linked surfaces", label)
    return found


# ---------------------------------------------------------------------------
# Weight, CG, and utility functions
# ---------------------------------------------------------------------------

def _detect_gravity(bdf_model: BDFModel) -> float:
    """모델 단위계에서 중력가속도를 추정한다 (mm/s^2 또는 m/s^2).

    우선순위:
      1) 덱의 GRAV 카드 크기 — 명시된 값이 있으면 그것이 정답이다.
      2) 기준 코드(REFC)와 기준 면적(REFS)의 정합 — 소형 mm 모델은
         REFC<100이어도 REFS가 코드^2 규모라 mm임이 드러난다.
      3) 모델 치수(최대 좌표 스팬).
    셋 다 없으면 mm 단위계(9810)로 본다. 추정에 쓰인 근거를 남긴다.
    """
    for load_list in (getattr(bdf_model, "loads", {}) or {}).values():
        for load in load_list:
            if getattr(load, "type", "") == "GRAV":
                a = float(np.linalg.norm(load.get_acceleration_vector()))
                if a > 1e-6:
                    logger.info("  중력을 GRAV 카드에서 취함: %.4g", a)
                    return a

    refc = refs = 0.0
    src = bdf_model.aeros or bdf_model.aero
    if src is not None:
        refc = float(getattr(src, "refc", 0.0) or 0.0)
        refs = float(getattr(src, "refs", 0.0) or 0.0)

    if refc > 100.0:
        return 9810.0
    if refc > 0.0 and refs > 0.0:
        # REFS/REFC^2 는 세장비 규모(수~수십)로 단위에 무관하다.
        # REFC가 작은데 REFS가 REFC^2 대비 정상이면 그 단위계가 맞다.
        return 9.81

    span = 0.0
    if bdf_model.nodes:
        xyz = np.array([n.xyz_global for n in bdf_model.nodes.values()])
        span = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
    if span > 0.0:
        g = 9810.0 if span > 100.0 else 9.81
        logger.info("  중력을 모델 치수(스팬 %.4g)로 추정: %.4g", span, g)
        return g
    return 9810.0


def _compute_total_weight(bdf_model: BDFModel) -> float:
    """Compute total weight from the SAME nodal mass lumping used for
    inertial loads (compute_node_masses), so the trim force target
    nz*W matches the inertial force sum exactly.

    A separate element-mass approximation here (the pre-2026-07 code)
    dropped NSM and lumped differently, leaving a permanent combined-Fz
    residual (GACOMP: 42 N = 0.33% W).
    """
    from ..loads_analysis.trim_loads import compute_node_masses

    total_mass = sum(compute_node_masses(bdf_model).values())
    g = _detect_gravity(bdf_model)
    weight = total_mass * g
    logger.info("  Total mass = %.4f (consistent units), Weight = %.2f, g = %.1f",
                total_mass, weight, g)
    return weight


def _spline_plane_frame(boxes, box_indices):
    """스플라인이 덮는 패널의 국부 면내 좌표계 (e1: 유선방향, e2: 스팬방향).

    IPS는 2차원 평면에 투영한 점들로 커널을 세운다. 전역 x-y를
    쓰면 수직 핀처럼 x-z 평면에 놓인 면은 투영이 퇴화하거나
    (수직미익) z 위치 정보를 잃는다(경사 V-미익). 패널 법선
    n에 수직인 평면을 잡아 그 안에서 스플라인하면 면 방향과
    무관하게 동작한다.

    수평면(n ~ +z)에서는 e1 ~ +x, e2 ~ +y가 되어 종전 전역 x-y
    투영과 같아진다.
    """
    n = np.zeros(3)
    for i in box_indices:
        n = n + np.asarray(boxes[i].normal, dtype=float)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n = n / nn

    # 유선방향: 전역 x를 패널 평면에 투영. n이 x와 나란하면 y로 대체
    e1 = np.array([1.0, 0.0, 0.0]) - n * float(n[0])
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.array([0.0, 1.0, 0.0]) - n * float(n[1])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def _project_plane(pts, e1, e2):
    """3D 점들을 (e1, e2) 면내 2D 좌표로 투영한다."""
    p = np.atleast_2d(np.asarray(pts, dtype=float))
    return np.column_stack([p @ e1, p @ e2])


def _nodes_are_collinear(xyz: np.ndarray, tol: float = 1e-6,
                         plane_xy: np.ndarray = None) -> bool:
    """IPS 스플라인이 퇴화하는 배치인지 판정한다.

    IPS는 x-y 평면에 투영한 점들로 커널을 세우므로, 3D에서
    공선이 아니어도 *투영이* 공선이면 특이해진다(예: 수직
    핀처럼 x-z 평면에 놓인 지지점 집합). 3D 공선만 검사하면
    그런 배치가 그대로 통과해 조용히 쓰레기 가중치가 나온다.
    투영 공선을 먼저 보고, 3D 공선도 함께 본다.
    """
    if xyz.shape[0] < 3:
        return True

    def _degenerate(points: np.ndarray) -> bool:
        centered = points - points.mean(axis=0)
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        if sv[0] < 1e-12:
            return True
        return sv[1] / sv[0] < tol

    # IPS 커널이 실제로 쓰는 면내 투영 (주어지면 그것을, 없으면 x-y)
    proj = xyz[:, :2] if plane_xy is None else np.asarray(plane_xy)
    if _degenerate(proj):
        return True
    return _degenerate(xyz)


def _compute_cg(bdf_model: BDFModel) -> np.ndarray:
    """Compute the CG position (3,) from the SAME nodal mass lumping used
    for inertial loads (compute_node_masses).

    The trim moment constraints take moments about this point; if it
    drifts from the inertial-load mass centroid, every trimmed case
    carries a permanent moment residual of W * dr (GACOMP x-drift:
    10.7 mm -> 135 kN-mm in My).
    """
    from ..loads_analysis.trim_loads import (compute_node_masses,
                                             compute_node_mass_centroids)

    node_mass = compute_node_masses(bdf_model)
    total_mass = sum(node_mass.values())
    if total_mass <= 1e-12:
        return np.zeros(3)
    # 질량 위치는 절점이 아니라 그 절점 집중질량의 질량중심이다
    # (CONM2 오프셋/CID=-1 절대좌표).
    centroids = compute_node_mass_centroids(bdf_model)
    moment = np.zeros(3)
    for nid, m in node_mass.items():
        if nid in centroids:
            moment += m * centroids[nid]
    return moment / total_mass


def _compute_cg_x(bdf_model: BDFModel) -> float:
    """CG x-coordinate (see _compute_cg)."""
    return float(_compute_cg(bdf_model)[0])
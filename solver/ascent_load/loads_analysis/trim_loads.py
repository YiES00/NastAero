"""Trim loads analysis: compute and verify nodal aerodynamic and inertial loads.

Given a converged SOL 144 trim solution, this module:
1. Transfers panel aerodynamic forces to structural nodes via spline transpose
2. Computes inertial (gravity) forces at each node for the trimmed load factor
3. Combines aero + inertial forces and verifies 6-DOF equilibrium (trim balance)
4. Outputs nodal force cards in Nastran FORCE format
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..bdf.model import BDFModel
from ..aero.panel import AeroBox
from ..config import logger


def compute_node_masses(bdf_model: BDFModel) -> Dict[int, float]:
    """Compute lumped mass at each node from element contributions.

    Uses the same mass lumping as gravity load computation:
    - CBAR/CBEAM/CROD: rho * A * L / 2 per node
    - CQUAD4: rho * t * area / 4 per node
    - CTRIA3: rho * t * area / 3 per node
    - CONM2: direct mass

    Returns
    -------
    node_mass : Dict[int, float]
        Node ID -> lumped mass value.
    """
    node_mass: Dict[int, float] = {}

    for eid, elem in bdf_model.elements.items():
        if not hasattr(elem, 'property_ref') or elem.property_ref is None:
            continue
        prop = elem.property_ref

        if elem.type in ("CBAR", "CBEAM", "CROD"):
            mat = getattr(prop, 'material_ref', None)
            if mat is None or mat.rho <= 0:
                continue
            n1 = bdf_model.nodes[elem.node_ids[0]]
            n2 = bdf_model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            em = mat.rho * prop.A * L + getattr(prop, 'nsm', 0.0) * L
            for nid in elem.node_ids:
                node_mass[nid] = node_mass.get(nid, 0.0) + em / 2.0

        elif elem.type == "CQUAD4":
            if hasattr(prop, 'equivalent_isotropic'):
                _, _, t, rho = prop.equivalent_isotropic()
            else:
                mat = getattr(prop, 'material_ref', None)
                if mat is None:
                    continue
                rho = mat.rho
                t = getattr(prop, 't', 0.0)
            if rho <= 0 or t <= 0:
                continue
            nids = elem.node_ids
            coords = np.array([bdf_model.nodes[nid].xyz_global for nid in nids])
            d13 = coords[2] - coords[0]
            d24 = coords[3] - coords[1]
            area = 0.5 * np.linalg.norm(np.cross(d13, d24))
            em = (rho * t + float(getattr(prop, 'nsm', 0.0) or 0.0)) * area
            m_per_node = em / 4.0
            for nid in nids:
                node_mass[nid] = node_mass.get(nid, 0.0) + m_per_node

        elif elem.type == "CTRIA3":
            if hasattr(prop, 'equivalent_isotropic'):
                _, _, t, rho = prop.equivalent_isotropic()
            else:
                mat = getattr(prop, 'material_ref', None)
                if mat is None:
                    continue
                rho = mat.rho
                t = getattr(prop, 't', 0.0)
            if rho <= 0 or t <= 0:
                continue
            nids = elem.node_ids
            coords = np.array([bdf_model.nodes[nid].xyz_global for nid in nids])
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            em = (rho * t + float(getattr(prop, 'nsm', 0.0) or 0.0)) * area
            m_per_node = em / 3.0
            for nid in nids:
                node_mass[nid] = node_mass.get(nid, 0.0) + m_per_node

    # CONM2 concentrated masses
    for mid, mass_elem in bdf_model.masses.items():
        nid = mass_elem.node_id
        node_mass[nid] = node_mass.get(nid, 0.0) + mass_elem.mass

    return node_mass


def compute_node_mass_centroids(bdf_model: BDFModel) -> Dict[int, np.ndarray]:
    """절점별 집중질량이 실제로 작용하는 위치(기본좌표계).

    구조 요소에서 집중된 질량은 절점 위치에 있지만, CONM2는 자체
    CG가 절점에서 떨어져 있을 수 있다(CID>0 오프셋, CID=0 기본
    좌표계 오프셋, CID=-1 절대좌표). 질량 중심을 절점 위치로
    간주하면 CG와 관성 하중의 모멘트 팔이 그만큼 어긋난다.

    Returns
    -------
    Dict[int, ndarray(3)]
        절점 ID -> 그 절점 집중질량의 질량중심 위치. CONM2 오프셋이
        없으면 절점 좌표와 같다.
    """
    node_mass = compute_node_masses(bdf_model)
    moment: Dict[int, np.ndarray] = {}
    conm2_mass: Dict[int, float] = {}

    for mass_elem in bdf_model.masses.values():
        nid = mass_elem.node_id
        node = bdf_model.nodes.get(nid)
        if node is None or mass_elem.mass <= 0.0:
            continue
        offset = np.asarray(getattr(mass_elem, 'offset', np.zeros(3)),
                            dtype=float)
        cid = int(getattr(mass_elem, 'cid', 0) or 0)
        if cid == -1:
            pos = offset                       # 기본좌표계 절대 CG
        else:
            if cid > 0 and cid in bdf_model.coords:
                offset = bdf_model.coords[cid].transform @ offset
            pos = node.xyz_global + offset
        moment[nid] = moment.get(nid, np.zeros(3)) + mass_elem.mass * pos
        conm2_mass[nid] = conm2_mass.get(nid, 0.0) + mass_elem.mass

    centroids: Dict[int, np.ndarray] = {}
    for nid, m_tot in node_mass.items():
        node = bdf_model.nodes.get(nid)
        if node is None:
            continue
        xyz = node.xyz_global
        m_c = conm2_mass.get(nid, 0.0)
        if m_c <= 0.0 or m_tot <= 1e-30:
            centroids[nid] = xyz
            continue
        # 구조 요소 몫은 절점에, CONM2 몫은 자기 CG에 있다
        centroids[nid] = (moment[nid] + (m_tot - m_c) * xyz) / m_tot
    return centroids


def compute_nodal_aero_forces(
    bdf_model: BDFModel,
    boxes: List[AeroBox],
    aero_forces: np.ndarray,
    G_eff_sparse,
    f_dofs: List[int],
    dof_mgr,
) -> Dict[int, np.ndarray]:
    """Transfer panel aerodynamic forces to structural nodes via spline.

    The spline matrix G_eff maps structural z-displacements to aero panel
    downwash. By virtual work, the transpose G_eff.T maps aero forces to
    structural forces. We use the spline z-weights to distribute the full
    3D force vector from each panel to structural nodes.

    Parameters
    ----------
    bdf_model : BDFModel
    boxes : list of AeroBox
    aero_forces : ndarray (n_boxes, 3)
        Aerodynamic force vector (Fx, Fy, Fz) per panel.
    G_eff_sparse : sparse matrix (n_boxes, n_free)
        Spline coupling matrix.
    f_dofs : list of int
        Free DOF indices.
    dof_mgr : DOFManager

    Returns
    -------
    nodal_forces : Dict[int, ndarray(6)]
        Node ID -> [Fx, Fy, Fz, Mx, My, Mz] aerodynamic force.
    """
    import scipy.sparse as sp

    n_boxes = len(boxes)
    n_free = len(f_dofs)
    nodal_forces: Dict[int, np.ndarray] = {}

    # Initialize all nodes to zero
    for nid in dof_mgr.node_ids:
        nodal_forces[nid] = np.zeros(6)

    # Build f_dofs lookup
    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}

    # For each aero panel, distribute its force to structural nodes
    # using the spline weights from G_eff
    G_csc = G_eff_sparse.tocsc() if sp.issparse(G_eff_sparse) else sp.csc_matrix(G_eff_sparse)

    # Strategy: For each box j, find which structural DOFs it connects to
    # G_eff[j, :] has nonzero entries at DOF indices corresponding to
    # z-translation and twist DOFs of structural nodes.
    # We use the z-translation weights to distribute the full 3D force.

    # Iterate over each box
    for j in range(n_boxes):
        F_j = aero_forces[j]  # (3,) force vector
        if np.linalg.norm(F_j) < 1e-30:
            continue

        # Get the row of G_eff for this box
        row = G_csc[j, :].toarray().ravel()  # (n_free,)

        # Find z-translation DOF weights only (not twist DOFs)
        # z-translation DOFs are component 3 (index 2 in 0-based within 6-DOF)
        for nid in dof_mgr.node_ids:
            z_dof_global = dof_mgr.get_dof(nid, 3)  # component 3 = z-trans
            if z_dof_global not in f_dof_index:
                continue
            f_idx = f_dof_index[z_dof_global]
            w = row[f_idx]
            if abs(w) < 1e-15:
                continue
            # Distribute full 3D force proportionally
            nodal_forces[nid][:3] += w * F_j

    return nodal_forces


def compute_nodal_aero_forces_fast(
    bdf_model: BDFModel,
    boxes: List[AeroBox],
    aero_forces: np.ndarray,
    G_eff_sparse,
    f_dofs: List[int],
    dof_mgr,
) -> Dict[int, np.ndarray]:
    """Vectorized version of aero force transfer via spline transpose.

    Uses G_eff.T to map each component of aero force to structural DOFs.
    For the z-component, this is exact (G_eff.T @ Fz).
    For x and y components, we use the z-weight distribution pattern.
    """
    import scipy.sparse as sp

    n_boxes = len(boxes)
    n_free = len(f_dofs)
    f_dof_set = set(f_dofs)

    # Initialize force vector in free-DOF space
    F_struct = np.zeros(n_free)

    G_csc = G_eff_sparse.tocsc() if sp.issparse(G_eff_sparse) else sp.csc_matrix(G_eff_sparse)

    # Build mapping: for each node, which f_dof index is its z-translation?
    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}
    nid_to_z_fidx = {}
    for nid in dof_mgr.node_ids:
        z_dof = dof_mgr.get_dof(nid, 3)
        if z_dof in f_dof_index:
            nid_to_z_fidx[nid] = f_dof_index[z_dof]

    # Collect z-weight per (box, node) pair
    # For each box, the G_eff row gives normalwash contribution from each DOF.
    # The z-translation DOFs give us the pure displacement weight.

    # Build a z-only spline matrix: (n_boxes x n_nodes) where entry [j,i]
    # is the weight of node i's z-displacement on box j's downwash
    sorted_nids = dof_mgr.node_ids
    nid_to_node_idx = {nid: i for i, nid in enumerate(sorted_nids)}
    n_nodes = len(sorted_nids)

    # Extract z-DOF columns from G_eff
    z_fidx_list = []
    node_idx_list = []
    for nid in sorted_nids:
        if nid in nid_to_z_fidx:
            z_fidx_list.append(nid_to_z_fidx[nid])
            node_idx_list.append(nid_to_node_idx[nid])

    if not z_fidx_list:
        return {nid: np.zeros(6) for nid in sorted_nids}

    z_fidx_arr = np.array(z_fidx_list)
    node_idx_arr = np.array(node_idx_list)

    # G_z = G_eff[:, z_fidx_arr] → (n_boxes x n_z_nodes)
    G_z = G_csc[:, z_fidx_arr].toarray()  # (n_boxes, n_z_nodes)

    # For each force component, compute nodal forces via transpose:
    # F_node_comp = G_z.T @ F_aero_comp
    nodal_forces: Dict[int, np.ndarray] = {}
    for nid in sorted_nids:
        nodal_forces[nid] = np.zeros(6)

    for comp in range(3):  # Fx, Fy, Fz
        f_comp = np.real(aero_forces[:, comp]).astype(float)
        # G_z.T @ f_comp → (n_z_nodes,)
        f_nodal = G_z.T @ f_comp
        for k, nidx in enumerate(node_idx_arr):
            nid = sorted_nids[nidx]
            nodal_forces[nid][comp] += f_nodal[k]

    return nodal_forces


def compute_nodal_inertial_forces(
    bdf_model: BDFModel,
    nz: float,
    g: float,
    ny: float = 0.0,
) -> Dict[int, np.ndarray]:
    """Compute inertial (gravity) forces at each node for given load factors.

    F_inertia_z = -m_node * nz * g * k_hat  (negative z for +nz)
    F_inertia_y = -m_node * ny * g * j_hat  (negative y for +ny)

    For 1g level flight: nz = 1.0, ny = 0.0.
    For yaw maneuvers: ny ≠ 0 from lateral acceleration.

    Parameters
    ----------
    bdf_model : BDFModel
    nz : float
        Vertical load factor (1.0 for 1g level flight).
    g : float
        Gravitational acceleration in model units.
    ny : float
        Lateral load factor (0.0 for symmetric flight).

    Returns
    -------
    nodal_forces : Dict[int, ndarray(6)]
        Node ID -> [Fx, Fy, Fz, Mx, My, Mz] inertial force.
    """
    node_mass = compute_node_masses(bdf_model)
    nodal_forces: Dict[int, np.ndarray] = {}

    centroids = compute_node_mass_centroids(bdf_model)

    for nid in bdf_model.nodes:
        f = np.zeros(6)
        m = node_mass.get(nid, 0.0)
        if m > 0:
            # Inertial load in -z direction for positive nz (upward acceleration)
            f[2] = -m * nz * g
            # Lateral inertial load for yaw/roll maneuvers
            if abs(ny) > 1e-6:
                f[1] = -m * ny * g
            # 질량 중심이 절점에서 떨어져 있으면(CONM2 오프셋) 그 힘을
            # 절점으로 옮기며 팔에 해당하는 모멘트가 생긴다.
            d = centroids.get(nid)
            if d is not None:
                d = d - bdf_model.nodes[nid].xyz_global
                if np.any(np.abs(d) > 1e-12):
                    f[3:6] += np.cross(d, f[:3])
        nodal_forces[nid] = f

    return nodal_forces


def _distribute_box_forces(aero_forces, G_force_node, dof_mgr
                            ) -> Dict[int, np.ndarray]:
    """박스 공력을 물리공간 스플라인 가중치로 절점에 분배한다.

    자유도 소거(RBE2 종속/SPC)와 무관한 행렬이므로 박스별 가중치 합이
    1로 유지되고, 뒤따르는 보존 재스케일이 사실상 항등이 된다.
    """
    node_ids = list(dof_mgr.node_ids)
    nodal: Dict[int, np.ndarray] = {nid: np.zeros(6) for nid in node_ids}
    GT = G_force_node.T
    for comp in range(3):
        f_nodal = GT @ np.real(aero_forces[:, comp]).astype(float)
        f_nodal = np.asarray(f_nodal).ravel()
        for k, nid in enumerate(node_ids):
            nodal[nid][comp] += f_nodal[k]
    return nodal


def compute_trim_nodal_loads(
    bdf_model: BDFModel,
    boxes: List[AeroBox],
    aero_forces: np.ndarray,
    G_eff_sparse,
    f_dofs: List[int],
    dof_mgr,
    nz: float = 1.0,
    g: float = 9810.0,
    ny: float = 0.0,
    G_force_node=None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Compute all three nodal force sets for a trimmed condition.

    Parameters
    ----------
    bdf_model : BDFModel
    boxes : list of AeroBox
    aero_forces : ndarray (n_boxes, 3)
    G_eff_sparse : sparse matrix
    f_dofs : list of int
    dof_mgr : DOFManager
    nz : float
        Vertical load factor.
    g : float
        Gravitational acceleration.
    ny : float
        Lateral load factor (0.0 for symmetric flight).
    G_force_node : sparse (n_boxes, n_nodes), optional
        물리공간 힘 분배 행렬. 주어지면 자유도 소거의 영향을 받지 않는
        이 행렬로 박스 힘을 분배한다(권장). 없으면 자유 자유도만 담은
        G_eff로 되돌아가며, 이 경우 RBE2 종속/SPC 절점의 가중치가
        빠져 분포가 왜곡될 수 있다.

    Returns
    -------
    aero_nodal : Dict[int, ndarray(6)]
    inertial_nodal : Dict[int, ndarray(6)]
    combined_nodal : Dict[int, ndarray(6)]
    """
    logger.info("Computing nodal trim loads...")

    # Aerodynamic forces
    if G_force_node is not None:
        aero_nodal = _distribute_box_forces(aero_forces, G_force_node, dof_mgr)
    else:
        aero_nodal = compute_nodal_aero_forces_fast(
            bdf_model, boxes, aero_forces, G_eff_sparse, f_dofs, dof_mgr)

    # --- Post-spline force conservation ---
    # The spline transpose G_z.T may not perfectly conserve total force
    # if the z-DOF interpolation weights don't partition unity (e.g. IPS
    # spline with non-trivial geometry). Scale each force component
    # independently so that total nodal force matches total panel force.
    total_panel = np.zeros(3)
    for comp in range(3):
        total_panel[comp] = float(np.sum(np.real(aero_forces[:, comp])))

    total_nodal_raw = np.zeros(3)
    for f in aero_nodal.values():
        total_nodal_raw += f[:3]

    for comp in range(3):
        if abs(total_nodal_raw[comp]) > 1.0 and abs(total_panel[comp]) > 1.0:
            scale = total_panel[comp] / total_nodal_raw[comp]
            if abs(scale - 1.0) > 1e-6:
                # 재스케일은 미세 보정용이다. 크게 벗어나면 분배
                # 가중치 자체가 결손된 것이므로(예: 자유도 소거로
                # 빠진 SET 절점) 균등 스케일이 분포를 왜곡한다.
                emit = logger.warning if abs(scale - 1.0) > 0.01 else logger.info
                emit("  Spline force conservation [%s]: "
                     "panel=%.1f, nodal=%.1f, scale=%.6f",
                     "XYZ"[comp], total_panel[comp],
                     total_nodal_raw[comp], scale)
                for nid in aero_nodal:
                    aero_nodal[nid][comp] *= scale

    # --- Residual force conservation ---
    # 위 스케일링은 |합|>1.0인 성분만 다룬다. 남은 작은 성분 차이(예:
    # 좌우 상쇄로 총합이 ~1 N인 측력)도 절점 하중 크기 비례로 분배해
    # 총힘을 성분별로 정확히 일치시킨다 — 그래야 아래 모멘트 보존이
    # 기준점에 무관해진다 (차이 1.3 N이 CG 팔에 곱해지면 Mz ~5 kN-mm).
    total_after = np.zeros(3)
    for f in aero_nodal.values():
        total_after += f[:3]
    dF = total_panel - total_after
    share_norms = {nid: float(np.linalg.norm(f[:3]))
                   for nid, f in aero_nodal.items()}
    share_total = sum(share_norms.values())
    if share_total > 1e-12 and np.linalg.norm(dF) > 1e-9:
        for nid, f in aero_nodal.items():
            f[:3] += (share_norms[nid] / share_total) * dF

    # --- Post-spline moment conservation ---
    # 힘 보존 스케일링 후에도 스플라인 전달(G^T)은 총모멘트까지 보존하지
    # 않는다 (GACOMP: 유효 하중중심이 x로 ~11 mm 이동해 My 136 kN-mm 누설).
    # 패널 힘(1/4-코드 더블릿 라인 작용)의 원점 기준 총모멘트와 절점 힘의
    # 총모멘트 차이를 절점 하중 크기에 비례한 우력으로 분배해 총모멘트를
    # 일치시킨다.
    if boxes:
        cp = np.array([b.doublet_point for b in boxes])
        f_panel = np.real(np.asarray(aero_forces)[:, :3])
        m_panel = np.cross(cp, f_panel).sum(axis=0)
        m_nodal = np.zeros(3)
        for nid, f in aero_nodal.items():
            if nid in bdf_model.nodes:
                m_nodal += np.cross(bdf_model.nodes[nid].xyz_global, f[:3])
                m_nodal += f[3:6]
        dm = m_panel - m_nodal
        norms = {nid: float(np.linalg.norm(f[:3]))
                 for nid, f in aero_nodal.items()}
        total_norm = sum(norms.values())
        if total_norm > 1e-12 and np.linalg.norm(dm) > 1e-9:
            logger.info("  Spline moment conservation: dM = "
                        "[%.1f, %.1f, %.1f]", dm[0], dm[1], dm[2])
            for nid, f in aero_nodal.items():
                f[3:6] += (norms[nid] / total_norm) * dm

    total_aero = np.zeros(3)
    for f in aero_nodal.values():
        total_aero += f[:3]
    logger.info("  Nodal aero forces: Fx=%.2f, Fy=%.2f, Fz=%.2f",
                total_aero[0], total_aero[1], total_aero[2])

    # Inertial forces
    inertial_nodal = compute_nodal_inertial_forces(bdf_model, nz, g, ny=ny)
    total_inertial = np.zeros(3)
    for f in inertial_nodal.values():
        total_inertial += f[:3]
    logger.info("  Nodal inertial forces: Fx=%.2f, Fy=%.2f, Fz=%.2f",
                total_inertial[0], total_inertial[1], total_inertial[2])

    # Combined
    combined_nodal: Dict[int, np.ndarray] = {}
    all_nids = set(aero_nodal.keys()) | set(inertial_nodal.keys())
    for nid in all_nids:
        f_aero = aero_nodal.get(nid, np.zeros(6))
        f_inertia = inertial_nodal.get(nid, np.zeros(6))
        combined_nodal[nid] = f_aero + f_inertia

    total_combined = np.zeros(3)
    for f in combined_nodal.values():
        total_combined += f[:3]
    logger.info("  Combined forces: Fx=%.2f, Fy=%.2f, Fz=%.2f",
                total_combined[0], total_combined[1], total_combined[2])

    return aero_nodal, inertial_nodal, combined_nodal


def apply_inertia_relief(
    bdf_model: BDFModel,
    inertial_nodal: Dict[int, np.ndarray],
    combined_nodal: Dict[int, np.ndarray],
    cg: np.ndarray,
    g: float,
) -> Dict[str, float]:
    """잔여 6분력을 강체 관성하중으로 정확히 닫는다 (하중 자기평형화).

    트림 방정식이 강제하지 못한 합력/합모멘트(예: §23.349 롤 기동에서
    에일러론 롤 모멘트, §23.351 러더킥의 요 모멘트)는 물리적으로 강체
    가속도로 흡수된다. 잔차 R=[F;M(CG)]에서 가속도를
        a = F_res / m_total,   ω̇ = I⁻¹ M_res   (I: CG 기준 관성텐서)
    로 풀고, 절점마다 f_i = -m_i (a + ω̇×r_i)를 관성·합산 하중에 더한다.
    합산 하중의 6분력이 정확히 0이 되어(자기평형) 구속점 가짜 집중하중
    없이 응력 모델에 적용할 수 있다.

    Parameters
    ----------
    bdf_model : BDFModel
    inertial_nodal, combined_nodal : Dict[int, ndarray(6)]
        in-place로 relief 하중이 더해진다.
    cg : ndarray(3)
        질량 CG (모멘트 기준점).
    g : float
        중력가속도 (하중배수 보고용).

    Returns
    -------
    dict
        relief_nx/ny/nz (하중배수 증분, g), p_dot/q_dot/r_dot (rad/s²).
    """
    node_mass = compute_node_masses(bdf_model)
    total_mass = sum(node_mass.values())
    if total_mass <= 1e-12:
        return {}

    # 잔여 합력/합모멘트 (CG 기준)
    F_res = np.zeros(3)
    M_res = np.zeros(3)
    for nid, f in combined_nodal.items():
        if nid not in bdf_model.nodes:
            continue
        r = bdf_model.nodes[nid].xyz_global - cg
        F_res += f[:3]
        M_res += np.cross(r, f[:3]) + f[3:6]

    # CG 기준 관성텐서 I = Σ m (|r|²E - r⊗r).
    # 팔은 절점이 아니라 그 절점 집중질량의 질량중심 기준이어야
    # Σm·r = 0 과 relief 합모멘트 상쇄가 정확히 성립한다.
    centroids = compute_node_mass_centroids(bdf_model)
    inertia = np.zeros((3, 3))
    for nid, m in node_mass.items():
        if nid not in centroids:
            continue
        r = centroids[nid] - cg
        inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    a_lin = F_res / total_mass                       # 선형 가속도
    omega_dot = np.linalg.lstsq(inertia, M_res, rcond=None)[0]  # 각가속도

    if np.linalg.norm(F_res) < 1e-9 and np.linalg.norm(M_res) < 1e-6:
        return {"relief_nx": 0.0, "relief_ny": 0.0, "relief_nz": 0.0,
                "p_dot": 0.0, "q_dot": 0.0, "r_dot": 0.0}

    # relief 분포 하중: 합력 -F_res, 합모멘트 -M_res 정확 (Σm·r=0, I 정의)
    for nid, m in node_mass.items():
        if nid not in centroids or m <= 0:
            continue
        r = centroids[nid] - cg
        f_rel = -m * (a_lin + np.cross(omega_dot, r))
        # 질량중심에 작용하는 힘을 절점으로 옮기는 오프셋 모멘트
        d = centroids[nid] - bdf_model.nodes[nid].xyz_global
        m_rel = np.cross(d, f_rel) if np.any(np.abs(d) > 1e-12) else None
        for target in (inertial_nodal, combined_nodal):
            if nid not in target:
                target[nid] = np.zeros(6)
            target[nid][:3] += f_rel
            if m_rel is not None:
                target[nid][3:6] += m_rel

    logger.info("  Inertia relief closure: |F_res|=%.2f, |M_res|=%.3e -> "
                "a=%s, omega_dot=%s",
                np.linalg.norm(F_res), np.linalg.norm(M_res),
                np.round(a_lin, 4), np.round(omega_dot, 5))
    return {
        "relief_nx": float(a_lin[0] / g),
        "relief_ny": float(a_lin[1] / g),
        "relief_nz": float(a_lin[2] / g),
        "p_dot": float(omega_dot[0]),
        "q_dot": float(omega_dot[1]),
        "r_dot": float(omega_dot[2]),
    }


def verify_trim_balance(
    bdf_model: BDFModel,
    combined_forces: Dict[int, np.ndarray],
    ref_point: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Check 6-DOF equilibrium of combined nodal forces.

    Computes the resultant force and moment about a reference point.
    For a properly trimmed condition, all 6 components should be near zero.

    Parameters
    ----------
    bdf_model : BDFModel
    combined_forces : Dict[int, ndarray(6)]
    ref_point : ndarray(3), optional
        Moment reference point. Defaults to the basic origin (0,0,0).
        (호출부가 CG를 원하면 명시적으로 넘긴다.)

    Returns
    -------
    balance : Dict[str, float]
        Keys: 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'
    """
    if ref_point is None:
        ref_point = np.zeros(3)

    total_force = np.zeros(3)
    total_moment = np.zeros(3)

    for nid, f in combined_forces.items():
        if nid not in bdf_model.nodes:
            continue
        pos = bdf_model.nodes[nid].xyz_global
        total_force += f[:3]
        # Moment about reference point
        r = pos - ref_point
        total_moment += np.cross(r, f[:3])
        # Add direct moments if any
        total_moment += f[3:6]

    balance = {
        'Fx': float(total_force[0]),
        'Fy': float(total_force[1]),
        'Fz': float(total_force[2]),
        'Mx': float(total_moment[0]),
        'My': float(total_moment[1]),
        'Mz': float(total_moment[2]),
    }

    logger.info("  Trim balance (6-DOF resultant):")
    logger.info("    Forces:  Fx=%+.4e  Fy=%+.4e  Fz=%+.4e",
                balance['Fx'], balance['Fy'], balance['Fz'])
    logger.info("    Moments: Mx=%+.4e  My=%+.4e  Mz=%+.4e",
                balance['Mx'], balance['My'], balance['Mz'])

    return balance


def write_force_cards(
    nodal_forces: Dict[int, np.ndarray],
    filepath: str,
    load_sid: int = 1,
    label: str = "COMBINED",
    cid: int = 0,
) -> None:
    """Write nodal forces in Nastran FORCE card format.

    Output format (fixed-8):
    FORCE   SID     G       CID     F       N1      N2      N3

    Parameters
    ----------
    nodal_forces : Dict[int, ndarray(6)]
    filepath : str
    load_sid : int
        Load set ID.
    label : str
        Comment label for the force set.
    cid : int
        Coordinate system ID (0 = basic).
    """
    with open(filepath, 'w') as f:
        f.write(f"$ {label} NODAL FORCES\n")
        f.write(f"$ Generated by ASCENT-Load SOL 144 Trim Loads Analysis\n")
        f.write("$\n")

        for nid in sorted(nodal_forces.keys()):
            fv = nodal_forces[nid]
            f_mag = np.linalg.norm(fv[:3])
            if f_mag < 1e-20:
                continue
            # Direction cosines
            n1, n2, n3 = fv[:3] / f_mag

            # Use Nastran fixed-16 format for precision
            f.write("FORCE*  %16d%16d%16d%16.8E\n" %
                    (load_sid, nid, cid, f_mag))
            f.write("*       %16.8E%16.8E%16.8E\n" %
                    (n1, n2, n3))

            # Write MOMENT card if rotational DOFs have values
            m_mag = np.linalg.norm(fv[3:6])
            if m_mag > 1e-20:
                mn1, mn2, mn3 = fv[3:6] / m_mag
                f.write("MOMENT* %16d%16d%16d%16.8E\n" %
                        (load_sid, nid, cid, m_mag))
                f.write("*       %16.8E%16.8E%16.8E\n" %
                        (mn1, mn2, mn3))

    logger.info("  FORCE cards written to: %s", filepath)

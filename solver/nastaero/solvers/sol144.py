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
- Control surface normalwash applied only to specific boxes
- Z-force AND pitch moment trim constraints
- CG-based moment reference point
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..fem.dof_manager import DOFManager
from ..bdf.model import BDFModel, Subcase
from ..aero.panel import generate_all_panels, get_box_index_map, AeroBox
from ..aero.dlm import build_aic_matrix, compute_aero_forces, circulation_to_delta_cp
from ..aero.spline import build_ips_spline, build_beam_spline
from ..output.result_data import ResultData, SubcaseResult
from ..config import logger
from typing import List, Dict, Tuple, Optional, Set


def solve_trim(bdf_model: BDFModel) -> ResultData:
    """Run SOL 144 static aeroelastic trim analysis."""
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)
    results = ResultData(title="NastAero SOL 144 - Static Aeroelastic Trim")

    # Generate aerodynamic panels with Nastran EID numbering
    logger.info("Generating aerodynamic panel mesh...")
    boxes = generate_all_panels(bdf_model, use_nastran_eid=True)
    n_boxes = len(boxes)
    logger.info("  %d aerodynamic boxes generated", n_boxes)

    if n_boxes == 0:
        logger.error("No aerodynamic boxes. Check CAERO1 cards.")
        return results

    # Build box_id → sequential index mapping
    box_id_to_index = get_box_index_map(boxes)

    # Get aero reference data
    aeros = bdf_model.aeros
    aero = bdf_model.aero
    if aeros is None and aero is None:
        logger.error("Missing AERO/AEROS card.")
        return results

    refc = aeros.refc if aeros else aero.refc
    refb = aeros.refb if aeros else 1.0
    refs = aeros.refs if aeros else 1.0
    velocity = aero.velocity if aero else 50.0

    # Process each subcase with TRIM
    subcases = bdf_model.subcases if bdf_model.subcases else [bdf_model.global_case]
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
        logger.info("Solving subcase %d, TRIM %d (M=%.3f, q=%.1f)...",
                     subcase.id, trim_id, trim.mach, trim.q)

        sc_result = _solve_trim_subcase(
            bdf_model, fe_model, boxes, box_id_to_index, trim, effective,
            refc, refb, refs, velocity)
        results.subcases.append(sc_result)

    return results


def _solve_trim_subcase(bdf_model: BDFModel, fe_model: FEModel,
                        boxes: List[AeroBox], box_id_to_index: Dict[int, int],
                        trim, subcase: Subcase,
                        refc: float, refb: float, refs: float,
                        velocity: float) -> SubcaseResult:
    """Solve a single trim subcase."""
    dof_mgr = fe_model.dof_mgr
    n_boxes = len(boxes)
    q = trim.q
    mach = trim.mach

    # 1. Get structural system (partitioned)
    K_ff, M_ff, F_f, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)
    n_free = len(f_dofs)

    # 2. Build AIC matrix (steady VLM)
    logger.info("  Building AIC matrix (%d x %d)...", n_boxes, n_boxes)
    D = build_aic_matrix(boxes, mach, reduced_freq=0.0)

    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        logger.warning("  AIC matrix singular, adding regularization")
        D_inv = np.linalg.inv(D + np.eye(n_boxes) * 1e-10)

    # 3. Build spline + DOF coupling matrix using per-spline mapping
    G_eff = _build_geff_per_spline(bdf_model, boxes, box_id_to_index,
                                    dof_mgr, f_dofs)

    logger.info("  G_eff: max = %.4f, nonzeros = %d / %d",
                np.max(np.abs(G_eff)) if G_eff.size > 0 else 0,
                np.count_nonzero(G_eff), G_eff.size)

    # 4. Build force diagonal (normalwash → force conversion)
    # delta_Cp = 2*gamma/chord, F = q*delta_Cp*area = 2*q*gamma*area/chord
    F_diag = np.zeros((n_boxes, n_boxes))
    for j in range(n_boxes):
        chord_j = boxes[j].chord
        if chord_j > 1e-12:
            F_diag[j, j] = 2.0 * q * boxes[j].area / chord_j

    # Normalwash-to-force matrix (in aero space)
    A_jj = F_diag @ D_inv  # (n_boxes x n_boxes)

    # Aero force matrix in free structural DOFs
    Q_aa_free = G_eff.T @ A_jj @ G_eff  # (n_free x n_free)

    logger.info("  Q_aa norm = %.2e, K_ff norm = %.2e",
                np.linalg.norm(Q_aa_free),
                np.linalg.norm(K_ff.toarray() if sp.issparse(K_ff) else K_ff))

    # 5. Parse trim variables
    trim_vars = {}
    for label, val in trim.variables:
        trim_vars[label] = val

    all_trim_labels = []
    for aid, aestat in bdf_model.aestats.items():
        all_trim_labels.append(aestat.label)
    for aid, aesurf in bdf_model.aesurfs.items():
        all_trim_labels.append(aesurf.label)

    fixed_labels = {}
    free_labels = []
    for label in all_trim_labels:
        if label in trim_vars:
            fixed_labels[label] = trim_vars[label]
        else:
            free_labels.append(label)

    n_trim_free = len(free_labels)
    logger.info("  Trim: %d fixed %s, %d free %s",
                len(fixed_labels), list(fixed_labels.keys()),
                n_trim_free, free_labels)

    # 6. Build normalwash vectors for trim variables
    w_fixed = np.zeros(n_boxes)
    for label, value in fixed_labels.items():
        w_contrib = _trim_variable_normalwash(label, boxes, box_id_to_index,
                                              bdf_model)
        w_fixed += value * w_contrib
    F_trim_fixed = G_eff.T @ A_jj @ w_fixed

    Q_ax = np.zeros((n_free, n_trim_free))
    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, box_id_to_index,
                                              bdf_model)
        Q_ax[:, k] = G_eff.T @ A_jj @ w_contrib

    # 7. Build trim constraint equations
    total_weight = _compute_total_weight(bdf_model)
    cg_x = _compute_cg_x(bdf_model)

    # Determine number of constraints based on free trim variables
    # For full aircraft: Fz balance + My balance = 2 constraints
    # For single-surface: just Fz balance = 1 constraint
    n_constraints = min(n_trim_free, 2)  # Up to 2: force + moment
    if n_constraints < 1:
        n_constraints = 0

    D_r = np.zeros((n_constraints, n_free))
    D_x = np.zeros((n_constraints, n_trim_free))
    rhs_trim = np.zeros(n_constraints)

    # Force summation vector (sum all box forces in z-direction)
    sum_force = np.ones(n_boxes)

    if n_constraints >= 1:
        # Z-force balance: sum aero Fz = weight
        D_r[0, :] = sum_force @ A_jj @ G_eff
        for k in range(n_trim_free):
            w_k = _trim_variable_normalwash(free_labels[k], boxes,
                                            box_id_to_index, bdf_model)
            D_x[0, k] = sum_force @ A_jj @ w_k
        F_z_fixed = sum_force @ A_jj @ w_fixed
        rhs_trim[0] = total_weight - F_z_fixed

    if n_constraints >= 2:
        # Pitch moment balance about CG: sum(F_z * (x_cp - x_cg)) = 0
        # Positive moment = nose-up
        moment_arm = np.array([boxes[i].control_point[0] - cg_x
                               for i in range(n_boxes)])
        D_r[1, :] = moment_arm @ A_jj @ G_eff
        for k in range(n_trim_free):
            w_k = _trim_variable_normalwash(free_labels[k], boxes,
                                            box_id_to_index, bdf_model)
            D_x[1, k] = moment_arm @ A_jj @ w_k
        M_y_fixed = moment_arm @ A_jj @ w_fixed
        rhs_trim[1] = -M_y_fixed  # Moment equilibrium: M_aero = 0

        logger.info("  Moment ref (CG_x) = %.4f m", cg_x)

    # 8. Assemble and solve
    #
    # Physical system:
    #   K*u = P_aero(u, x) + P_external
    #   P_aero = Q_aa*u + Q_ax*x + F_trim_fixed
    #
    # Rearranging:
    #   [K + Q_aa | -Q_ax] [u]   [F_f + F_trim_fixed]
    #   [  D_r    |  D_x ] [x] = [     rhs_trim      ]
    #
    n_total = n_free + n_trim_free
    K_dense = K_ff.toarray() if sp.issparse(K_ff) else K_ff

    if n_trim_free > 0:
        A_sys = np.zeros((n_total + n_constraints, n_total))
        rhs_sys = np.zeros(n_total + n_constraints)

        A_sys[:n_free, :n_free] = K_dense + Q_aa_free
        A_sys[:n_free, n_free:n_total] = -Q_ax  # negative: trim aero force on RHS
        rhs_sys[:n_free] = F_f + F_trim_fixed

        A_sys[n_total:n_total+n_constraints, :n_free] = D_r
        A_sys[n_total:n_total+n_constraints, n_free:n_total] = D_x
        rhs_sys[n_total:n_total+n_constraints] = rhs_trim

        logger.info("  Solving coupled system (%d x %d, %d constraints)...",
                     A_sys.shape[0], A_sys.shape[1], n_constraints)
        sol, residuals, rank, sv = np.linalg.lstsq(A_sys, rhs_sys, rcond=None)

        u_f = sol[:n_free]
        x_trim = sol[n_free:n_total]
    else:
        K_eff = K_dense + Q_aa_free
        rhs_eff = F_f + F_trim_fixed
        u_f = np.linalg.solve(K_eff, rhs_eff)
        x_trim = np.array([])

    # 9. Post-process
    ndof = dof_mgr.total_dof
    u_full = np.zeros(ndof)
    for i, dof in enumerate(f_dofs):
        u_full[dof] = u_f[i]

    # Total normalwash
    w_total = G_eff @ u_f + w_fixed
    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, box_id_to_index,
                                              bdf_model)
        w_total += x_trim[k] * w_contrib

    gamma = D_inv @ w_total
    delta_cp = circulation_to_delta_cp(boxes, gamma)
    aero_forces = compute_aero_forces(boxes, delta_cp, q)

    # Store results
    sc_result = SubcaseResult(subcase_id=subcase.id)
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
    logger.info("  Trim results:")
    for label, val in sc_result.trim_variables.items():
        if label == "ANGLEA":
            logger.info("    %s = %.4f rad (%.2f deg)", label, val, np.degrees(val))
        else:
            logger.info("    %s = %.6f", label, val)
    logger.info("  Max displacement = %.6e", np.max(np.abs(u_full)))
    total_lift = np.sum(aero_forces[:, 2])
    logger.info("  Total aero Fz = %.2f N (weight = %.2f N)", total_lift, total_weight)

    # Pitch moment about CG
    if n_boxes > 0:
        my_total = sum(aero_forces[i, 2] * (boxes[i].control_point[0] - cg_x)
                       for i in range(n_boxes))
        logger.info("  Pitch moment about CG = %.2f N*m", my_total)

    return sc_result


def _build_geff_per_spline(bdf_model: BDFModel, boxes: List[AeroBox],
                            box_id_to_index: Dict[int, int],
                            dof_mgr: DOFManager,
                            f_dofs: List[int]) -> np.ndarray:
    """Build G_eff matrix using per-spline mapping.

    Each SPLINE1/SPLINE2 card connects a range of aero boxes (BOX1..BOX2 or
    the CAERO1's boxes) to a set of structural nodes (SETG). This enables
    separate wing and tail interpolation.

    Parameters
    ----------
    bdf_model : BDFModel
    boxes : list of AeroBox
    box_id_to_index : dict
        Mapping from Nastran box_id to sequential index.
    dof_mgr : DOFManager
    f_dofs : list of int
        Free DOF indices.

    Returns
    -------
    G_eff : ndarray (n_boxes, n_free)
        Coupling matrix mapping free DOFs to aero z-displacement.
    """
    n_boxes = len(boxes)
    n_free = len(f_dofs)
    G_eff = np.zeros((n_boxes, n_free))

    # Build f_dofs lookup for fast index finding
    f_dof_index = {dof: idx for idx, dof in enumerate(f_dofs)}

    if not bdf_model.splines:
        # No splines defined, fall back to global beam spline
        all_nids = sorted(bdf_model.nodes.keys())
        struct_xyz = np.array([bdf_model.nodes[nid].xyz_global
                               for nid in all_nids])
        aero_pts = np.array([box.control_point for box in boxes])
        G_ka_z = build_beam_spline(struct_xyz, aero_pts, axis=1)
        _fill_geff(G_eff, G_ka_z, range(n_boxes), all_nids,
                   aero_pts, struct_xyz, dof_mgr, f_dof_index)
        return G_eff

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
        # Nastran convention: box IDs start at CAERO1 EID
        # BOX1/BOX2 = 0 means "use default" (full CAERO1 range)
        # If BOX1/BOX2 < CAERO1 EID, treat as sequential (legacy) offset
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
            # Legacy: BOX2 is a sequential count from CAERO1 start
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

        # Get aero control points for these boxes
        aero_pts = np.array([boxes[idx].control_point for idx in spline_box_indices])

        # Determine spline axis from structural nodes
        span_range = np.ptp(struct_xyz, axis=0)
        span_axis = int(np.argmax(span_range))

        # Build interpolation matrix
        is_spline2 = hasattr(spline, 'dtor')  # SPLINE2 has dtor field

        # Check if structural nodes are collinear (1D beam model)
        # IPS (2D thin-plate spline) becomes ill-conditioned for collinear nodes
        is_collinear = _nodes_are_collinear(struct_xyz)

        if is_spline2 or is_collinear:
            G_ka_z = build_beam_spline(struct_xyz, aero_pts, axis=span_axis)
            if is_collinear and not is_spline2:
                logger.info("    Spline %d: collinear nodes, using beam spline "
                           "instead of IPS", sid)
        else:
            dz = getattr(spline, 'dz', 0.0)
            G_ka_z = build_ips_spline(struct_xyz, aero_pts, dz)

        logger.info("  Spline %d: CAERO %d, boxes %d-%d (%d boxes), "
                     "%d struct nodes, axis=%d",
                     sid, caero_eid, box1, box2, len(spline_box_indices),
                     len(spline_nids), span_axis)

        # Fill G_eff for this spline's boxes
        _fill_geff(G_eff, G_ka_z, spline_box_indices, spline_nids,
                   aero_pts, struct_xyz, dof_mgr, f_dof_index)

    return G_eff


def _fill_geff(G_eff: np.ndarray, G_ka_z: np.ndarray,
               box_indices: list, struct_nids: list,
               aero_pts: np.ndarray, struct_xyz: np.ndarray,
               dof_mgr: DOFManager, f_dof_index: dict) -> None:
    """Fill G_eff matrix entries for a set of boxes and structural nodes.

    For each aero point k and structural node j:
      z_aero_k = sum_j G_ka[k,j] * (uz_j + theta_twist * dx)

    The twist DOF coupling depends on beam orientation:
    - Beam along Y: DOF 4 (Rx) is twist, dx = x_aero - x_struct
    - Beam along X: DOF 5 (Ry) is twist, dy = y_aero - y_struct
    """
    n_local = len(box_indices)
    n_spline = len(struct_nids)

    # Determine beam axis from structural node layout
    if n_spline >= 2:
        span_vec = struct_xyz[-1] - struct_xyz[0]
        span_vec_abs = np.abs(span_vec)
        beam_axis = int(np.argmax(span_vec_abs))
    else:
        beam_axis = 1  # default Y

    for i_local, i_box in enumerate(box_indices):
        for j_node in range(n_spline):
            nid = struct_nids[j_node]
            w_j = G_ka_z[i_local, j_node]
            if abs(w_j) < 1e-15:
                continue

            # z-translation (DOF 3)
            z_dof = dof_mgr.get_dof(nid, 3)
            if z_dof in f_dof_index:
                G_eff[i_box, f_dof_index[z_dof]] += w_j

            # Twist coupling: depends on beam orientation
            if beam_axis == 1:
                # Beam along Y: DOF 4 (Rx) is twist
                # Twist creates z-displacement proportional to chordwise offset
                dx = aero_pts[i_local, 0] - struct_xyz[j_node, 0]
                rx_dof = dof_mgr.get_dof(nid, 4)
                if rx_dof in f_dof_index and abs(dx) > 1e-12:
                    G_eff[i_box, f_dof_index[rx_dof]] += w_j * dx
            elif beam_axis == 0:
                # Beam along X: DOF 5 (Ry) is twist
                # Twist creates z-displacement proportional to spanwise offset
                dy = aero_pts[i_local, 1] - struct_xyz[j_node, 1]
                ry_dof = dof_mgr.get_dof(nid, 5)
                if ry_dof in f_dof_index and abs(dy) > 1e-12:
                    G_eff[i_box, f_dof_index[ry_dof]] += w_j * dy


def _get_control_surface_boxes(label: str, bdf_model: BDFModel,
                                box_id_to_index: Dict[int, int]
                                ) -> Tuple[Set[int], float]:
    """Get the set of sequential box indices for a control surface.

    Parameters
    ----------
    label : str
        Control surface label (e.g., "ELEV").
    bdf_model : BDFModel
    box_id_to_index : dict

    Returns
    -------
    cs_indices : set of int
        Sequential indices of boxes belonging to this control surface.
    eff : float
        Control surface effectiveness factor.
    """
    for aid, aesurf in bdf_model.aesurfs.items():
        if aesurf.label == label:
            eff = aesurf.eff
            cs_indices = set()

            # Get box IDs from AELIST (referenced by alid1/alid2)
            for alid in [aesurf.alid1, aesurf.alid2]:
                if alid > 0 and alid in bdf_model.aelists:
                    aelist = bdf_model.aelists[alid]
                    for box_eid in aelist.elements:
                        if box_eid in box_id_to_index:
                            cs_indices.add(box_id_to_index[box_eid])

            if cs_indices:
                return cs_indices, eff

            # Fallback: if no AELIST, use all boxes in the CAERO1
            # referenced by the spline that covers this surface
            # This is a less precise fallback
            logger.warning("  Control surface '%s': no AELIST found, "
                           "applying to all boxes", label)
            return set(range(len(box_id_to_index))), eff

    return set(), 1.0


def _trim_variable_normalwash(label: str, boxes: List[AeroBox],
                               box_id_to_index: Dict[int, int],
                               bdf_model: BDFModel) -> np.ndarray:
    """Compute normalwash vector for a unit value of a trim variable.

    For ANGLEA (angle of attack):
        w/V = -alpha per radian. Applied to ALL boxes.

    For control surfaces (AESURF):
        w/V = -eff per radian. Applied ONLY to boxes in the AELIST.

    Parameters
    ----------
    label : str
        Trim variable label.
    boxes : list of AeroBox
    box_id_to_index : dict
        Mapping from Nastran box_id to sequential index.
    bdf_model : BDFModel

    Returns
    -------
    w : ndarray (n_boxes,)
        Normalwash (w/V) per box for a unit value of the trim variable.
    """
    n = len(boxes)
    w = np.zeros(n)

    if label == "ANGLEA":
        w[:] = -1.0  # w/V = -alpha per radian, all boxes

    elif label == "SIDES":
        pass  # No z-normalwash for planar wing

    elif label.startswith("URDD"):
        if label == "URDD5":
            # Pitch acceleration: generates normalwash proportional to x
            for i in range(n):
                w[i] = -boxes[i].control_point[0]
        # URDD3 (vertical accel) doesn't produce normalwash directly

    else:
        # Control surface deflection - apply ONLY to specific boxes
        cs_indices, eff = _get_control_surface_boxes(label, bdf_model,
                                                      box_id_to_index)
        if cs_indices:
            for i in cs_indices:
                w[i] = -eff
            logger.info("    Control surface '%s': %d boxes, eff=%.3f",
                         label, len(cs_indices), eff)
        else:
            logger.warning("    Control surface '%s' not found", label)

    return w


def _compute_total_weight(bdf_model: BDFModel) -> float:
    """Compute total structural weight from mass properties."""
    total_mass = 0.0

    for eid, elem in bdf_model.elements.items():
        if not hasattr(elem, 'property_ref') or elem.property_ref is None:
            continue
        prop = elem.property_ref
        if not hasattr(prop, 'material_ref') or prop.material_ref is None:
            continue
        mat = prop.material_ref

        if elem.type == "CBAR":
            n1 = bdf_model.nodes[elem.node_ids[0]]
            n2 = bdf_model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            total_mass += mat.rho * prop.A * L

    for mid, mass_elem in bdf_model.masses.items():
        total_mass += mass_elem.mass

    weight = total_mass * 9.81
    logger.info("  Total mass = %.4f kg, Weight = %.2f N", total_mass, weight)
    return weight


def _nodes_are_collinear(xyz: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if a set of 3D points are approximately collinear.

    Uses PCA: if the second principal component has negligible variance
    compared to the first, the points are collinear.

    Parameters
    ----------
    xyz : ndarray (n, 3)
        Point coordinates.
    tol : float
        Relative tolerance for collinearity.

    Returns
    -------
    bool
        True if points are collinear.
    """
    if xyz.shape[0] < 3:
        return True
    centered = xyz - xyz.mean(axis=0)
    # Use singular values to check dimensionality
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-12:
        return True
    # Collinear if second singular value is negligible
    return s[1] / s[0] < tol


def _compute_cg_x(bdf_model: BDFModel) -> float:
    """Compute CG x-coordinate for moment reference.

    Uses mass-weighted average of structural element and CONM2 locations.
    """
    total_mass = 0.0
    moment_x = 0.0

    for eid, elem in bdf_model.elements.items():
        if not hasattr(elem, 'property_ref') or elem.property_ref is None:
            continue
        prop = elem.property_ref
        if not hasattr(prop, 'material_ref') or prop.material_ref is None:
            continue
        mat = prop.material_ref

        if elem.type == "CBAR":
            n1 = bdf_model.nodes[elem.node_ids[0]]
            n2 = bdf_model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            mass = mat.rho * prop.A * L
            x_mid = 0.5 * (n1.xyz_global[0] + n2.xyz_global[0])
            total_mass += mass
            moment_x += mass * x_mid

    for mid, mass_elem in bdf_model.masses.items():
        nid = mass_elem.node_id
        if nid in bdf_model.nodes:
            x = bdf_model.nodes[nid].xyz_global[0]
            total_mass += mass_elem.mass
            moment_x += mass_elem.mass * x

    if total_mass > 1e-12:
        return moment_x / total_mass
    return 0.0

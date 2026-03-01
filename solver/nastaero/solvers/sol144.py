"""SOL 144 - Static Aeroelastic Trim Analysis.

Solves the coupled structural-aerodynamic system for trim equilibrium.

The VLM (Vortex-Lattice Method, steady k=0) approach:
1. Build structural stiffness matrix K
2. Generate aerodynamic panels from CAERO1 cards
3. Build AIC matrix D such that {w/V} = [D]{gamma}  (gamma = Gamma/V)
4. Build spline + DOF coupling matrix G_eff
5. Form aero-structural coupling Q_aa in structural DOF space
6. Set up trim equations with force balance constraints
7. Solve coupled system for displacements and trim variables

Key normalization: gamma = Gamma/V (normalized circulation), so
  delta_Cp_j = 2 * gamma_j / chord_j
  F_j = q * delta_Cp_j * area_j
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..fem.dof_manager import DOFManager
from ..bdf.model import BDFModel, Subcase
from ..aero.panel import generate_all_panels, AeroBox
from ..aero.dlm import build_aic_matrix, compute_aero_forces, circulation_to_delta_cp
from ..aero.spline import build_ips_spline, build_beam_spline
from ..output.result_data import ResultData, SubcaseResult
from ..config import logger
from typing import List, Dict, Tuple


def solve_trim(bdf_model: BDFModel) -> ResultData:
    """Run SOL 144 static aeroelastic trim analysis."""
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)
    results = ResultData(title="NastAero SOL 144 - Static Aeroelastic Trim")

    # Generate aerodynamic panels
    logger.info("Generating aerodynamic panel mesh...")
    boxes = generate_all_panels(bdf_model)
    n_boxes = len(boxes)
    logger.info("  %d aerodynamic boxes generated", n_boxes)

    if n_boxes == 0:
        logger.error("No aerodynamic boxes. Check CAERO1 cards.")
        return results

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
            bdf_model, fe_model, boxes, trim, effective,
            refc, refb, refs, velocity)
        results.subcases.append(sc_result)

    return results


def _solve_trim_subcase(bdf_model: BDFModel, fe_model: FEModel,
                        boxes: List[AeroBox], trim, subcase: Subcase,
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

    # 3. Build spline + DOF coupling matrix
    # G_eff maps the free DOF vector to normalwash at aero control points.
    # For each aero box, the z-displacement depends on:
    #   - z-translation of nearby structural nodes (interpolated via spline)
    #   - x-rotation (theta_x) of nearby nodes * (y_aero - y_struct)  [for spanwise wing]
    #   - y-rotation (theta_y) of nearby nodes * -(x_aero - x_struct) [for chordwise offset]
    # This is the key: structural pitch rotation affects normalwash through
    # the chordwise offset between structural and aero grids.

    spline_struct_nids = _get_spline_nodes(bdf_model)
    if not spline_struct_nids:
        spline_struct_nids = sorted(bdf_model.nodes.keys())

    n_spline = len(spline_struct_nids)
    struct_xyz = np.array([bdf_model.nodes[nid].xyz_global
                           for nid in spline_struct_nids])
    aero_pts = np.array([box.control_point for box in boxes])

    logger.info("  Building spline matrix (%d aero -> %d struct)...",
                n_boxes, n_spline)

    # Use beam spline along span axis (y) for typical wing config
    # This is more robust than IPS for line-type structural models
    G_ka_z = build_beam_spline(struct_xyz, aero_pts, axis=1)

    # Build G_eff: maps free DOF vector to aero z-displacements
    # For each aero point k and structural node j:
    #   z_aero_k = sum_j G_ka[k,j] * (uz_j + theta_x_j * dy_kj - theta_y_j * dx_kj)
    # where dx_kj = x_aero_k - x_struct_j, dy_kj = y_aero_k - y_struct_j
    #
    # For a wing along y with bending in z:
    #   Structural beam along y: uz is bending, theta_x is twist about span axis
    #   theta_x rotation causes z-displacement = theta_x * dx at chordwise offset dx
    #   This is the critical torsion coupling!

    G_eff = np.zeros((n_boxes, n_free))

    for i_box in range(n_boxes):
        for j_node in range(n_spline):
            nid = spline_struct_nids[j_node]
            w_j = G_ka_z[i_box, j_node]
            if abs(w_j) < 1e-15:
                continue

            # Chordwise offset: x_aero - x_struct
            dx = aero_pts[i_box, 0] - struct_xyz[j_node, 0]

            # z-translation (DOF 3): uz contributes directly to aero z
            z_dof = dof_mgr.get_dof(nid, 3)
            if z_dof in f_dofs:
                G_eff[i_box, f_dofs.index(z_dof)] += w_j

            # x-rotation (DOF 4): theta_x (twist about x-axis)
            # For wing along y: theta_x rotation doesn't directly cause z-displacement
            # unless there's a y-offset (dy), but for the same spanwise station, dy≈0
            # So theta_x (roll) contribution is small for typical wings

            # y-rotation (DOF 5): theta_y (pitch about y-axis)
            # z_aero += theta_y * (-dx)  ... actually:
            # For small rotation theta_y about y-axis: z_new = z - x * theta_y
            # So dz = -dx * theta_y
            # But for CBAR along y, theta_y is bending rotation, not twist
            # For Goland wing: elastic axis along y, bending in z, twist about y
            # Actually: CBAR along y with v-vector (0,0,1) → bending in XZ plane
            # DOF 4 = rotation about x (twist for wing along y!)
            # DOF 5 = rotation about y (bending curvature)

            # For wing along y-axis:
            #   DOF 4 (Rx, theta_x): TWIST about spanwise axis
            #     z_aero = theta_x * dx  (chordwise offset creates z-displacement)
            #   DOF 5 (Ry, theta_y): bending curvature
            #     Small effect, skip for now

            rx_dof = dof_mgr.get_dof(nid, 4)  # theta_x (twist)
            if rx_dof in f_dofs and abs(dx) > 1e-12:
                # Twist about x-axis: positive theta_x with positive dx gives positive dz
                G_eff[i_box, f_dofs.index(rx_dof)] += w_j * dx

    logger.info("  G_eff: max = %.4f, nonzeros = %d / %d",
                np.max(np.abs(G_eff)), np.count_nonzero(G_eff),
                G_eff.size)

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
        w_contrib = _trim_variable_normalwash(label, boxes, bdf_model)
        w_fixed += value * w_contrib
    F_trim_fixed = G_eff.T @ A_jj @ w_fixed

    Q_ax = np.zeros((n_free, n_trim_free))
    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, bdf_model)
        Q_ax[:, k] = G_eff.T @ A_jj @ w_contrib

    # 7. Build trim constraint equations
    total_weight = _compute_total_weight(bdf_model)

    # Force summation vector (sum all box forces in z-direction)
    sum_force = np.ones(n_boxes)

    n_constraints = min(n_trim_free, 1)
    D_r = np.zeros((n_constraints, n_free))
    D_x = np.zeros((n_constraints, n_trim_free))
    rhs_trim = np.zeros(n_constraints)

    if n_constraints >= 1:
        # Z-force balance: sum aero Fz = weight
        D_r[0, :] = sum_force @ A_jj @ G_eff
        for k in range(n_trim_free):
            w_k = _trim_variable_normalwash(free_labels[k], boxes, bdf_model)
            D_x[0, k] = sum_force @ A_jj @ w_k
        F_z_fixed = sum_force @ A_jj @ w_fixed
        rhs_trim[0] = total_weight - F_z_fixed

    # 8. Assemble and solve
    #
    # Physical system:
    #   K*u = P_aero(u, x) + P_external
    #   P_aero = Q_aa*u + Q_ax*x + F_trim_fixed  (aero forces from deformation + trim vars)
    #
    # Rearranging:
    #   (K - Q_aa)*u - Q_ax*x = P_external + F_trim_fixed
    #
    # But Q_aa here represents restoring aero feedback (negative for positive displacement),
    # so K + Q_aa gives increased stiffness. The trim variable force Q_ax*x acts as
    # external loading and goes on the opposite side:
    #
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

        logger.info("  Solving coupled system (%d x %d)...",
                     A_sys.shape[0], A_sys.shape[1])
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
        w_contrib = _trim_variable_normalwash(label, boxes, bdf_model)
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

    return sc_result


def _get_spline_nodes(bdf_model: BDFModel) -> List[int]:
    """Get structural node IDs referenced by spline SET cards."""
    spline_nids = set()
    for sid, spline in bdf_model.splines.items():
        setg = spline.setg
        if setg in bdf_model.sets:
            spline_nids.update(bdf_model.sets[setg].ids)
    return sorted(spline_nids)


def _trim_variable_normalwash(label: str, boxes: List[AeroBox],
                               bdf_model: BDFModel) -> np.ndarray:
    """Compute normalwash vector for a unit value of a trim variable.

    For ANGLEA (angle of attack):
        w/V = -alpha per radian.
        Positive alpha → flow from below → negative normalwash → positive lift.

    Parameters
    ----------
    label : str
        Trim variable label.
    boxes : list of AeroBox
    bdf_model : BDFModel

    Returns
    -------
    w : ndarray (n_boxes,)
        Normalwash (w/V) per box for a unit value of the trim variable.
    """
    n = len(boxes)
    w = np.zeros(n)

    if label == "ANGLEA":
        w[:] = -1.0  # w/V = -alpha per radian

    elif label == "SIDES":
        pass  # No z-normalwash for planar wing

    elif label.startswith("URDD"):
        if label == "URDD5":
            # Pitch acceleration: generates normalwash proportional to x
            for i in range(n):
                w[i] = -boxes[i].control_point[0]
        # URDD3 (vertical accel) doesn't produce normalwash directly

    else:
        # Control surface deflection
        for aid, aesurf in bdf_model.aesurfs.items():
            if aesurf.label == label:
                eff = aesurf.eff
                for i in range(n):
                    w[i] = -eff
                break

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

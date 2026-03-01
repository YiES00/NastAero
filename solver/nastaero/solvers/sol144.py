"""SOL 144 - Static Aeroelastic Trim Analysis.

Solves the coupled structural-aerodynamic system for trim equilibrium:
1. Build structural stiffness matrix K
2. Generate aerodynamic panels from CAERO1 cards
3. Build AIC matrix (steady VLM)
4. Build spline interpolation matrix G_ka
5. Form aero-structural coupling: Q_aa = G_ka^T * Q_jj * G_ka
6. Set up trim equations with constraint conditions
7. Solve coupled system for displacements and trim variables
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..fem.dof_manager import DOFManager
from ..bdf.model import BDFModel, Subcase
from ..aero.panel import generate_all_panels, AeroBox
from ..aero.dlm import build_aic_matrix, compute_aero_forces
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

    # Process each subcase with TRIM
    subcases = bdf_model.subcases if bdf_model.subcases else [bdf_model.global_case]
    for subcase in subcases:
        effective = bdf_model.get_effective_subcase(subcase)
        trim_id = effective.trim_id
        if trim_id == 0:
            # Check global case
            trim_id = bdf_model.global_case.trim_id
        if trim_id == 0:
            # Use first available trim
            if bdf_model.trims:
                trim_id = next(iter(bdf_model.trims.keys()))

        if trim_id not in bdf_model.trims:
            logger.warning("Subcase %d: no TRIM card found", subcase.id)
            continue

        trim = bdf_model.trims[trim_id]
        logger.info("Solving subcase %d, TRIM %d (M=%.3f, q=%.1f)...",
                     subcase.id, trim_id, trim.mach, trim.q)

        sc_result = _solve_trim_subcase(
            bdf_model, fe_model, boxes, trim, effective, refc, refb, refs)
        results.subcases.append(sc_result)

    return results


def _solve_trim_subcase(bdf_model: BDFModel, fe_model: FEModel,
                        boxes: List[AeroBox], trim, subcase: Subcase,
                        refc: float, refb: float, refs: float) -> SubcaseResult:
    """Solve a single trim subcase."""
    dof_mgr = fe_model.dof_mgr
    n_boxes = len(boxes)
    q = trim.q
    mach = trim.mach

    # 1. Get structural system (partitioned)
    K_ff, M_ff, F_f, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)
    n_free = len(f_dofs)

    # 2. Build AIC matrix (steady)
    logger.info("  Building AIC matrix (%d x %d)...", n_boxes, n_boxes)
    D = build_aic_matrix(boxes, mach, reduced_freq=0.0)

    # Invert AIC: delta_cp = D^{-1} * w/V
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        logger.warning("  AIC matrix singular, adding regularization")
        D_inv = np.linalg.inv(D + np.eye(n_boxes) * 1e-10)

    # 3. Build spline matrix
    # Find structural nodes in spline sets
    spline_struct_nids = _get_spline_nodes(bdf_model)
    if not spline_struct_nids:
        # Use all structural nodes
        spline_struct_nids = sorted(bdf_model.nodes.keys())

    struct_xyz = np.array([bdf_model.nodes[nid].xyz_global
                           for nid in spline_struct_nids])
    aero_pts = np.array([box.control_point for box in boxes])

    logger.info("  Building spline matrix (%d aero -> %d struct)...",
                n_boxes, len(spline_struct_nids))
    G_ka = build_ips_spline(struct_xyz, aero_pts, dz=0.0)

    # 4. Map spline nodes to free DOF indices
    # G_ka maps z-displacements: we need to extract z-DOF (component 3)
    # from the free DOF vector
    z_dof_map = []  # Maps spline node index to free DOF index
    for nid in spline_struct_nids:
        z_dof = dof_mgr.get_dof(nid, 3)  # z-translation
        if z_dof in f_dofs:
            z_dof_map.append(f_dofs.index(z_dof))
        else:
            z_dof_map.append(-1)  # constrained

    # Build mapping matrix: T_zs (n_struct_spline x n_free_dof)
    # Extracts z-displacements from the full free DOF vector
    n_spline = len(spline_struct_nids)
    T_zs = np.zeros((n_spline, n_free))
    for i, fdof_idx in enumerate(z_dof_map):
        if fdof_idx >= 0:
            T_zs[i, fdof_idx] = 1.0

    # 5. Build aero force matrix in structural DOFs
    # Normalwash due to structural deformation:
    #   w_j = G_ka * T_zs * u_f (vertical velocity)
    # Pressure from AIC: delta_cp = D^{-1} * w_j
    # Force on aero boxes: f_j = q * delta_cp * A_j * n_j
    # Force transferred to structure: F_aero = T_zs^T * G_ka^T * (q * S * D^{-1} * G_ka * T_zs) * u_f

    # S = diagonal area matrix
    S = np.diag([box.area for box in boxes])

    # Q_jj = S * D^{-1} (aero influence in force space)
    Q_jj = S @ D_inv

    # Q_aa = G_ka^T * Q_jj * G_ka (in spline node space, z-DOFs)
    # Then map to free DOF space
    G_eff = G_ka @ T_zs  # (n_boxes x n_free)
    Q_aa_free = q * G_eff.T @ Q_jj @ G_eff  # (n_free x n_free)

    # 6. Parse trim variables
    # Fixed vs free trim variables
    trim_vars = {}  # label -> value (fixed) or None (free)
    for label, val in trim.variables:
        trim_vars[label] = val

    # Determine free trim variables (not set in TRIM card)
    # Standard rigid body DOFs from AESTAT
    all_trim_labels = []
    for aid, aestat in bdf_model.aestats.items():
        all_trim_labels.append(aestat.label)
    # Add control surfaces from AESURF
    for aid, aesurf in bdf_model.aesurfs.items():
        all_trim_labels.append(aesurf.label)

    fixed_labels = {}
    free_labels = []
    for label in all_trim_labels:
        if label in trim_vars:
            fixed_labels[label] = trim_vars[label]
        # Labels set to FREE or not specified are free
        # In Nastran, labels listed in TRIM with value = free
        # For simplicity, any AESTAT not in TRIM card is free

    # For now, treat specified values as fixed, unspecified as free
    for label in all_trim_labels:
        if label not in trim_vars:
            free_labels.append(label)

    # Also check for "FREE" keyword
    for label, val in trim.variables:
        if label not in [a.label for _, a in bdf_model.aestats.items()] and \
           label not in [a.label for _, a in bdf_model.aesurfs.items()]:
            # Label might be a free variable indicator
            pass

    n_trim_free = len(free_labels)
    logger.info("  Trim variables: %d fixed, %d free", len(fixed_labels), n_trim_free)

    # 7. Build normalwash vectors for trim variables
    # For angle of attack (ANGLEA): w_alpha = -1 (unit downwash per radian)
    # For sideslip (SIDES): lateral component
    # For control surfaces (AESURF): w from deflection

    Q_ax = np.zeros((n_free, n_trim_free))  # Aero force due to trim vars
    w_fixed = np.zeros(n_boxes)  # Fixed trim contribution to normalwash

    for label, value in fixed_labels.items():
        w_contrib = _trim_variable_normalwash(label, boxes, bdf_model)
        w_fixed += value * w_contrib

    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, bdf_model)
        # Force contribution: q * G_eff^T * Q_jj * w_contrib
        Q_ax[:, k] = q * G_eff.T @ Q_jj @ w_contrib

    # Force from fixed trim variables
    F_trim_fixed = q * G_eff.T @ Q_jj @ w_fixed

    # 8. Assemble and solve coupled system
    # [K_ff + q*Q_aa   Q_ax ] [u_f ] = [F_f + F_trim_fixed]
    # [   D_r           D_x ] [x   ]   [     rhs_trim     ]

    # Trim constraint equations:
    # For level flight: sum(Fz) = W (weight), sum(My) = 0 (pitch moment)
    # We'll use rigid body equilibrium constraints

    # Build rigid body force summation rows
    n_constraints = min(n_trim_free, 2)  # Fz balance + My balance typically

    D_r = np.zeros((n_constraints, n_free))
    D_x = np.zeros((n_constraints, n_trim_free))
    rhs_trim = np.zeros(n_constraints)

    # Total weight (gravity loads)
    total_weight = _compute_total_weight(bdf_model)

    if n_constraints >= 1:
        # Force balance in z-direction
        # Sum of structural z-forces from aero = weight
        for i, nid in enumerate(spline_struct_nids):
            z_dof = dof_mgr.get_dof(nid, 3)
            if z_dof in f_dofs:
                fi = f_dofs.index(z_dof)
                # Aero force contribution from deformation
                D_r[0, fi] = np.sum(Q_aa_free[:, fi])
                # Aero force contribution per box (z-component)
                for bj in range(n_boxes):
                    D_r[0, fi] += 0  # Already included in Q_aa

        # Actually use simpler approach: total z-force balance
        # Sum of all aero z-forces = Weight
        for k in range(n_trim_free):
            w_contrib = _trim_variable_normalwash(free_labels[k], boxes, bdf_model)
            f_aero_k = q * Q_jj @ w_contrib
            # Z-component of force for each box
            for j in range(n_boxes):
                D_x[0, k] += f_aero_k[j] * boxes[j].normal[2] * boxes[j].area

        rhs_trim[0] = total_weight  # z-force = weight

    if n_constraints >= 2 and n_trim_free >= 2:
        # Pitching moment balance about reference point (typically CG)
        # My = 0 for trimmed flight
        x_ref = 0.0
        if bdf_model.aeros:
            x_ref = refc * 0.25  # Assume moment ref at quarter chord
        for k in range(n_trim_free):
            w_contrib = _trim_variable_normalwash(free_labels[k], boxes, bdf_model)
            f_aero_k = q * Q_jj @ w_contrib
            for j in range(n_boxes):
                fz = f_aero_k[j] * boxes[j].normal[2] * boxes[j].area
                arm = boxes[j].control_point[0] - x_ref
                D_x[1, k] += fz * arm
        rhs_trim[1] = 0.0  # Pitch moment = 0

    # Assemble full system
    n_total = n_free + n_trim_free
    if n_trim_free > 0:
        A = np.zeros((n_total + n_constraints, n_total))
        rhs = np.zeros(n_total + n_constraints)

        # Structural equations: (K + q*Q_aa)*u + Q_ax*x = F + F_trim_fixed
        K_dense = K_ff.toarray() if sp.issparse(K_ff) else K_ff
        A[:n_free, :n_free] = K_dense + Q_aa_free
        A[:n_free, n_free:n_total] = Q_ax
        rhs[:n_free] = F_f + F_trim_fixed

        # Trim constraints
        A[n_total:n_total+n_constraints, :n_free] = D_r
        A[n_total:n_total+n_constraints, n_free:n_total] = D_x
        rhs[n_total:n_total+n_constraints] = rhs_trim

        # Solve least-squares (overdetermined system)
        logger.info("  Solving coupled system (%d x %d)...", A.shape[0], A.shape[1])
        sol, residuals, rank, sv = np.linalg.lstsq(A, rhs, rcond=None)

        u_f = sol[:n_free]
        x_trim = sol[n_free:n_total]
    else:
        # No free trim variables - just solve structural
        K_dense = K_ff.toarray() if sp.issparse(K_ff) else K_ff
        K_eff = K_dense + Q_aa_free
        rhs_eff = F_f + F_trim_fixed
        u_f = np.linalg.solve(K_eff, rhs_eff)
        x_trim = np.array([])

    # 9. Post-process results
    # Reconstruct full displacement vector
    ndof = dof_mgr.total_dof
    u_full = np.zeros(ndof)
    for i, dof in enumerate(f_dofs):
        u_full[dof] = u_f[i]

    # Compute aero pressures
    w_total = G_eff @ u_f + w_fixed
    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, bdf_model)
        w_total += x_trim[k] * w_contrib

    delta_cp = D_inv @ w_total
    aero_forces = compute_aero_forces(boxes, delta_cp, q)

    # Log results
    sc_result = SubcaseResult(subcase_id=subcase.id)
    for nid in dof_mgr.node_ids:
        nd = dof_mgr.get_node_dofs(nid)
        sc_result.displacements[nid] = u_full[nd[0]:nd[5]+1]

    # Store trim results
    sc_result.trim_variables = {}
    for label, val in fixed_labels.items():
        sc_result.trim_variables[label] = val
    for k, label in enumerate(free_labels):
        sc_result.trim_variables[label] = x_trim[k]

    sc_result.aero_pressures = delta_cp
    sc_result.aero_forces = aero_forces
    sc_result.aero_boxes = boxes

    # Log trim results
    logger.info("  Trim results:")
    for label, val in sc_result.trim_variables.items():
        if label == "ANGLEA":
            logger.info("    %s = %.4f rad (%.2f deg)", label, val, np.degrees(val))
        else:
            logger.info("    %s = %.6f", label, val)
    logger.info("  Max displacement = %.6e", np.max(np.abs(u_full)))
    logger.info("  Total aero Fz = %.2f", np.sum(aero_forces[:, 2]))

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

    Parameters
    ----------
    label : str
        Trim variable label (ANGLEA, SIDES, control surface name, etc.)
    boxes : list of AeroBox
    bdf_model : BDFModel

    Returns
    -------
    w : ndarray (n_boxes,)
        Normalwash (w/V) per box for unit trim variable.
    """
    n = len(boxes)
    w = np.zeros(n)

    if label == "ANGLEA":
        # Angle of attack: uniform downwash w/V = -alpha
        # (negative because positive alpha means flow comes from below)
        for i in range(n):
            w[i] = -1.0  # per radian

    elif label == "SIDES":
        # Sideslip angle: lateral wash
        for i in range(n):
            w[i] = 0.0  # No z-normalwash from sideslip for planar wing

    elif label.startswith("URDD"):
        # Rigid body accelerations (not used in steady trim directly)
        pass

    else:
        # Control surface deflection
        # Find AESURF with this label
        for aid, aesurf in bdf_model.aesurfs.items():
            if aesurf.label == label:
                # Find the AELIST (control surface box IDs)
                # For now, apply unit deflection to all boxes
                # (simplified - should only affect control surface boxes)
                eff = aesurf.eff
                for i in range(n):
                    # Simple model: all boxes get equal wash from control
                    w[i] = -eff * 0.1  # Rough approximation
                break

    return w


def _compute_total_weight(bdf_model: BDFModel) -> float:
    """Compute total structural weight from mass properties."""
    total_mass = 0.0

    # From structural elements
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

    # From concentrated masses
    for mid, mass_elem in bdf_model.masses.items():
        total_mass += mass_elem.mass

    weight = total_mass * 9.81  # gravity
    logger.info("  Total mass = %.4f kg, Weight = %.2f N", total_mass, weight)
    return weight

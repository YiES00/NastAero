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
        logger.info("Solving subcase %d, TRIM %d (M=%.3f, q=%.6g)...",
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
    """Solve a single trim subcase.

    For large models (n_free > 10000), uses an iterative solver approach
    that avoids forming the dense Q_aa matrix. The aero contribution
    Q_aa * u = G_eff.T @ (A_jj @ (G_eff @ u)) is computed as a sequence
    of small matrix-vector products through the aero space (n_boxes).
    """
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

    # Convert G_eff to sparse for memory-efficient operations
    G_sp = sp.csr_matrix(G_eff)
    del G_eff  # Free dense memory

    # 4. Build force diagonal (normalwash → force conversion)
    # delta_Cp = 2*gamma/chord, F = q*delta_Cp*area = 2*q*gamma*area/chord
    f_diag_vec = np.zeros(n_boxes)
    for j in range(n_boxes):
        chord_j = boxes[j].chord
        if chord_j > 1e-12:
            f_diag_vec[j] = 2.0 * q * boxes[j].area / chord_j

    # A_jj = F_diag @ D_inv  (n_boxes x n_boxes, dense but small)
    A_jj = np.diag(f_diag_vec) @ D_inv

    # Helper: apply Q_aa to a vector without forming full matrix
    # Q_aa * v = G.T @ A_jj @ (G @ v)
    def apply_Q_aa(v):
        """Compute Q_aa @ v = G_eff.T @ A_jj @ (G_eff @ v) efficiently."""
        w = G_sp @ v              # (n_boxes,)  - struct→aero normalwash
        f = A_jj @ w              # (n_boxes,)  - aero forces
        return G_sp.T @ f         # (n_free,)   - aero→struct forces

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
    F_trim_fixed = G_sp.T @ (A_jj @ w_fixed)

    Q_ax = np.zeros((n_free, n_trim_free))
    for k, label in enumerate(free_labels):
        w_contrib = _trim_variable_normalwash(label, boxes, box_id_to_index,
                                              bdf_model)
        Q_ax[:, k] = G_sp.T @ (A_jj @ w_contrib)

    # 7. Build trim constraint equations
    total_weight = _compute_total_weight(bdf_model)
    cg_x = _compute_cg_x(bdf_model)

    # Determine number of constraints based on free trim variables
    # For full aircraft: Fz balance + My balance = 2 constraints
    # For single-surface: just Fz balance = 1 constraint
    n_constraints = min(n_trim_free, 2)  # Up to 2: force + moment
    if n_constraints < 1:
        n_constraints = 0

    # D_r[c, :] is a vector of size n_free, D_x[c, :] of size n_trim_free
    D_r = np.zeros((n_constraints, n_free))
    D_x = np.zeros((n_constraints, n_trim_free))
    rhs_trim = np.zeros(n_constraints)

    # Force summation vector (sum all box forces in z-direction)
    sum_force = np.ones(n_boxes)

    if n_constraints >= 1:
        # Z-force balance: sum aero Fz = weight
        # D_r[0,:] = sum_force @ A_jj @ G → (1,n_boxes) @ (n_boxes,n_boxes) @ (n_boxes,n_free)
        sum_A = sum_force @ A_jj  # (n_boxes,)
        D_r[0, :] = (G_sp.T @ sum_A).ravel()
        for k in range(n_trim_free):
            w_k = _trim_variable_normalwash(free_labels[k], boxes,
                                            box_id_to_index, bdf_model)
            D_x[0, k] = sum_A @ w_k
        F_z_fixed = sum_A @ w_fixed
        rhs_trim[0] = total_weight - F_z_fixed

    if n_constraints >= 2:
        # Pitch moment balance about CG: sum(F_z * (x_cp - x_cg)) = 0
        moment_arm = np.array([boxes[i].control_point[0] - cg_x
                               for i in range(n_boxes)])
        mom_A = moment_arm @ A_jj  # (n_boxes,)
        D_r[1, :] = (G_sp.T @ mom_A).ravel()
        for k in range(n_trim_free):
            w_k = _trim_variable_normalwash(free_labels[k], boxes,
                                            box_id_to_index, bdf_model)
            D_x[1, k] = mom_A @ w_k
        M_y_fixed = mom_A @ w_fixed
        rhs_trim[1] = -M_y_fixed  # Moment equilibrium: M_aero = 0

        logger.info("  Moment ref (CG_x) = %.4f", cg_x)

    # 8. Solve: use iterative approach for large models
    #
    # The coupled aeroelastic trim system:
    #   (K + Q_aa) u + (-Q_ax) x = F_f + F_trim_fixed
    #   D_r u + D_x x = rhs_trim
    #
    # Strategy: solve for x_trim from constraint equations, then for u iteratively.
    # For n_trim_free free variables, first compute condensed system.

    use_iterative = n_free > 10000
    logger.info("  Solver mode: %s (%d free DOFs, %d trim vars)",
                "iterative" if use_iterative else "dense", n_free, n_trim_free)

    if use_iterative:
        u_f, x_trim = _solve_iterative(K_ff, apply_Q_aa, Q_ax, F_f,
                                         F_trim_fixed, D_r, D_x, rhs_trim,
                                         n_free, n_trim_free, n_constraints,
                                         G_sp, A_jj, w_fixed, boxes, D_inv,
                                         free_labels, box_id_to_index, bdf_model)
    else:
        u_f, x_trim = _solve_dense(K_ff, G_sp, A_jj, Q_ax, F_f,
                                    F_trim_fixed, D_r, D_x, rhs_trim,
                                    n_free, n_trim_free, n_constraints)

    # 9. Post-process
    ndof = dof_mgr.total_dof
    u_full = np.zeros(ndof)
    for i, dof in enumerate(f_dofs):
        u_full[dof] = u_f[i]

    # Total normalwash
    w_total = G_sp @ u_f + w_fixed
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

    # 10. Compute nodal trim loads (aero + inertial + combined)
    try:
        from ..loads_analysis.trim_loads import (
            compute_trim_nodal_loads, verify_trim_balance)

        g = _detect_gravity(bdf_model)
        aero_nodal, inertial_nodal, combined_nodal = compute_trim_nodal_loads(
            bdf_model, boxes, aero_forces, G_sp, f_dofs, dof_mgr,
            nz=1.0, g=g)

        sc_result.nodal_aero_forces = aero_nodal
        sc_result.nodal_inertial_forces = inertial_nodal
        sc_result.nodal_combined_forces = combined_nodal

        # Verify trim balance
        cg_pt = np.array([cg_x, 0.0, 0.0])
        balance = verify_trim_balance(bdf_model, combined_nodal, ref_point=cg_pt)
        sc_result.trim_balance = balance
    except Exception as e:
        logger.warning("  Trim loads computation failed: %s", e)

    return sc_result


def _solve_dense(K_ff, G_sp, A_jj, Q_ax, F_f, F_trim_fixed,
                 D_r, D_x, rhs_trim, n_free, n_trim_free, n_constraints):
    """Solve trim using dense matrices (small models)."""
    G_dense = G_sp.toarray()
    Q_aa_free = G_dense.T @ A_jj @ G_dense

    n_total = n_free + n_trim_free
    K_dense = K_ff.toarray() if sp.issparse(K_ff) else K_ff

    if n_trim_free > 0:
        A_sys = np.zeros((n_total + n_constraints, n_total))
        rhs_sys = np.zeros(n_total + n_constraints)

        A_sys[:n_free, :n_free] = K_dense + Q_aa_free
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
        K_eff = K_dense + Q_aa_free
        rhs_eff = F_f + F_trim_fixed
        u_f = np.linalg.solve(K_eff, rhs_eff)
        x_trim = np.array([])

    return u_f, x_trim


def _solve_iterative(K_ff, apply_Q_aa, Q_ax, F_f, F_trim_fixed,
                     D_r, D_x, rhs_trim, n_free, n_trim_free, n_constraints,
                     G_sp, A_jj, w_fixed, boxes, D_inv,
                     free_labels, box_id_to_index, bdf_model):
    """Solve trim using sparse direct solver for large models.

    For SOL 144 static aeroelastic analysis, the aero contribution Q_aa
    is low-rank (n_boxes << n_free). We add it to K_ff as a sparse
    low-rank update using the Woodbury identity approach, or more
    practically, we add the aero stiffness as sparse COO entries.

    The system: (K + G^T A_jj G) u = F + Q_ax x
    with G sparse (n_boxes x n_free), A_jj dense (n_boxes x n_boxes).

    Strategy: form K_eff = K_ff + G^T @ A_jj @ G as sparse using
    the fact that G is very sparse. The product G^T @ A_jj @ G has
    at most (nnz_G)^2 / n_boxes nonzeros per row.
    """
    import time
    t_start = time.perf_counter()

    K_sparse = K_ff if sp.issparse(K_ff) else sp.csc_matrix(K_ff)

    # Build Q_aa as sparse: Q_aa = G^T @ A_jj @ G
    # Since G is sparse with ~28K nonzeros and n_boxes=783,
    # AG = A_jj @ G is (n_boxes x n_free) but computed column-by-column
    # via G's sparse structure.
    logger.info("  Building sparse Q_aa via G^T @ A_jj @ G ...")
    t_q = time.perf_counter()

    # A_jj @ G_sp.T gives (n_boxes x n_free) dense, but we only need
    # G_sp.T @ (A_jj @ G_sp) which uses G's sparsity
    # Compute AG = A_jj @ G_sp^T → (n_boxes x n_free), but only for
    # columns with nonzero entries in G
    # Better approach: G^T @ A_jj @ G = (G^T @ A_jj) @ G
    # GA = G_sp @ A_jj.T = (A_jj @ G_sp^T)^T  → transpose approach
    # Actually: G_sp^T has shape (n_free, n_boxes)
    # Q_aa = G_sp^T @ A_jj @ G_sp

    # For G_sp (n_boxes x n_free), A_jj (n_boxes x n_boxes):
    # Step 1: B = A_jj @ G_sp → (n_boxes x n_free) - G_sp is sparse
    # This is n_boxes x n_free but we can use sparse G to keep it manageable
    # Step 2: Q_aa = G_sp.T @ B → (n_free x n_free)
    # With G_sp sparse, the result is also sparse.

    # Convert G to CSC for efficient column operations
    G_csc = G_sp.tocsc()

    # B = A_jj @ G → dense * sparse = dense, but only nonzero columns matter
    # Get nonzero column indices of G
    col_nnz = np.diff(G_csc.indptr) > 0
    active_cols = np.where(col_nnz)[0]
    n_active = len(active_cols)
    logger.info("  G has %d active columns out of %d", n_active, n_free)

    # Compute B_active = A_jj @ G[:, active_cols] → (n_boxes x n_active)
    G_active = G_csc[:, active_cols].toarray()  # (n_boxes x n_active)
    B_active = A_jj @ G_active  # (n_boxes x n_active)

    # Q_active = G[:, active_cols]^T @ B_active → (n_active x n_active)
    Q_active = G_active.T @ B_active  # (n_active x n_active)
    logger.info("  Q_active size: %d x %d, computed in %.2f s",
                n_active, n_active, time.perf_counter() - t_q)

    # Build K_eff = K_ff + Q_aa (in reduced active space)
    # Add small regularization for near-singular K
    K_reg = K_sparse.copy()
    diag = np.abs(K_reg.diagonal())
    avg_diag = np.mean(diag[diag > 0]) if np.any(diag > 0) else 1.0
    eps_reg = avg_diag * 1e-8
    K_reg = K_reg + sp.eye(n_free, format='csc') * eps_reg
    logger.info("  Regularization: eps = %.2e (avg_diag = %.2e)", eps_reg, avg_diag)

    # Add Q_aa contribution at active DOFs (vectorized COO construction)
    if n_active > 0 and n_active < 5000:
        # Build COO indices via broadcasting (no Python loops)
        row_idx = np.repeat(active_cols, n_active)       # (n_active^2,)
        col_idx = np.tile(active_cols, n_active)          # (n_active^2,)
        q_vals = Q_active.ravel()                         # (n_active^2,)
        mask = np.abs(q_vals) > 1e-30
        if mask.any():
            Q_sp = sp.coo_matrix((q_vals[mask], (row_idx[mask], col_idx[mask])),
                                 shape=(n_free, n_free)).tocsc()
            K_eff = K_reg + Q_sp
        else:
            K_eff = K_reg
    else:
        K_eff = K_reg

    # Factorize K_eff using best available solver
    logger.info("  Factorizing K_eff (%d x %d)...", n_free, n_free)
    t_lu = time.perf_counter()

    K_eff_csc = K_eff.tocsc()

    # Strategy: try ILU-preconditioned CG first (2-3x faster for SPD systems),
    # fall back to direct LU if CG fails
    _solver_mode = None  # 'pardiso', 'ilu_cg', or 'splu'
    _pardiso_solve = None
    _ilu_pc = None
    K_lu = None

    # 1. Try pypardiso (MKL PARDISO) — multi-threaded direct solver
    try:
        from pypardiso import spsolve as pardiso_solve
        _solver_mode = 'pardiso'
        _pardiso_solve = pardiso_solve
        logger.info("  Using PyPardiso (MKL PARDISO) solver")
    except ImportError:
        pass

    # 2. Try ILU-preconditioned CG (much faster for large SPD systems)
    if _solver_mode is None:
        try:
            K_sym = ((K_eff_csc + K_eff_csc.T) * 0.5).tocsc()
            ilu = spla.spilu(K_sym, fill_factor=10)
            _ilu_pc = spla.LinearOperator(K_sym.shape, matvec=ilu.solve)
            _solver_mode = 'ilu_cg'
            logger.info("  Using ILU(10)-preconditioned CG solver")
        except Exception as e:
            logger.info("  ILU build failed (%s), falling back to direct LU", e)

    # 3. Fallback: SciPy SuperLU
    if _solver_mode is None:
        try:
            K_lu = spla.splu(K_eff_csc, permc_spec='COLAMD')
            _solver_mode = 'splu'
        except Exception as e:
            logger.error("  All solvers failed: %s", e)
            return np.zeros(n_free), np.zeros(n_trim_free)

    logger.info("  Factorization done in %.2f s", time.perf_counter() - t_lu)

    def _solve_system(rhs_vec):
        """Solve K_eff @ x = rhs using best available solver."""
        if _solver_mode == 'pardiso':
            return _pardiso_solve(K_eff_csc, rhs_vec)
        elif _solver_mode == 'ilu_cg':
            x_cg, info = spla.cg(K_sym, rhs_vec, M=_ilu_pc,
                                  rtol=1e-10, maxiter=500)
            if info != 0:
                logger.warning("  CG did not converge (info=%d), using direct solve", info)
                lu_fallback = spla.splu(K_eff_csc, permc_spec='COLAMD')
                return lu_fallback.solve(rhs_vec)
            return x_cg
        return K_lu.solve(rhs_vec)

    # Iterative trim solution
    x_trim = np.zeros(n_trim_free)
    max_iter = 20
    tol_trim = 1e-6

    for iteration in range(max_iter):
        rhs = F_f + F_trim_fixed
        if n_trim_free > 0:
            rhs = rhs + Q_ax @ x_trim

        u_f = _solve_system(rhs)

        if n_constraints == 0 or n_trim_free == 0:
            break

        # Update trim variables
        residual_trim = rhs_trim - D_r @ u_f
        if n_trim_free == n_constraints:
            x_new = np.linalg.solve(D_x, residual_trim)
        else:
            x_new, _, _, _ = np.linalg.lstsq(D_x, residual_trim, rcond=None)

        dx = np.linalg.norm(x_new - x_trim)
        x_trim = x_new

        logger.info("  Trim iter %d: dx = %.2e, x = %s",
                     iteration, dx, np.array2string(x_trim, precision=6))

        if dx < tol_trim:
            logger.info("  Trim converged in %d iterations", iteration + 1)
            break

    t_elapsed = time.perf_counter() - t_start
    logger.info("  Iterative solve done in %.2f s", t_elapsed)

    return u_f, x_trim


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


def _detect_gravity(bdf_model: BDFModel) -> float:
    """Detect gravitational acceleration from model unit system.

    Heuristic: if AEROS reference chord > 100, model is in mm → g = 9810 mm/s²
    Otherwise assume m → g = 9.81 m/s²
    """
    refc = 0.0
    if bdf_model.aeros:
        refc = bdf_model.aeros.refc
    elif bdf_model.aero:
        refc = bdf_model.aero.refc

    if refc > 100:
        # mm-N-sec system: g = 9810 mm/s²
        return 9810.0
    else:
        # m-N-sec system: g = 9.81 m/s²
        return 9.81


def _compute_total_weight(bdf_model: BDFModel) -> float:
    """Compute total structural weight from mass properties."""
    total_mass = 0.0

    for eid, elem in bdf_model.elements.items():
        if not hasattr(elem, 'property_ref') or elem.property_ref is None:
            continue
        prop = elem.property_ref

        if elem.type in ("CBAR", "CBEAM"):
            mat = getattr(prop, 'material_ref', None)
            if mat is None:
                continue
            n1 = bdf_model.nodes[elem.node_ids[0]]
            n2 = bdf_model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            total_mass += mat.rho * prop.A * L

        elif elem.type == "CROD":
            mat = getattr(prop, 'material_ref', None)
            if mat is None:
                continue
            n1 = bdf_model.nodes[elem.node_ids[0]]
            n2 = bdf_model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            total_mass += mat.rho * prop.A * L

        elif elem.type in ("CQUAD4", "CTRIA3"):
            # Get rho and t depending on property type
            if hasattr(prop, 'equivalent_isotropic'):
                # PCOMP: use equivalent properties
                E, nu, t, rho = prop.equivalent_isotropic()
            else:
                mat = getattr(prop, 'material_ref', None)
                if mat is None:
                    continue
                rho = mat.rho
                t = getattr(prop, 't', 0.0)

            if rho > 0 and t > 0:
                # Compute area
                coords = np.array([bdf_model.nodes[nid].xyz_global
                                   for nid in elem.node_ids])
                if elem.type == "CTRIA3":
                    v1 = coords[1] - coords[0]
                    v2 = coords[2] - coords[0]
                    area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                else:
                    # CQUAD4: sum of 2 triangles
                    v1 = coords[1] - coords[0]; v2 = coords[2] - coords[0]
                    v3 = coords[2] - coords[0]; v4 = coords[3] - coords[0]
                    area = 0.5 * (np.linalg.norm(np.cross(v1, v2)) +
                                  np.linalg.norm(np.cross(v3, v4)))
                total_mass += rho * t * area

    for mid, mass_elem in bdf_model.masses.items():
        total_mass += mass_elem.mass

    g = _detect_gravity(bdf_model)
    weight = total_mass * g
    logger.info("  Total mass = %.4f (consistent units), Weight = %.2f, g = %.1f",
                total_mass, weight, g)
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

        if elem.type in ("CBAR", "CBEAM", "CROD"):
            mat = getattr(prop, 'material_ref', None)
            if mat is None:
                continue
            n1 = bdf_model.nodes[elem.node_ids[0]]
            n2 = bdf_model.nodes[elem.node_ids[1]]
            L = np.linalg.norm(n2.xyz_global - n1.xyz_global)
            mass = mat.rho * prop.A * L
            x_mid = 0.5 * (n1.xyz_global[0] + n2.xyz_global[0])
            total_mass += mass
            moment_x += mass * x_mid

        elif elem.type in ("CQUAD4", "CTRIA3"):
            if hasattr(prop, 'equivalent_isotropic'):
                E, nu, t, rho = prop.equivalent_isotropic()
            else:
                mat = getattr(prop, 'material_ref', None)
                if mat is None:
                    continue
                rho = mat.rho
                t = getattr(prop, 't', 0.0)

            if rho > 0 and t > 0:
                coords = np.array([bdf_model.nodes[nid].xyz_global
                                   for nid in elem.node_ids])
                centroid_x = np.mean(coords[:, 0])
                if elem.type == "CTRIA3":
                    v1 = coords[1] - coords[0]; v2 = coords[2] - coords[0]
                    area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                else:
                    v1 = coords[1] - coords[0]; v2 = coords[2] - coords[0]
                    v3 = coords[2] - coords[0]; v4 = coords[3] - coords[0]
                    area = 0.5 * (np.linalg.norm(np.cross(v1, v2)) +
                                  np.linalg.norm(np.cross(v3, v4)))
                mass = rho * t * area
                total_mass += mass
                moment_x += mass * centroid_x

    for mid, mass_elem in bdf_model.masses.items():
        nid = mass_elem.node_id
        if nid in bdf_model.nodes:
            x = bdf_model.nodes[nid].xyz_global[0]
            total_mass += mass_elem.mass
            moment_x += mass_elem.mass * x

    if total_mass > 1e-12:
        return moment_x / total_mass
    return 0.0

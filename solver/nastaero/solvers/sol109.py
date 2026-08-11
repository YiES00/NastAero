"""SOL 109 - Direct Transient Response Analysis.

Solves the structural dynamics equation:
    M·ü + C·u̇ + K·u = F(t)

using the Newmark-β implicit time integration method
(β=0.25, γ=0.5: unconditionally stable, no numerical damping).

Damping model: Rayleigh proportional damping
    C = α·M + β·K
where α and β are chosen to give target damping ratios at
two reference frequencies.

References
----------
- Newmark, N.M., "A Method of Computation for Structural Dynamics",
  ASCE J. Eng. Mech. Div., 1959.
- Bathe, K.J., "Finite Element Procedures", Ch. 9.
- MSC Nastran Quick Reference Guide, SOL 109.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..bdf.model import BDFModel
from ..output.result_data import ResultData, SubcaseResult
from ..config import logger


def rayleigh_damping(M: sp.spmatrix, K: sp.spmatrix,
                     zeta: float = 0.02,
                     f1: float = 0.5, f2: float = 5.0,
                     ) -> sp.spmatrix:
    """Compute Rayleigh damping matrix C = alpha*M + beta*K.

    Parameters
    ----------
    M, K : sparse matrices
    zeta : float
        Target damping ratio at both reference frequencies.
    f1, f2 : float
        Reference frequencies (Hz) for damping ratio targeting.

    Returns
    -------
    C : sparse matrix
        Proportional damping matrix.
    """
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    # Solve 2x2 system: zeta = alpha/(2*wi) + beta*wi/2
    det = w2**2 - w1**2
    if abs(det) < 1e-12:
        alpha = 0.0
        beta_coeff = 2 * zeta / (w1 + w2)
    else:
        alpha = 2 * zeta * w1 * w2 * (w2 - w1) / det
        beta_coeff = 2 * zeta * (w2 - w1) / det
    alpha = max(alpha, 0.0)
    beta_coeff = max(beta_coeff, 0.0)
    logger.info("  Rayleigh damping: alpha=%.4f, beta=%.6f (zeta=%.1f%% at %.1f-%.1f Hz)",
                alpha, beta_coeff, zeta * 100, f1, f2)
    return alpha * M + beta_coeff * K


def solve_direct_transient(
    bdf_model: BDFModel,
    force_func: Callable[[float, np.ndarray], np.ndarray],
    t_array: np.ndarray,
    zeta: float = 0.02,
    f1: float = 0.5,
    f2: float = 5.0,
    output_node_ids: Optional[List[int]] = None,
    output_interval: int = 1,
) -> Dict:
    """Run SOL 109 direct transient analysis.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model (will be cross-referenced).
    force_func : callable(t, dof_mgr) -> F_full(ndof,)
        Time-varying force vector function.
        Returns force vector in full DOF ordering.
    t_array : ndarray
        Time points for integration.
    zeta : float
        Structural damping ratio for Rayleigh damping.
    f1, f2 : float
        Reference frequencies (Hz) for Rayleigh damping.
    output_node_ids : list of int or None
        Nodes to save displacement history. None = all.
    output_interval : int
        Save every N-th time step.

    Returns
    -------
    dict with keys:
        t : ndarray (n_output,) — time points
        displacements : dict {node_id: ndarray(n_output, 6)}
        peak_disp : dict {node_id: ndarray(6)} — peak absolute
        reactions : ndarray(n_output, n_spc_dofs) — SPC reactions
    """
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)
    dof_mgr = fe_model.dof_mgr

    subcase = bdf_model.subcases[0] if bdf_model.subcases else bdf_model.global_case
    K_ff, M_ff, _, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)
    n_free = K_ff.shape[0]
    n_total = dof_mgr.total_dof

    logger.info("SOL 109: %d free DOFs, %d time steps, dt=%.4f s",
                n_free, len(t_array), t_array[1] - t_array[0])

    # Damping matrix
    C_ff = rayleigh_damping(M_ff, K_ff, zeta, f1, f2)

    # Map full DOFs to free DOFs
    full_to_free = np.full(n_total, -1, dtype=int)
    for i, dof in enumerate(f_dofs):
        full_to_free[dof] = i

    # Newmark-β parameters (average acceleration: unconditionally stable)
    beta = 0.25
    gamma = 0.5
    dt = t_array[1] - t_array[0]

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt / 2.0 * (gamma / beta - 2.0)

    # Effective stiffness (constant, factored once)
    K_eff = K_ff + a0 * M_ff + a1 * C_ff
    K_eff_csc = sp.csc_matrix(K_eff)
    lu = spla.splu(K_eff_csc)
    logger.info("  K_eff factored (LU): %d nnz", K_eff_csc.nnz)

    # Initial conditions
    u = np.zeros(n_free)
    v = np.zeros(n_free)
    a = np.zeros(n_free)

    # Initial acceleration from F(0) = M*a + C*v + K*u → a = M^{-1}*(F - K*u - C*v)
    F0_full = force_func(t_array[0], dof_mgr)
    F0_f = F0_full[f_dofs]
    # For zero initial conditions, a0 = M^{-1} * F0
    # Skip if F0 is zero (hover equilibrium)

    # Output storage
    output_times = t_array[::output_interval]
    n_out = len(output_times)

    # Determine output nodes
    if output_node_ids is None:
        output_node_ids = dof_mgr.node_ids

    disp_history = {nid: np.zeros((n_out, 6)) for nid in output_node_ids}
    peak_disp = {nid: np.zeros(6) for nid in output_node_ids}
    out_idx = 0

    # Time integration loop
    for step in range(len(t_array)):
        t = t_array[step]

        if step > 0:
            # Get force at current time
            F_full = force_func(t, dof_mgr)
            F_f = F_full[f_dofs]

            # Effective force
            F_eff = (F_f
                     + M_ff @ (a0 * u + a2 * v + a3 * a)
                     + C_ff @ (a1 * u + a4 * v + a5 * a))

            # Solve for new displacement
            u_new = lu.solve(F_eff)

            # Update acceleration and velocity
            a_new = a0 * (u_new - u) - a2 * v - a3 * a
            v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)

            u = u_new
            v = v_new
            a = a_new

        # Store output
        if step % output_interval == 0 and out_idx < n_out:
            # Expand to full DOF
            u_full = np.zeros(n_total)
            u_full[f_dofs] = u

            for nid in output_node_ids:
                nd_dofs = dof_mgr.get_node_dofs(nid)
                d = u_full[nd_dofs[0]:nd_dofs[5]+1]
                disp_history[nid][out_idx] = d
                peak_disp[nid] = np.maximum(peak_disp[nid], np.abs(d))

            out_idx += 1

        if step % 200 == 0 and step > 0:
            max_u = np.max(np.abs(u))
            logger.info("  Step %d/%d (t=%.3fs): max|u|=%.4e",
                        step, len(t_array), t, max_u)

    logger.info("  SOL 109 complete: %d output steps saved", out_idx)

    return {
        "t": output_times[:out_idx],
        "displacements": disp_history,
        "peak_disp": peak_disp,
        "dof_mgr": dof_mgr,
        "f_dofs": f_dofs,
    }

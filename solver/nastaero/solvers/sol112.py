"""SOL 112 - Modal Transient Response Analysis.

Projects the structural dynamics equation onto modal space:
    M·ü + C·u̇ + K·u = F(t)

becomes N decoupled SDOF equations (one per mode):
    q̈_i + 2·ζ_i·ω_i·q̇_i + ω_i²·q_i = f_i(t) / m_i

where:
    u(t) = Φ·q(t)           (modal superposition)
    m_i = Φ_i^T·M·Φ_i       (generalized mass)
    f_i(t) = Φ_i^T·F(t)     (generalized force)

Advantages over SOL 109:
- 30 modal DOFs vs 4000 structural DOFs → ~100x faster
- Each modal equation is independent → trivially parallelizable
- Modal damping (per-mode ζ) is more physical than Rayleigh

References
----------
- Bathe, K.J., "Finite Element Procedures", Ch. 9.4
- MSC Nastran Quick Reference Guide, SOL 112
- Craig, R.R., "Structural Dynamics", Ch. 12
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..bdf.model import BDFModel
from ..output.result_data import ResultData, SubcaseResult
from ..config import logger


def solve_modal_transient(
    bdf_model: BDFModel,
    force_func: Callable[[float, 'DOFManager'], np.ndarray],
    t_array: np.ndarray,
    n_modes: int = 30,
    zeta: float = 0.02,
    output_node_ids: Optional[List[int]] = None,
    output_interval: int = 1,
) -> Dict:
    """Run SOL 112 modal transient analysis.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model.
    force_func : callable(t, dof_mgr) -> F_full(ndof,)
        Time-varying force vector in full DOF space.
    t_array : ndarray
        Time points for integration.
    n_modes : int
        Number of modes to include (modal truncation).
    zeta : float
        Modal damping ratio (constant for all modes).
    output_node_ids : list of int or None
        Nodes to save displacement history.
    output_interval : int
        Save every N-th time step.

    Returns
    -------
    dict with keys:
        t : ndarray — output time points
        displacements : dict {node_id: ndarray(n_out, 6)}
        peak_disp : dict {node_id: ndarray(6)}
        modal_coords : ndarray(n_out, n_modes) — modal coordinate history
        frequencies : ndarray(n_modes,) — natural frequencies
        modal_forces : ndarray(n_out, n_modes) — generalized forces
    """
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)
    dof_mgr = fe_model.dof_mgr

    subcase = bdf_model.subcases[0] if bdf_model.subcases else bdf_model.global_case
    K_ff, M_ff, _, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)
    n_free = K_ff.shape[0]
    n_total = dof_mgr.total_dof

    # ── Step 1: Modal extraction (SOL 103) ──
    if n_modes >= n_free:
        n_modes = max(n_free - 2, 1)
    logger.info("SOL 112: Extracting %d modes from %d free DOFs ...", n_modes, n_free)

    eigenvalues, eigenvectors = spla.eigsh(
        K_ff, k=n_modes, M=M_ff, sigma=0.0, which='LM')

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    omega2 = eigenvalues
    omega = np.sqrt(np.maximum(omega2, 0.0))
    frequencies = omega / (2 * np.pi)

    logger.info("  Modes: %.3f to %.3f Hz (%d modes)", frequencies[0],
                frequencies[-1], n_modes)

    # ── Step 2: Generalized mass normalization ──
    # Φ^T M Φ = diag(m_i)
    Phi = eigenvectors  # (n_free, n_modes)
    gen_mass = np.array([Phi[:, i] @ M_ff @ Phi[:, i] for i in range(n_modes)])

    # Normalize to unit generalized mass
    for i in range(n_modes):
        if gen_mass[i] > 0:
            Phi[:, i] /= np.sqrt(gen_mass[i])
    gen_mass[:] = 1.0  # After normalization

    logger.info("  Modal damping: zeta = %.1f%% for all modes", zeta * 100)

    # ── Step 3: Time integration (Newmark-β per modal DOF) ──
    dt = t_array[1] - t_array[0]
    n_steps = len(t_array)
    beta_nm = 0.25
    gamma_nm = 0.5

    # Modal state: q, q_dot, q_ddot for each mode
    q = np.zeros(n_modes)
    q_dot = np.zeros(n_modes)
    q_ddot = np.zeros(n_modes)

    # Precompute per-mode constants
    # For each mode: m*q_ddot + c*q_dot + k*q = f(t)
    # with m=1 (unit gen mass), c=2*zeta*omega, k=omega^2
    c_modal = 2 * zeta * omega  # modal damping coefficients
    k_modal = omega2  # modal stiffness

    # Newmark constants for each mode (with m=1)
    a0_nm = 1.0 / (beta_nm * dt**2)
    a1_nm = gamma_nm / (beta_nm * dt)
    a2_nm = 1.0 / (beta_nm * dt)
    a3_nm = 1.0 / (2.0 * beta_nm) - 1.0
    a4_nm = gamma_nm / beta_nm - 1.0
    a5_nm = dt / 2.0 * (gamma_nm / beta_nm - 2.0)

    # Effective modal stiffness: k_eff_i = k_i + a0*1 + a1*c_i
    k_eff = k_modal + a0_nm + a1_nm * c_modal  # (n_modes,)

    # Output storage
    output_times = t_array[::output_interval]
    n_out = len(output_times)

    if output_node_ids is None:
        output_node_ids = dof_mgr.node_ids

    disp_history = {nid: np.zeros((n_out, 6)) for nid in output_node_ids}
    peak_disp = {nid: np.zeros(6) for nid in output_node_ids}
    modal_coords = np.zeros((n_out, n_modes))
    modal_forces = np.zeros((n_out, n_modes))

    # Precompute Phi columns for output nodes (for fast recovery)
    node_phi = {}
    for nid in output_node_ids:
        nd_dofs = dof_mgr.get_node_dofs(nid)
        # Map node DOFs (full space) → free DOF indices
        phi_rows = []
        for d in range(6):
            full_dof = nd_dofs[d]
            free_idx = None
            for fi, fd in enumerate(f_dofs):
                if fd == full_dof:
                    free_idx = fi
                    break
            phi_rows.append(free_idx)
        node_phi[nid] = phi_rows

    # Build f_dofs lookup for fast indexing
    f_dofs_set = set(f_dofs)
    full_to_free = np.full(n_total, -1, dtype=int)
    for i, fd in enumerate(f_dofs):
        full_to_free[fd] = i

    out_idx = 0
    logger.info("  Integrating %d steps (dt=%.4f s) ...", n_steps, dt)

    for step in range(n_steps):
        t = t_array[step]

        # Get force and project onto modal space
        F_full = force_func(t, dof_mgr)
        F_f = F_full[f_dofs]
        f_modal = Phi.T @ F_f  # (n_modes,)

        if step > 0:
            # Newmark-β for each modal DOF
            for i in range(n_modes):
                if k_eff[i] < 1e-20:
                    continue
                f_eff = (f_modal[i]
                         + (a0_nm * q[i] + a2_nm * q_dot[i] + a3_nm * q_ddot[i])
                         + c_modal[i] * (a1_nm * q[i] + a4_nm * q_dot[i] + a5_nm * q_ddot[i]))
                q_new = f_eff / k_eff[i]
                q_ddot_new = a0_nm * (q_new - q[i]) - a2_nm * q_dot[i] - a3_nm * q_ddot[i]
                q_dot_new = q_dot[i] + dt * ((1.0 - gamma_nm) * q_ddot[i] + gamma_nm * q_ddot_new)
                q[i] = q_new
                q_dot[i] = q_dot_new
                q_ddot[i] = q_ddot_new

        # Store output
        if step % output_interval == 0 and out_idx < n_out:
            modal_coords[out_idx] = q.copy()
            modal_forces[out_idx] = f_modal.copy()

            # Recover physical displacements: u_free = Phi @ q
            u_free = Phi @ q  # (n_free,)

            for nid in output_node_ids:
                d = np.zeros(6)
                phi_rows = node_phi[nid]
                for comp in range(6):
                    fi = phi_rows[comp]
                    if fi is not None:
                        d[comp] = u_free[fi]
                disp_history[nid][out_idx] = d
                peak_disp[nid] = np.maximum(peak_disp[nid], np.abs(d))

            out_idx += 1

        if step % 200 == 0 and step > 0:
            max_q = np.max(np.abs(q))
            logger.info("  Step %d/%d (t=%.3fs): max|q|=%.4e",
                        step, n_steps, t, max_q)

    logger.info("  SOL 112 complete: %d output steps, %d modes", out_idx, n_modes)

    return {
        "t": output_times[:out_idx],
        "displacements": disp_history,
        "peak_disp": peak_disp,
        "modal_coords": modal_coords[:out_idx],
        "modal_forces": modal_forces[:out_idx],
        "frequencies": frequencies,
        "dof_mgr": dof_mgr,
        "f_dofs": f_dofs,
        "eigenvectors": Phi,
    }

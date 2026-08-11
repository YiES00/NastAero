"""SOL 146 - Dynamic Aeroelastic Response (Transient/Frequency).

Solves the coupled aeroelastic equation in modal coordinates:
    [-omega^2 M_hh + i*omega*B_hh + K_hh - q_dyn*Q_hh(m,k)] {u_h} = {P_h(omega)}

For Phase A (k=0 approximation):
- Computes Q_hh at k=0 using steady DLM (same as SOL 144)
- Valid for low-speed/hover where aero coupling is quasi-steady
- Falls back to SOL 112-style time-domain Newmark-beta integration
  when V < 10 m/s (negligible aero coupling)
- Uses frequency-domain FFT solve for V >= 10 m/s

The frequency-domain impedance matrix at each frequency omega_j is:

    Z(omega_j) = diag(-omega_j^2 + 2i*zeta*omega_i*omega_j + omega_i^2)
                 - q_dyn * Q_hh

where Q_hh is CONSTANT for Phase A (k=0), making each frequency
solve a cheap n_modes x n_modes linear system.

References
----------
- MSC Nastran Aeroelastic Analysis User's Guide, Ch. 2, Eq. 2-183
- Bisplinghoff, Ashley, Halfman, "Aeroelasticity", Ch. 6
- Bathe, K.J., "Finite Element Procedures", Ch. 9.4 (Newmark)
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..bdf.model import BDFModel
from ..config import logger


# Velocity threshold below which we use time-domain integration
# instead of frequency-domain solve (aero coupling negligible)
_V_THRESHOLD = 10.0  # m/s


def solve_aeroelastic_transient(
    bdf_model: BDFModel,
    force_func: Optional[Callable] = None,
    t_array: Optional[np.ndarray] = None,
    n_modes: int = 30,
    zeta: float = 0.02,
    use_aero_coupling: bool = True,
    output_node_ids: Optional[List[int]] = None,
    output_interval: int = 1,
) -> Dict:
    """Run SOL 146 dynamic aeroelastic transient analysis.

    Solves the coupled structural-aerodynamic transient response using
    modal superposition with optional aeroelastic coupling.

    Parameters
    ----------
    bdf_model : BDFModel
        Parsed BDF model with structural and (optionally) aerodynamic data.
    force_func : callable(t, dof_mgr) -> F_full(ndof,), optional
        Time-varying force vector in full DOF space.  If None and DLOAD
        cards exist, builds automatically from DLOAD/TLOAD1/DAREA/TABLED1.
    t_array : ndarray, optional
        Time points for integration.  If None, uses TSTEP card from model.
    n_modes : int
        Number of modes for modal truncation.
    zeta : float
        Default modal damping ratio (used if no TABDMP1 present).
    use_aero_coupling : bool
        If True and CAERO1 panels exist, include aerodynamic coupling.
    output_node_ids : list of int or None
        Nodes to save displacement history.  None = all nodes.
    output_interval : int
        Save every N-th time step.

    Returns
    -------
    dict with keys:
        t : ndarray -- output time points
        displacements : dict {node_id: ndarray(n_out, 6)}
        peak_disp : dict {node_id: ndarray(6)}
        modal_coords : ndarray(n_out, n_modes)
        modal_forces : ndarray(n_out, n_modes)
        frequencies : ndarray(n_modes,) -- natural frequencies (Hz)
        dof_mgr : DOFManager
        f_dofs : list
        eigenvectors : ndarray(n_free, n_modes)
        aero_coupling_active : bool
    """
    # ── Step 0: Cross-reference and build FE model ──
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)
    dof_mgr = fe_model.dof_mgr

    subcase = bdf_model.subcases[0] if bdf_model.subcases else bdf_model.global_case
    K_ff, M_ff, _, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)
    n_free = K_ff.shape[0]
    n_total = dof_mgr.total_dof

    # ── Build time array from TSTEP if not provided ──
    if t_array is None:
        t_array = _build_t_array_from_model(bdf_model, subcase)

    # ── Build force function from BDF cards if not provided ──
    if force_func is None:
        force_func = _build_force_func_from_model(bdf_model, subcase, dof_mgr,
                                                   n_total)

    # ── Step 1: Modal extraction ──
    if n_modes >= n_free:
        n_modes = max(n_free - 2, 1)
    logger.info("SOL 146: Extracting %d modes from %d free DOFs ...",
                n_modes, n_free)

    eigenvalues, eigenvectors = spla.eigsh(
        K_ff, k=n_modes, M=M_ff, sigma=0.0, which='LM')

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    omega2 = eigenvalues
    omega = np.sqrt(np.maximum(omega2, 0.0))
    frequencies = omega / (2.0 * np.pi)

    logger.info("  Modes: %.3f to %.3f Hz (%d modes)",
                frequencies[0], frequencies[-1], n_modes)

    # ── Step 2: Generalized mass normalization ──
    Phi = eigenvectors  # (n_free, n_modes)
    gen_mass = np.array([Phi[:, i] @ M_ff @ Phi[:, i]
                         for i in range(n_modes)])
    for i in range(n_modes):
        if gen_mass[i] > 0:
            Phi[:, i] /= np.sqrt(gen_mass[i])
    gen_mass[:] = 1.0

    # ── Step 3: Aero coupling (Phase A: k=0) ──
    has_aero_panels = bool(bdf_model.caero_panels)
    use_aero = use_aero_coupling and has_aero_panels

    Q_hh = None  # modal aero influence matrix
    q_dyn = 0.0
    V = 0.0

    if use_aero:
        Q_hh, q_dyn, V = _build_aero_coupling(bdf_model, Phi, f_dofs,
                                                dof_mgr, n_modes)
        if Q_hh is not None:
            logger.info("  Aero coupling: V=%.1f m/s, q_dyn=%.1f, "
                        "Q_hh max=%.4e",
                        V, q_dyn, np.max(np.abs(Q_hh)))
        else:
            use_aero = False

    # ── Get per-mode damping from TABDMP1 if available ──
    zeta_vec = _get_modal_damping(bdf_model, subcase, frequencies, zeta)
    logger.info("  Modal damping: mean zeta = %.1f%%",
                np.mean(zeta_vec) * 100)

    # ── Step 4: Choose solver path ──
    aero_coupling_active = use_aero and Q_hh is not None and q_dyn > 0.0

    if aero_coupling_active and V >= _V_THRESHOLD:
        # Frequency-domain solve (FFT)
        logger.info("  Using frequency-domain solver (V=%.1f >= %.1f m/s)",
                    V, _V_THRESHOLD)
        result = _solve_frequency_domain(
            t_array, Phi, omega, omega2, zeta_vec, Q_hh, q_dyn,
            force_func, dof_mgr, f_dofs, n_total, n_modes,
            output_node_ids, output_interval)
    else:
        # Time-domain Newmark-beta (SOL 112 path with optional aero stiffness)
        if aero_coupling_active:
            logger.info("  Using time-domain solver with aero stiffness "
                        "(V=%.1f < %.1f m/s)", V, _V_THRESHOLD)
        else:
            logger.info("  Using time-domain solver (no aero coupling)")
        result = _solve_time_domain(
            t_array, Phi, omega, omega2, zeta_vec, Q_hh, q_dyn,
            force_func, dof_mgr, f_dofs, n_total, n_modes,
            output_node_ids, output_interval,
            use_aero=aero_coupling_active)

    # Add common metadata
    result["frequencies"] = frequencies
    result["dof_mgr"] = dof_mgr
    result["f_dofs"] = f_dofs
    result["eigenvectors"] = Phi
    result["aero_coupling_active"] = aero_coupling_active

    return result


# ---------------------------------------------------------------------------
# Aero coupling construction
# ---------------------------------------------------------------------------

def _build_aero_coupling(
    bdf_model: BDFModel,
    Phi: np.ndarray,
    f_dofs: list,
    dof_mgr,
    n_modes: int,
) -> tuple:
    """Build modal aero influence matrix Q_hh at k=0.

    Uses the same spline/panel/AIC machinery as SOL 144.

    Returns
    -------
    Q_hh : ndarray (n_modes, n_modes) or None
        Modal aero influence matrix.
    q_dyn : float
        Dynamic pressure = 0.5 * rho * V^2.
    V : float
        Freestream velocity.
    """
    from .sol144 import _build_geff_per_spline
    from ..aero.panel import generate_all_panels, get_box_index_map
    from ..aero.dlm import build_aic_matrix

    # Get flight condition from AERO card
    aero = bdf_model.aero
    if aero is None:
        logger.warning("  No AERO card found, skipping aero coupling")
        return None, 0.0, 0.0

    V = aero.velocity
    rho = aero.rhoref
    q_dyn = 0.5 * rho * V ** 2
    mach = 0.0  # Phase A: approximate as M=0 for subsonic

    if V < 1e-6:
        logger.info("  V ~ 0, skipping aero coupling")
        return None, 0.0, 0.0

    # Generate aero panels
    boxes = generate_all_panels(bdf_model, use_nastran_eid=True)
    n_boxes = len(boxes)
    if n_boxes == 0:
        return None, 0.0, 0.0

    box_id_to_index = get_box_index_map(boxes)

    # Build spline coupling matrices
    G_sp_dense, G_disp_dense = _build_geff_per_spline(
        bdf_model, boxes, box_id_to_index, dof_mgr, f_dofs)

    # Build AIC at k=0 (steady, real-valued)
    D = build_aic_matrix(boxes, mach, reduced_freq=0.0)
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        D_inv = np.linalg.inv(D + np.eye(n_boxes) * 1e-10)

    # Build force diagonal: f_diag[j] = 2 * q * area_j / chord_j
    f_diag_vec = np.zeros(n_boxes)
    for j in range(n_boxes):
        chord_j = boxes[j].chord
        if chord_j > 1e-12:
            f_diag_vec[j] = 2.0 * q_dyn * boxes[j].area / chord_j

    # A_jj = diag(f_diag) @ D_inv
    A_jj = np.diag(f_diag_vec) @ D_inv

    # Q_aa = G_disp^T @ A_jj @ G_sp  (physical DOF space, asymmetric)
    Q_aa = G_disp_dense.T @ (A_jj @ G_sp_dense)

    # Transform to modal space: Q_hh = Phi^T @ Q_aa @ Phi
    Q_hh = Phi.T @ Q_aa @ Phi

    logger.info("  Aero panels: %d boxes, V=%.1f m/s, rho=%.3f, "
                "q=%.1f", n_boxes, V, rho, q_dyn)

    return Q_hh, q_dyn, V


# ---------------------------------------------------------------------------
# Time-domain solver (Newmark-beta, SOL 112 path)
# ---------------------------------------------------------------------------

def _solve_time_domain(
    t_array: np.ndarray,
    Phi: np.ndarray,
    omega: np.ndarray,
    omega2: np.ndarray,
    zeta_vec: np.ndarray,
    Q_hh: Optional[np.ndarray],
    q_dyn: float,
    force_func: Callable,
    dof_mgr,
    f_dofs: list,
    n_total: int,
    n_modes: int,
    output_node_ids: Optional[List[int]],
    output_interval: int,
    use_aero: bool = False,
) -> Dict:
    """Newmark-beta time integration in modal space.

    When aero coupling is active, modifies modal stiffness:
        k_eff_i = omega_i^2 - q_dyn * Q_hh[i,i]
    This is equivalent to SOL 112 with aeroelastic stiffness correction.
    """
    dt = t_array[1] - t_array[0]
    n_steps = len(t_array)
    beta_nm = 0.25
    gamma_nm = 0.5

    # Modal state
    q = np.zeros(n_modes)
    q_dot = np.zeros(n_modes)
    q_ddot = np.zeros(n_modes)

    # Modal damping and stiffness
    c_modal = 2.0 * zeta_vec * omega
    k_modal = omega2.copy()

    # Add aero stiffness to diagonal (Phase A quasi-steady approximation)
    if use_aero and Q_hh is not None:
        # For time-domain, only use diagonal of Q_hh to preserve
        # decoupled modal equations (off-diagonal coupling is small for k=0)
        for i in range(n_modes):
            k_modal[i] -= q_dyn * Q_hh[i, i]
        logger.info("  Aero stiffness shift: max delta_k = %.4e",
                    np.max(np.abs(q_dyn * np.diag(Q_hh)[:n_modes])))

    # Newmark constants
    a0_nm = 1.0 / (beta_nm * dt ** 2)
    a1_nm = gamma_nm / (beta_nm * dt)
    a2_nm = 1.0 / (beta_nm * dt)
    a3_nm = 1.0 / (2.0 * beta_nm) - 1.0
    a4_nm = gamma_nm / beta_nm - 1.0
    a5_nm = dt / 2.0 * (gamma_nm / beta_nm - 2.0)

    # Effective modal stiffness
    k_eff = k_modal + a0_nm + a1_nm * c_modal

    # Output storage
    output_times = t_array[::output_interval]
    n_out = len(output_times)

    if output_node_ids is None:
        output_node_ids = dof_mgr.node_ids

    disp_history = {nid: np.zeros((n_out, 6)) for nid in output_node_ids}
    peak_disp = {nid: np.zeros(6) for nid in output_node_ids}
    modal_coords = np.zeros((n_out, n_modes))
    modal_forces = np.zeros((n_out, n_modes))

    # Precompute Phi rows for output nodes
    node_phi = _precompute_node_phi(output_node_ids, dof_mgr, f_dofs)

    out_idx = 0
    logger.info("  Integrating %d steps (dt=%.4f s) ...", n_steps, dt)

    for step in range(n_steps):
        t = t_array[step]

        # Get force and project onto modal space
        F_full = force_func(t, dof_mgr)
        F_f = F_full[f_dofs]
        f_modal = Phi.T @ F_f

        if step > 0:
            # Newmark-beta per modal DOF
            for i in range(n_modes):
                if abs(k_eff[i]) < 1e-20:
                    continue
                f_eff = (f_modal[i]
                         + (a0_nm * q[i] + a2_nm * q_dot[i]
                            + a3_nm * q_ddot[i])
                         + c_modal[i] * (a1_nm * q[i] + a4_nm * q_dot[i]
                                         + a5_nm * q_ddot[i]))
                q_new = f_eff / k_eff[i]
                q_ddot_new = (a0_nm * (q_new - q[i])
                              - a2_nm * q_dot[i] - a3_nm * q_ddot[i])
                q_dot_new = (q_dot[i]
                             + dt * ((1.0 - gamma_nm) * q_ddot[i]
                                     + gamma_nm * q_ddot_new))
                q[i] = q_new
                q_dot[i] = q_dot_new
                q_ddot[i] = q_ddot_new

        # Store output
        if step % output_interval == 0 and out_idx < n_out:
            modal_coords[out_idx] = q.copy()
            modal_forces[out_idx] = f_modal.copy()

            u_free = Phi @ q
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

    logger.info("  SOL 146 (time-domain) complete: %d output steps, "
                "%d modes", out_idx, n_modes)

    return {
        "t": output_times[:out_idx],
        "displacements": disp_history,
        "peak_disp": peak_disp,
        "modal_coords": modal_coords[:out_idx],
        "modal_forces": modal_forces[:out_idx],
    }


# ---------------------------------------------------------------------------
# Frequency-domain solver (FFT)
# ---------------------------------------------------------------------------

def _solve_frequency_domain(
    t_array: np.ndarray,
    Phi: np.ndarray,
    omega: np.ndarray,
    omega2: np.ndarray,
    zeta_vec: np.ndarray,
    Q_hh: np.ndarray,
    q_dyn: float,
    force_func: Callable,
    dof_mgr,
    f_dofs: list,
    n_total: int,
    n_modes: int,
    output_node_ids: Optional[List[int]],
    output_interval: int,
) -> Dict:
    """Frequency-domain solve using FFT/IFFT.

    For each FFT frequency omega_j, solves:
        Z(omega_j) @ U_modal(:, j) = P_modal(:, j)

    where the impedance matrix is:
        Z = diag(-omega_j^2 + 2i*zeta*omega_i*omega_j + omega_i^2)
            - q_dyn * Q_hh

    Phase A: Q_hh is constant (k=0), so each Z is n_modes x n_modes.
    """
    N_fft = len(t_array)
    dt = t_array[1] - t_array[0]

    logger.info("  Building time-domain force matrix (%d modes x %d steps)...",
                n_modes, N_fft)

    # Build modal force time history
    F_modal = np.zeros((n_modes, N_fft))
    for step in range(N_fft):
        F_full = force_func(t_array[step], dof_mgr)
        F_f = F_full[f_dofs]
        F_modal[:, step] = Phi.T @ F_f

    # FFT to frequency domain
    logger.info("  FFT (%d points)...", N_fft)
    P_modal = np.fft.fft(F_modal, axis=1)  # (n_modes, N_fft)
    omega_fft = 2.0 * np.pi * np.fft.fftfreq(N_fft, d=dt)

    # Frequency-domain solve
    logger.info("  Solving impedance system at %d frequencies...", N_fft)
    U_modal = np.zeros_like(P_modal, dtype=complex)

    for j in range(N_fft):
        w = omega_fft[j]

        # Build impedance matrix Z(omega_j)
        # Diagonal: structural dynamics
        z_diag = -w ** 2 + 1j * w * 2.0 * zeta_vec * omega + omega2
        Z = np.diag(z_diag)

        # Subtract aero coupling (constant for Phase A)
        if q_dyn > 0:
            Z -= q_dyn * Q_hh

        # Solve Z @ u = p
        try:
            U_modal[:, j] = np.linalg.solve(Z, P_modal[:, j])
        except np.linalg.LinAlgError:
            # Regularize if singular (e.g. at DC with free-free modes)
            Z_reg = Z + np.eye(n_modes) * 1e-10
            U_modal[:, j] = np.linalg.solve(Z_reg, P_modal[:, j])

    # IFFT back to time domain
    logger.info("  IFFT to time domain...")
    q_history = np.fft.ifft(U_modal, axis=1).real  # (n_modes, N_fft)

    # Recover physical displacements
    output_times = t_array[::output_interval]
    n_out = len(output_times)

    if output_node_ids is None:
        output_node_ids = dof_mgr.node_ids

    disp_history = {nid: np.zeros((n_out, 6)) for nid in output_node_ids}
    peak_disp = {nid: np.zeros(6) for nid in output_node_ids}
    modal_coords = np.zeros((n_out, n_modes))
    modal_forces = np.zeros((n_out, n_modes))

    node_phi = _precompute_node_phi(output_node_ids, dof_mgr, f_dofs)

    out_idx = 0
    for step in range(N_fft):
        if step % output_interval != 0:
            continue
        if out_idx >= n_out:
            break

        q_step = q_history[:, step]
        modal_coords[out_idx] = q_step
        modal_forces[out_idx] = F_modal[:, step]

        u_free = Phi @ q_step
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

    logger.info("  SOL 146 (freq-domain) complete: %d output steps, "
                "%d modes", out_idx, n_modes)

    return {
        "t": output_times[:out_idx],
        "displacements": disp_history,
        "peak_disp": peak_disp,
        "modal_coords": modal_coords[:out_idx],
        "modal_forces": modal_forces[:out_idx],
    }


# ---------------------------------------------------------------------------
# Helper: precompute Phi row indices for output nodes
# ---------------------------------------------------------------------------

def _precompute_node_phi(output_node_ids: List[int], dof_mgr,
                         f_dofs: list) -> Dict:
    """Map output node DOFs to free DOF indices for fast recovery."""
    f_dof_set = {}
    for i, fd in enumerate(f_dofs):
        f_dof_set[fd] = i

    node_phi = {}
    for nid in output_node_ids:
        nd_dofs = dof_mgr.get_node_dofs(nid)
        phi_rows = []
        for d in range(6):
            full_dof = nd_dofs[d]
            phi_rows.append(f_dof_set.get(full_dof, None))
        node_phi[nid] = phi_rows
    return node_phi


# ---------------------------------------------------------------------------
# Helper: modal damping from TABDMP1
# ---------------------------------------------------------------------------

def _get_modal_damping(bdf_model: BDFModel, subcase, frequencies: np.ndarray,
                       default_zeta: float) -> np.ndarray:
    """Get per-mode damping ratios from TABDMP1 or use default."""
    n_modes = len(frequencies)
    zeta_vec = np.full(n_modes, default_zeta)

    # Check for SDAMPING case control → TABDMP1
    sdamp_id = getattr(subcase, 'sdamp_id', 0)
    if sdamp_id == 0:
        sdamp_id = getattr(bdf_model.global_case, 'sdamp_id', 0)

    if sdamp_id > 0 and sdamp_id in bdf_model.tabdmp1s:
        tabdmp = bdf_model.tabdmp1s[sdamp_id]
        for i in range(n_modes):
            g_val = tabdmp.get_damping(frequencies[i])
            if tabdmp.damp_type == "G":
                # Structural damping g → equivalent viscous zeta = g/2
                zeta_vec[i] = g_val / 2.0
            elif tabdmp.damp_type == "CRIT":
                zeta_vec[i] = g_val
            elif tabdmp.damp_type == "Q":
                # Q-factor → zeta = 1/(2Q)
                if g_val > 0:
                    zeta_vec[i] = 1.0 / (2.0 * g_val)
        logger.info("  TABDMP1 %d (%s): damping range %.4f - %.4f",
                    sdamp_id, tabdmp.damp_type,
                    np.min(zeta_vec), np.max(zeta_vec))

    return zeta_vec


# ---------------------------------------------------------------------------
# Helper: build time array from TSTEP card
# ---------------------------------------------------------------------------

def _build_t_array_from_model(bdf_model: BDFModel, subcase) -> np.ndarray:
    """Build time array from TSTEP card if available, else default."""
    tstep_id = getattr(subcase, 'tstep_id', 0)
    if tstep_id == 0:
        tstep_id = getattr(bdf_model.global_case, 'tstep_id', 0)

    if tstep_id > 0 and tstep_id in bdf_model.tsteps:
        tstep = bdf_model.tsteps[tstep_id]
        n_steps = tstep.n_steps
        dt = tstep.dt
        if n_steps > 0 and dt > 0:
            logger.info("  TSTEP %d: %d steps, dt=%.4f s", tstep_id,
                        n_steps, dt)
            return np.arange(n_steps + 1) * dt

    # Default: 1 second, 1000 steps
    logger.info("  No TSTEP card, using default: 1000 steps, dt=0.001 s")
    return np.linspace(0.0, 1.0, 1001)


# ---------------------------------------------------------------------------
# Helper: build force function from DLOAD/TLOAD1/DAREA/TABLED1
# ---------------------------------------------------------------------------

def _build_force_func_from_model(bdf_model: BDFModel, subcase,
                                  dof_mgr, n_total: int) -> Callable:
    """Build force_func from BDF dynamic load cards.

    Resolves the chain: DLOAD → TLOAD1 → DAREA + TABLED1

    Returns a callable(t, dof_mgr) → F(n_total,).
    """
    dload_id = getattr(subcase, 'dload_id', 0)
    if dload_id == 0:
        dload_id = getattr(bdf_model.global_case, 'dload_id', 0)

    # Collect all (scale, tload) pairs from DLOAD → TLOAD1
    load_entries = []  # [(overall_scale, tload, darea_list, tabled1)]

    if dload_id > 0 and dload_id in bdf_model.dloads:
        dload = bdf_model.dloads[dload_id]
        for sf, lid in zip(dload.scale_factors, dload.load_ids):
            overall_scale = dload.scale * sf
            if lid in bdf_model.tloads:
                tload = bdf_model.tloads[lid]
                darea_list = bdf_model.dareas.get(tload.exciteid, [])
                tabled1 = bdf_model.tabled1s.get(tload.tid, None)
                load_entries.append((overall_scale, tload, darea_list,
                                     tabled1))
    elif dload_id > 0 and dload_id in bdf_model.tloads:
        # DLOAD directly references a TLOAD1
        tload = bdf_model.tloads[dload_id]
        darea_list = bdf_model.dareas.get(tload.exciteid, [])
        tabled1 = bdf_model.tabled1s.get(tload.tid, None)
        load_entries.append((1.0, tload, darea_list, tabled1))

    if not load_entries:
        logger.info("  No dynamic load cards found, using zero force")
        def zero_force(t, dof_mgr_arg):
            return np.zeros(n_total)
        return zero_force

    logger.info("  Built force function from %d TLOAD1 entries",
                len(load_entries))

    def force_func(t, dof_mgr_arg):
        F = np.zeros(n_total)
        for overall_scale, tload, darea_list, tabled1 in load_entries:
            # Time value with delay
            t_eff = t - tload.delay
            if t_eff < 0:
                continue

            # Get amplitude from table
            amplitude = 1.0
            if tabled1 is not None:
                amplitude = tabled1.evaluate(t_eff)

            # Apply to DAREA points
            for darea in darea_list:
                for nid, comp, scale in darea.entries:
                    dof_idx = dof_mgr_arg.get_dof(nid, comp)
                    if 0 <= dof_idx < n_total:
                        F[dof_idx] += overall_scale * scale * amplitude
        return F

    return force_func

"""SOL 103 - Normal Modes (Real Eigenvalue) Analysis."""
from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..bdf.model import BDFModel, Subcase
from ..output.result_data import ResultData, SubcaseResult
from ..config import logger


def solve_modes(bdf_model: BDFModel) -> ResultData:
    """Run SOL 103 real eigenvalue analysis."""
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)
    results = ResultData(title="NastAero SOL 103 - Normal Modes Analysis")

    subcases = bdf_model.subcases if bdf_model.subcases else [bdf_model.global_case]
    for subcase in subcases:
        effective = bdf_model.get_effective_subcase(subcase)
        logger.info("Solving subcase %d (METHOD=%d) ...", subcase.id, effective.method_id)

        K_ff, M_ff, F_f, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)

        # Get EIGRL card for extraction parameters
        eigrl = bdf_model.eigrls.get(effective.method_id)
        if eigrl is None:
            logger.warning("METHOD %d not found, using default (10 modes)", effective.method_id)
            nd = 10
            v1, v2, norm = 0.0, 0.0, "MAX"
        else:
            nd = eigrl.nd if eigrl.nd > 0 else 10
            v1, v2, norm = eigrl.v1, eigrl.v2, eigrl.norm

        n_free = K_ff.shape[0]
        if nd >= n_free:
            nd = max(n_free - 2, 1)
        logger.info("  Extracting %d eigenvalues from %d free DOFs ...", nd, n_free)

        # Use shift-invert mode for better convergence on structural problems
        # sigma=0 finds modes near zero frequency (lowest modes)
        try:
            eigenvalues, eigenvectors = spla.eigsh(
                K_ff, k=nd, M=M_ff, sigma=0.0, which='LM'
            )
        except Exception as exc:
            logger.warning("Shift-invert failed (%s), trying standard mode", exc)
            eigenvalues, eigenvectors = spla.eigsh(
                K_ff, k=nd, M=M_ff, which='SM'
            )

        # Sort by eigenvalue (ascending)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Convert to natural frequencies
        # eigenvalue = omega^2, frequency = omega / (2*pi)
        omega2 = eigenvalues
        frequencies = np.sqrt(np.abs(omega2)) / (2.0 * np.pi)

        # Normalize eigenvectors
        if norm == "MASS":
            for j in range(nd):
                phi = eigenvectors[:, j]
                gen_mass = phi @ M_ff @ phi
                if gen_mass > 0:
                    eigenvectors[:, j] = phi / np.sqrt(gen_mass)
        else:  # MAX normalization (default)
            for j in range(nd):
                phi = eigenvectors[:, j]
                max_val = np.max(np.abs(phi))
                if max_val > 0:
                    eigenvectors[:, j] = phi / max_val

        # Store results
        sc_result = SubcaseResult(subcase_id=subcase.id)
        sc_result.eigenvalues = omega2
        sc_result.frequencies = frequencies
        sc_result.eigenvectors_full = []

        ndof = fe_model.dof_mgr.total_dof
        dof_mgr = fe_model.dof_mgr

        for j in range(nd):
            u_full = np.zeros(ndof)
            for i, dof in enumerate(f_dofs):
                u_full[dof] = eigenvectors[i, j]
            mode_disp = {}
            for nid in dof_mgr.node_ids:
                nd_dofs = dof_mgr.get_node_dofs(nid)
                mode_disp[nid] = u_full[nd_dofs[0]:nd_dofs[5]+1]
            sc_result.mode_shapes.append(mode_disp)
            sc_result.eigenvectors_full.append(u_full)

        results.subcases.append(sc_result)
        logger.info("  Extracted %d modes. Frequencies (Hz):", len(frequencies))
        for j, f in enumerate(frequencies):
            logger.info("    Mode %d: %.4f Hz (omega^2=%.6e)", j+1, f, omega2[j])

    return results

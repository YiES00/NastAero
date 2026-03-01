"""SOL 101 - Linear Static Analysis."""
from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as spla
from ..fem.model import FEModel
from ..bdf.model import BDFModel, Subcase
from ..output.result_data import ResultData, SubcaseResult
from ..config import logger


def solve_static(bdf_model: BDFModel) -> ResultData:
    """Run SOL 101 linear static analysis."""
    bdf_model.cross_reference()
    fe_model = FEModel(bdf_model)
    results = ResultData(title="NastAero SOL 101 - Linear Static Analysis")

    subcases = bdf_model.subcases if bdf_model.subcases else [bdf_model.global_case]
    for subcase in subcases:
        logger.info("Solving subcase %d ...", subcase.id)
        K_ff, M_ff, F_f, f_dofs, s_dofs = fe_model.get_partitioned_system(subcase)

        # Solve K_ff * u_f = F_f
        logger.info("  Solving %d x %d system ...", K_ff.shape[0], K_ff.shape[1])
        u_f = spla.spsolve(K_ff, F_f)

        # Reconstruct full displacement vector
        ndof = fe_model.dof_mgr.total_dof
        u_full = np.zeros(ndof)
        for i, dof in enumerate(f_dofs):
            u_full[dof] = u_f[i]

        # SPC reaction forces: F_reaction = K_sf * u_f - F_applied_s
        # where F_applied_s is the external load applied at constrained DOFs
        K = fe_model.K
        f_idx = np.array(f_dofs)
        s_idx = np.array(s_dofs)
        spc_forces_full = np.zeros(ndof)
        if len(s_dofs) > 0:
            # Get full load vector to extract loads at constrained DOFs
            F_full = fe_model.get_load_vector(subcase)
            F_spc = K[np.ix_(s_idx, f_idx)] @ u_f - F_full[s_idx]
            for i, dof in enumerate(s_dofs):
                spc_forces_full[dof] = F_spc[i]

        # Store results
        sc_result = SubcaseResult(subcase_id=subcase.id)
        dof_mgr = fe_model.dof_mgr
        for nid in dof_mgr.node_ids:
            nd = dof_mgr.get_node_dofs(nid)
            sc_result.displacements[nid] = u_full[nd[0]:nd[5]+1]
            sc_result.spc_forces[nid] = spc_forces_full[nd[0]:nd[5]+1]
        results.subcases.append(sc_result)
        logger.info("  Subcase %d complete. Max displacement = %.6e",
                     subcase.id, np.max(np.abs(u_full)))
    return results

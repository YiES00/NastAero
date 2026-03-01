"""Boundary condition application: SPC partitioning."""
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np
import scipy.sparse as sp
from ..config import logger

def apply_spcs(K, M, F, constrained_dofs, enforced_disps):
    ndof = K.shape[0]
    s_dofs = sorted(constrained_dofs); f_dofs = sorted(set(range(ndof)) - constrained_dofs)
    logger.info("SPC: %d constrained, %d free DOFs", len(s_dofs), len(f_dofs))
    f_idx = np.array(f_dofs); s_idx = np.array(s_dofs)
    K_ff = K[np.ix_(f_idx, f_idx)]; M_ff = M[np.ix_(f_idx, f_idx)]
    F_f = F[f_idx].copy()
    if enforced_disps:
        u_s = np.zeros(len(s_dofs))
        s_map = {d: i for i, d in enumerate(s_dofs)}
        for dof, val in enforced_disps.items():
            if dof in s_map: u_s[s_map[dof]] = val
        F_f -= K[np.ix_(f_idx, s_idx)] @ u_s
    return K_ff, M_ff, F_f, f_dofs, s_dofs

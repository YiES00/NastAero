"""FEModel: numerical finite element model built from BDFModel."""
from __future__ import annotations
from typing import Tuple, List
import numpy as np
import scipy.sparse as sp
from .dof_manager import DOFManager
from .assembly import assemble_global_matrices
from .load_vector import assemble_load_vector
from .boundary import apply_spcs
from ..bdf.model import BDFModel, Subcase
from ..config import logger

class FEModel:
    def __init__(self, bdf_model: BDFModel):
        self.bdf_model = bdf_model
        self.dof_mgr = DOFManager(list(bdf_model.nodes.keys()))
        logger.info("Assembling global stiffness and mass matrices...")
        self.K, self.M = assemble_global_matrices(bdf_model, self.dof_mgr)

    def get_partitioned_system(self, subcase: Subcase):
        effective = self.bdf_model.get_effective_subcase(subcase)
        F = assemble_load_vector(self.bdf_model, effective, self.dof_mgr)
        # Resolve SPCADD to get all SPC/SPC1 entries
        spc_list = self.bdf_model.resolve_spc_ids(effective.spc_id)
        constrained, enforced = self.dof_mgr.get_constrained_dofs(spc_list, self.bdf_model.nodes)
        return apply_spcs(self.K, self.M, F, constrained, enforced)

    def get_load_vector(self, subcase: Subcase) -> np.ndarray:
        """Get the full (unpartitioned) load vector for a subcase."""
        effective = self.bdf_model.get_effective_subcase(subcase)
        return assemble_load_vector(self.bdf_model, effective, self.dof_mgr)

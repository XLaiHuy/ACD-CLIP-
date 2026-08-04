"""Phase 4 H6 components.

Only Progress 1 is implemented in this branch.  The public modules are kept
small so later progress branches can add visual experts or consistency without
changing the Phase2B adapter contract.
"""

from .losses import (
    center_loss,
    concept_key_diversity_loss,
    dynamic_residual_diagnostics,
    dynamic_residual_diversity_loss,
    factor_stage_diagnostics,
    factor_aware_center_loss,
    factor_orthogonal_loss,
    prototype_diagnostics,
    router_teacher_loss,
    routing_balance_loss,
    teacher_candidate_diagnostics,
)
from .model import H6Progress1
from .router import PatchRouter
from .semantic_bank import CoPSSemanticCore

__all__ = [
    "CoPSSemanticCore",
    "H6Progress1",
    "PatchRouter",
    "center_loss",
    "concept_key_diversity_loss",
    "dynamic_residual_diagnostics",
    "dynamic_residual_diversity_loss",
    "factor_stage_diagnostics",
    "factor_aware_center_loss",
    "factor_orthogonal_loss",
    "prototype_diagnostics",
    "router_teacher_loss",
    "routing_balance_loss",
    "teacher_candidate_diagnostics",
]

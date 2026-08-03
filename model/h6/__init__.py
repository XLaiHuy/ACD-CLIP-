"""Phase 4 H6 components.

Only Progress 1 is implemented in this branch.  The public modules are kept
small so later progress branches can add visual experts or consistency without
changing the Phase2B adapter contract.
"""

from .losses import center_loss, factor_orthogonal_loss, routing_balance_loss
from .model import H6Progress1
from .router import PatchRouter
from .semantic_bank import CoPSSemanticCore

__all__ = [
    "CoPSSemanticCore",
    "H6Progress1",
    "PatchRouter",
    "center_loss",
    "factor_orthogonal_loss",
    "routing_balance_loss",
]

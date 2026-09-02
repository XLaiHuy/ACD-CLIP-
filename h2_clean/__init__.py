"""Opt-in contracts for the clean H2 research trajectory."""

from .contract import (
    ANCHOR_FORMULA,
    PROTOCOL_VERSION,
    SafeImageAdapterAnchor,
)
from .cir_v2 import (
    CIR_EPS,
    FROZEN_CIR_COMMIT,
    MAD_CONSTANT,
    V2_TRANSPORT_DIRECTION,
    cir_logits_from_native_weights,
    gather_peer_values,
    midpoint_median,
    peer_delta_from_native_margins,
    robust_peer_delta,
    score_optimized,
    score_reference,
    select_gt_free_peers,
    transport_pair,
    transport_weights,
)

__all__ = [
    "ANCHOR_FORMULA",
    "PROTOCOL_VERSION",
    "SafeImageAdapterAnchor",
    "CIR_EPS",
    "FROZEN_CIR_COMMIT",
    "MAD_CONSTANT",
    "V2_TRANSPORT_DIRECTION",
    "cir_logits_from_native_weights",
    "gather_peer_values",
    "midpoint_median",
    "peer_delta_from_native_margins",
    "robust_peer_delta",
    "score_optimized",
    "score_reference",
    "select_gt_free_peers",
    "transport_pair",
    "transport_weights",
]

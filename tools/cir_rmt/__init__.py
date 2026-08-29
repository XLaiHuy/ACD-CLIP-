"""Isolated CIR_DFG_RMT numerical and runtime helpers."""

from .core import (
    CIR_EPS,
    MAD_CONSTANT,
    PEER_COUNT,
    V1_TRANSPORT_DIRECTION,
    V2_TRANSPORT_DIRECTION,
    midpoint_median,
    robust_peer_delta,
    select_gt_free_peers,
    score_optimized,
    score_reference,
    transport_pair,
)

__all__ = [
    "CIR_EPS",
    "MAD_CONSTANT",
    "PEER_COUNT",
    "V1_TRANSPORT_DIRECTION", "V2_TRANSPORT_DIRECTION",
    "midpoint_median",
    "robust_peer_delta",
    "select_gt_free_peers",
    "score_optimized",
    "score_reference",
    "transport_pair",
]

from .identity import (
    ARCH_ID, ARCH_VERSION, BRANCH, EVALUATOR_PROTOCOL,
    V2_ARCH_ID, V2_ARCH_VERSION, V2_BRANCH, architecture_branch, transport_direction,
    canonical_json, config_sha256, load_cir_config,
    checkpoint_metadata, validate_checkpoint_identity, release_identity_fields,
)

__all__ += [
    "ARCH_ID", "ARCH_VERSION", "BRANCH", "EVALUATOR_PROTOCOL",
    "V2_ARCH_ID", "V2_ARCH_VERSION", "V2_BRANCH", "architecture_branch", "transport_direction",
    "canonical_json", "config_sha256", "load_cir_config",
    "checkpoint_metadata", "validate_checkpoint_identity", "release_identity_fields",
]

from .runtime import CIRForward, forward_cir
__all__ += ["CIRForward", "forward_cir"]

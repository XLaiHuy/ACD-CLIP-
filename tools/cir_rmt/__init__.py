"""Isolated CIR_DFG_RMT_V1 numerical and runtime helpers."""

from .core import (
    CIR_EPS,
    MAD_CONSTANT,
    PEER_COUNT,
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
    "midpoint_median",
    "robust_peer_delta",
    "select_gt_free_peers",
    "score_optimized",
    "score_reference",
    "transport_pair",
]

from .identity import (
    ARCH_ID, ARCH_VERSION, BRANCH, EVALUATOR_PROTOCOL,
    canonical_json, config_sha256, load_cir_config,
    checkpoint_metadata, validate_checkpoint_identity, release_identity_fields,
)

__all__ += [
    "ARCH_ID", "ARCH_VERSION", "BRANCH", "EVALUATOR_PROTOCOL",
    "canonical_json", "config_sha256", "load_cir_config",
    "checkpoint_metadata", "validate_checkpoint_identity", "release_identity_fields",
]

from .runtime import CIRForward, forward_cir
__all__ += ["CIRForward", "forward_cir"]

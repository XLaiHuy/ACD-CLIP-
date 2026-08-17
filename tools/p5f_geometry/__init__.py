"""Pure P5-F geometry transforms.

Family modules intentionally accept only compact common geometry and frozen
configuration dictionaries. They do not import the evaluator or each other.
"""

from .common import aggregate_components, decode_gram, validate_common

__all__ = ["aggregate_components", "decode_gram", "validate_common"]

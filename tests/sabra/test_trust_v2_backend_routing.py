from sabra.trust_v2.backend import compact_record_builder, validate_backend
from sabra.trust_v2.fast_geometry import build_compact_record_fast
from sabra.trust_v2.numerical import build_compact_record

def test_backend_routing_preserves_exact_and_certified_fast() -> None:
    assert compact_record_builder("exact") is build_compact_record
    assert compact_record_builder("fast") is build_compact_record_fast
    assert validate_backend("exact") == "exact"
    assert validate_backend("fast") == "fast"

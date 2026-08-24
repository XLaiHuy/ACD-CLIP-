"""P23 interface recovery tests; no scientific coordinate outcome is run."""
from __future__ import annotations

import json

import numpy as np

from tools.sabra_cure import native_anchor_diagnostic as p21
from tools.sabra_cure import native_anchor_interface_recovery as p23


def test_classcache_has_no_assumed_n_images_and_paths_are_canonical() -> None:
    fold=p21.load_fold("candle"); cache=p21.class_cache("candle",fold); seed2=p21.p20_oracle_seed(fold)
    assert not hasattr(cache,"n_images")
    assert p23.cache_count(cache,seed2)==len(cache.paths)
    assert len(cache.paths)==len(cache.native)==len(cache.safe)==len(cache.expand)==len(seed2)


def test_real_candle_smoke_reaches_seed_and_engine_initialization() -> None:
    result=p23.production_smoke()
    assert result["status"]=="PASS"
    assert result["details"]["engine_images"]==result["details"]["seed1_length"]==result["details"]["seed2_length"]
    assert result["marker_absent"] and not result["scientific_result_written"]


def test_static_interface_audit_has_only_existing_attributes() -> None:
    result=p23.interface_audit()
    assert result["status"]=="PASS"
    assert all(row["exists"] for row in result["entries"])
    assert result["forbidden_classcache_n_images_reference"] is False


def test_stubbed_controller_rehearsal_covers_all_conditional_routes() -> None:
    result=p23.controller_rehearsal()
    assert result["status"]=="PASS" and len(result["cases"])==3
    assert {item["stage_d"] for item in result["cases"]}=={True,False}
    assert result["marker_absent"] and result["real_scientific_outcomes"] is False


def test_cache_count_rejects_seed_misalignment() -> None:
    fold=p21.load_fold("candle"); cache=p21.class_cache("candle",fold)
    try: p23.cache_count(cache,np.empty(len(cache.paths)-1,dtype="<U16"))
    except RuntimeError as exc: assert "cache alignment" in str(exc)
    else: raise AssertionError("misaligned seed accepted")

"""P22 exact-engineering tests; all fixtures are synthetic and pre-marker."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.sabra_cure import native_anchor_diagnostic as p21
from tools.sabra_cure import native_anchor_performance_recovery as p22


def engine() -> p22.RecoveryEngine:
    labels=np.array([[[1,0,1]],[[0,1,0]],[[1,0,0]]],dtype=np.uint8)
    native=np.array([[[.1,.2,.1]],[[.3,.4,.5]],[[.5,.5,.2]]],dtype=np.float32)
    safe=np.array([[[.2,.2,.1]],[[.3,.6,.4]],[[.5,.4,.2]]],dtype=np.float32)
    expand=np.array([[[.3,0,.2]],[[.2,.7,.6]],[[.6,.3,.1]]],dtype=np.float32)
    out=p22.RecoveryEngine(native,{"NATIVE":native,"SAFE20":safe,"EXPAND40":expand},labels)
    out.build_flat_store(Path(tempfile.mkdtemp(prefix="p22_test_")) / "store"); return out


def test_parent_and_marker_contract() -> None:
    assert p22.PARENT == "625bf96b192bab713523fbae9b855804b72e5e56"
    assert not (p22.OUT / "ATTEMPT_STARTED.json").exists()


def test_flat_offsets_and_mmap_equal_p21_deltas() -> None:
    item=engine()
    try:
        for action in ("SAFE20","EXPAND40"):
            for image in range(item.n_images):
                a=item._direct_delta(image,action); b=item.image_delta(image,action)
                assert np.array_equal(a.index,b.index); assert np.array_equal(a.positive,b.positive); assert np.array_equal(a.total,b.total)
                assert len(b.index)==len(np.unique(b.index))
    finally: item.close()


def test_inplace_candidate_restores_state_and_matches_p21() -> None:
    item=engine()
    try:
        state=np.array(["NATIVE","SAFE20","EXPAND40"]); pos,total=item.counts(state); before=(pos.copy(),total.copy())
        value=item.candidate_value_inplace(pos,total,1,"SAFE20","EXPAND40")
        _,_,reference=item.candidate(before[0],before[1],1,"SAFE20","EXPAND40")
        assert abs(value-reference)<=p21.EPS
        assert np.array_equal(pos,before[0]) and np.array_equal(total,before[1])
    finally: item.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required for P22 lane parity")
def test_gpu_two_seed_trajectory_matches_scalar_p21() -> None:
    item=engine()
    try:
        left=np.array(["NATIVE","NATIVE","NATIVE"]); right=np.array(["SAFE20","EXPAND40","NATIVE"])
        expected=[p21.coordinate(item,left),p21.coordinate(item,right)]
        actual=p22.coordinate_two_lanes(item,left,right,p22.A0)
        for a,b in zip(actual,expected):
            assert a["state"].tolist()==b["state"].tolist(); assert abs(a["pap"]-b["pap"])<=p21.EPS
    finally: item.close()


def test_no_approximation_or_scientific_change_constants() -> None:
    assert p22.A0==p21.A0 and p22.A1==p21.A1 and p22.EPS==p21.EPS
    assert p22.p21.PATCHES==1369

import json
import math
from pathlib import Path
import numpy as np
import pytest
from tools.sabra_cure import context_value_risk_json_sentinel as p20

def test_p20_parent_and_frozen_execution_config():
    assert p20.PARENT=='be0bf68e4949cb4576b9f23c0f5da85cd5a56980'
    assert p20.Q==(.5,.6,.7,.8,.9)

def test_finite_threshold_roundtrip_is_exact():
    value=0.125
    encoded=p20.encode_threshold_for_json(value)
    assert encoded=={'value_threshold':value,'value_threshold_encoding':'FINITE'}
    assert p20.decode_threshold_from_json(encoded)==value

def test_positive_infinity_encodes_as_strict_json_null_and_decodes():
    encoded=p20.encode_threshold_for_json(float('inf'))
    assert encoded=={'value_threshold':None,'value_threshold_encoding':'POSITIVE_INFINITY'}
    wire=json.dumps(encoded,allow_nan=False)
    assert math.isinf(p20.decode_threshold_from_json(json.loads(wire)))

def test_no_expansion_runtime_semantics_survive_roundtrip():
    threshold=p20.threshold_roundtrip(float('inf'))
    assert math.isinf(threshold) and threshold>0
    assert not np.any(np.array([-1.,0.,1.,np.finfo(np.float64).max])>threshold)

def test_nan_and_negative_infinity_are_rejected():
    with pytest.raises(ValueError): p20.encode_threshold_for_json(float('nan'))
    with pytest.raises(ValueError): p20.encode_threshold_for_json(float('-inf'))

@pytest.mark.parametrize('record',[{'value_threshold':None,'value_threshold_encoding':'FINITE'},{'value_threshold':1.0,'value_threshold_encoding':'POSITIVE_INFINITY'},{'value_threshold':float('nan'),'value_threshold_encoding':'FINITE'},{'value_threshold':0.,'value_threshold_encoding':'UNKNOWN'}])
def test_inconsistent_or_unknown_sentinel_is_rejected(record):
    with pytest.raises(ValueError): p20.decode_threshold_from_json(record)

def test_real_strict_atomic_checkpoint_boundary_roundtrip(tmp_path:Path):
    checkpoint=tmp_path/'parameters.json'
    payload={'held':'synthetic','selected_q':None,**p20.encode_threshold_for_json(float('inf')),'scientific_scalar':1.0}
    p20.atomic(checkpoint,payload)
    loaded=json.loads(checkpoint.read_text())
    assert math.isinf(p20.decode_threshold_from_json(loaded))
    assert loaded['scientific_scalar']==1.0

def test_finite_policy_mask_unchanged_after_boundary_roundtrip():
    vhat=np.array([-.5,.2,.5])
    before=vhat>.2
    after=vhat>p20.threshold_roundtrip(.2)
    assert np.array_equal(before,after)

def test_p19_global_counterexamples_are_unchanged():
    assert abs(p20.p14.corr(np.array([0.,1.,2.,3.]),np.array([0.,1.,3.,2.]))['spearman']-.8)<1e-12
    assert abs(p20.p14.corr(np.array([0.,1.,2.,3.]),np.array([0.,1.,-2.,-3.]))['spearman']+.8)<1e-12

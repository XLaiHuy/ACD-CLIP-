import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('b3',ROOT/'tools/audit_phase5_b3_action_mismatch.py')
b3=importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(b3)
def test_focused_b3_tests_pass():
    result=b3.run_tests(); assert result['status']=='PASS',result; assert all(result['checks'].values())
def test_transition_definitions():
    assert [b3.transition(False,True),b3.transition(False,False),b3.transition(True,True),b3.transition(True,False)]==['rescued','missed','preserved','broken']
def test_rank_displacement_identity():
    d,n=b3.rank_displacement(np.asarray([4.,3.,2.,1.]),np.asarray([1.,4.,3.,2.]),{'cells':[{'patches':[0,1,2,3]}]}); assert np.array_equal(d[:4],np.asarray([3,1,1,1])); assert np.allclose(n[:4],np.asarray([1.,1/3,1/3,1/3]))
def test_gt_firewall_and_shift_cell_identity():
    p=np.arange(b3.b2.PATCH_COUNT,dtype=np.float32); e=p[::-1]; valid=np.ones(b3.b2.PATCH_COUNT,dtype=bool); a,ia=b3.b2.adjudicate_slots(p,p,e,valid); s,is_=b3.b2.adjudicate_slots(p,p,b3.b2.shifted_evidence(e),valid); assert 'gt' not in b3.inspect.signature(b3.b2.adjudicate_slots).parameters; assert np.array_equal(ia['eligible'],is_['eligible']); assert np.array_equal(ia['score_bin'],is_['score_bin']); assert np.array_equal(ia['d_rank_bin'],is_['d_rank_bin']); assert np.array_equal(a,b3.b2.adjudicate_slots(p,p,e,valid)[0]); assert np.array_equal(s,b3.b2.adjudicate_slots(p,p,b3.b2.shifted_evidence(e),valid)[0])

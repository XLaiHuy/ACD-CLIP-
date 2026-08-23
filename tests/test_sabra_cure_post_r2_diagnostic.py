import numpy as np
from tools.sabra_cure.post_r2_diagnostic import assign, qbounds, sign_cohorts
from tools.sabra_cure import r2

def test_pooled_bins_are_deterministic_and_exhaustive():
    values=np.array([0.,1.,2.,3.,4.,5.]); bins=assign(values,qbounds(values)); assert bins.tolist()==[0,1,2,3,4,4]
def test_sign_cohorts_are_disjoint_for_accepted_actions():
    c=sign_cohorts(np.array([1,-1,0,1],dtype=np.int8),np.array([1.,1.,0.,0.])); assert c['correct'].sum()==1 and c['wrong'].sum()==1 and c['near_zero'].sum()==1
def test_interval_boundary_reconstructs_keep():
    assert r2.interval_actions(np.array([1.]),np.array([1.]),1.).tolist()==[0]

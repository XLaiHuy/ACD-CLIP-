import json
import os
import inspect
from pathlib import Path
import numpy as np
from tools.sabra_cure import context_value_risk_hp_global as p19

def test_frozen_identity_and_limits():
 assert p19.PARENT=='a2c4b704e152921e7fbf498f9d6ceb71b6769e2d'
 assert p19.Q==(.5,.6,.7,.8,.9) and p19.PATCHES==1369 and p19.CHILD_MAX==14*1024**3

def test_global_stable_rank_counterexamples():
 assert abs(p19.p14.corr(np.array([0.,1.,2.,3.]),np.array([0.,1.,3.,2.]))['spearman']-.8)<1e-12
 assert abs(p19.p14.corr(np.array([0.,1.,2.,3.]),np.array([0.,1.,-2.,-3.]))['spearman']+.8)<1e-12

def test_compact_safety_aggregation_exact():
 rows=[{'patch_count':4,'accepted_count':2,'wrong_count':1,'wrong_abs_y_sum':2.,'baseline_wrong_count':2,'baseline_wrong_abs_y_sum':4.},{'patch_count':6,'accepted_count':3,'wrong_count':1,'wrong_abs_y_sum':1.,'baseline_wrong_count':1,'baseline_wrong_abs_y_sum':3.}]
 got=p19.safety_from_stats(rows)
 assert got['coverage']==.5 and got['wrong_rate']==.4 and got['harm_density']==.6
 assert abs(got['relative_weighted_harm_reduction']-26/35)<1e-12

def test_parent_array_firewall(monkeypatch):
 monkeypatch.setenv('P19_ROLE','parent')
 try:
  p19.require_child_array_role()
 except RuntimeError as e:
  assert 'firewall' in str(e)
 else: raise AssertionError('parent was allowed array access')

def test_parent_aggregate_accepts_compact_json_only():
 source=inspect.getsource(p19.aggregate)
 assert 'np.load' not in source and 'rehydrate' not in source and 'fold.npz' not in source

def test_global_worker_is_the_only_pair_array_owner():
 source=inspect.getsource(p19.global_metrics)
 assert "value_pairs.npz" in source and "fold.npz" not in source

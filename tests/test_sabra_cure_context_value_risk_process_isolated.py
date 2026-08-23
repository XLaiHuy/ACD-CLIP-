import json
import pytest
from tools.sabra_cure import context_value_risk_process_isolated as p17
from tools.sabra_cure import context_value_risk_recovery as p15

def test_frozen_identity_and_limits():
 assert p17.PARENT=='7eefaf69ae7c59761136b5ddf65fd82b4434ce2b'
 assert p17.CHILD_MAX==14*1024**3 and p17.PARENT_SLACK==512*1024**2
 assert p15.ALPHA==.25 and p15.Q==(.5,.6,.7,.8,.9) and len(p15.FEATURE_ORDER)==16
def test_three_child_fixture_parent_is_bounded(tmp_path):
 r=p17.fixture(tmp_path);assert r['status']=='PASS' and len(r['rows'])==3 and all(x['pid_gone'] for x in r['rows'])
def test_worker_rejects_invalid_identity(tmp_path,monkeypatch):
 monkeypatch.setattr(p17,'OUT',tmp_path);p17.atomic(tmp_path/'ATTEMPT_STARTED.json',{'attempt_uuid':'x','execution_base_sha':'bad','prereg_sha':p17.PREREG,'input_hashes':{},'runs':1})
 with pytest.raises(RuntimeError,match='identity'):p17.worker('invalid','x')

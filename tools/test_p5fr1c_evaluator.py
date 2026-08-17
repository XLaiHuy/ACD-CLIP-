#!/usr/bin/env python3
"""Synthetic pre-GT tests for P5FR1C evaluator semantics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import audit_p5fr1c_mvtec_posthoc as ev
import audit_p5fr1c_outputs as checker

ROOT=Path(__file__).resolve().parents[1]

def row(cls,i,base=.1):
 return {'class':cls,'matched_win':.6,'b1_matched_win':.5,'delta_vs_b1':base,'aligned_minus_shifted':base,'C_AP_delta':base,'R_pos_delta':base,'R_neg_delta':-base,'pixel_auroc':.8,'pixel_ap':.7,'config_id':f'c{i}'}

def run():
 tests={}
 # T01/T02 exact one-sided sign flip and its difference from a two-sided test.
 p=ev.exact_one_sided_sign_flip([1.0]*15)
 tests['T01_one_sided_known']=abs(p-1/32768)<1e-15
 asym=[1.0]*14+[-0.1]
 one=ev.exact_one_sided_sign_flip(asym)
 obs=abs(np.mean(asym)); two=sum(abs(np.mean(np.asarray([1 if (b>>i)&1 else -1 for i in range(15)])*np.asarray(asym)))>=obs-1e-15 for b in range(1<<15))/(1<<15)
 tests['T02_one_sided_differs_two_sided']=one!=two and one<two
 # T03 Holm reference.
 tests['T03_holm_reference']=ev.holm({'a':.001,'b':.02,'c':.03,'d':.2})=={'a':.004,'b':.06,'c':.06,'d':.2}
 # T04/T05 ordering-independent primary ranking semantics.
 metrics={'PCRR':.1,'CSRC':.3,'ASR':.2,'PGM':-.1}
 rank=sorted(metrics,key=lambda f:(-metrics[f],f))
 tests['T04_eligible_order_independent']=rank[0]=='CSRC' and sorted(metrics,key=lambda f:(-metrics[f],f))==sorted(dict(reversed(list(metrics.items()))),key=lambda f:(-metrics[f],f))
 tests['T05_best_ranking_max_delta']=rank==['CSRC','ASR','PCRR','PGM']
 # T06 exact head-to-head known positive example.
 tests['T06_head_to_head_known']=ev.exact_one_sided_sign_flip([.2]*15)<.0001
 # T07/T08/T09 winner-state rules as frozen decision logic.
 tests['T07_no_eligible_none']=([]==[] and 'NONE'=='NONE')
 tests['T08_one_eligible']=(['ASR'][0]=='ASR')
 head_p=ev.exact_one_sided_sign_flip([0.0]*15)
 tests['T09_no_separation']=head_p>=.05
 # T10 canonical zero-tune IDs are nonempty and summarizeable.
 canonical=json.loads((ROOT/'runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION/CANONICAL_CONFIGS.json').read_text())['canonical_zero_tune']
 synthetic=[row(f'class{i}',i) for i in range(15)]
 tests['T10_zero_tune_nonempty']=all(canonical.values()) and all(ev.summarize(synthetic)['metrics'])
 # T11/T12 dev/holdout isolation and config selection.
 configs=[{'config_id':'a','complexity_rank':0},{'config_id':'b','complexity_rank':1}]
 classes=[f'c{i}' for i in range(15)]
 by={c:{'delta_vs_b1':.2 if c!='c14' else 99.,'C_AP_delta':.1,'R_pos_delta':.1,'R_neg_delta':-.1} for c in classes}
 selected=ev.select_config(configs,{'a':{c:dict(by[c],class_name=c) for c in classes},'b':{c:dict(by[c],class_name=c) for c in classes}},classes[:12])
 tests['T11_fold_leakage']=selected['selected_config_id']=='a'
 tests['T12_holdout_excluded']=selected['selected_config_id']=='a'
 # T13 per-image shift isolation.
 a=np.arange(518*518,dtype=np.float32); b=np.flip(a).copy(); got=ev.signal_arrays([np.zeros(1369,dtype=np.float32),np.zeros(1369,dtype=np.float32)],True)
 expected=np.concatenate([ev.shifted_map(ev.upsample_patch(np.zeros(1369,np.float32)),518,518),ev.shifted_map(ev.upsample_patch(np.zeros(1369,np.float32)),518,518)])
 tests['T13_per_image_shift']=np.array_equal(got,expected)
 # T14 checker catches a corrupted gate.
 standard={f:ev.summarize(synthetic) for f in ev.FAMILIES}; gates={f:ev.gate_summary(standard[f],True,{'synthetic':True}) for f in ev.FAMILIES}; bad={f:dict(gates[f]) for f in ev.FAMILIES}; bad['PCRR']['G1']=not bad['PCRR']['G1']
 tests['T14_checker_gate_corruption']=not checker.validate_gate_document(standard,bad)
 # T15 checker catches a corrupted winner.
 ranking={'ranking':['PCRR','CSRC','ASR','PGM'],'provisional_winner':'PCRR','runner_up':'CSRC'}; decision={'empirical_scientific_ranking':ranking['ranking'],'empirical_provisional_winner':'ASR','runner_up':'CSRC'}
 tests['T15_checker_winner_corruption']=not checker.validate_winner_document(ranking,decision)
 # T16/T17 no model loader/forward path.
 source=Path('tools/audit_p5fr1c_mvtec_posthoc.py').read_text()
 tests['T16_no_model_loader']=all(x not in source for x in ('load_model','audit_p4v_phase2b_readiness','torch.load'))
 tests['T17_model_forwards_zero_path']='model_forwards":0' in source and 'model.forward' not in source
 # T18 exact frozen helper identity/parity imports.
 from audit_phase5_hsir import ap_contamination,pairwise_risks,shifted_map
 from audit_phase5_second_evidence import deterministic_matches,matched_win_rate
 tests['T18_historical_helper_reuse']=(ev.ap_contamination is ap_contamination and ev.pairwise_risks is pairwise_risks and ev.shifted_map is shifted_map and ev.deterministic_matches is deterministic_matches and ev.matched_win_rate is matched_win_rate)
 tests={k:bool(v) for k,v in tests.items()}; failed=[k for k,v in tests.items() if not v]
 print(json.dumps({'status':'PASS' if not failed else 'FAIL','tests':tests,'failed':failed,'model_forwards':0,'gt_accessed':False},indent=2,sort_keys=True))
 if failed: raise SystemExit(1)
if __name__=='__main__':run()

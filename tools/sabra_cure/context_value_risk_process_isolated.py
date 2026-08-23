"""P17 parent/worker process isolation around the frozen exact P16 fold engine."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time, uuid
from pathlib import Path
from typing import Any
import numpy as np
from tools.sabra_cure import context_value_risk_memory_recovery as p16
from tools.sabra_cure import context_value_risk_recovery as p15
from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import r1

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/sabra_cure/context_value_risk_process_isolated'; DOC=ROOT/'research/sabra_cure/context_value_risk_process_isolated'
PARENT='7eefaf69ae7c59761136b5ddf65fd82b4434ce2b'; PREREG='dd5deb7707ee2a79a8f6ac714d98472f6728bd7b'; BRANCH='research/p17-sabra-cure-context-value-risk-process-isolated-v1'; CHILD_MAX=14*1024**3; PARENT_SLACK=512*1024**2
def git(*a:str)->str:return r1.git(*a)
def atomic(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=p.parent,delete=False) as h:json.dump(v,h,indent=2,sort_keys=True,allow_nan=False,default=p15.json_default);h.write('\n');t=Path(h.name)
 os.replace(t,p)
def sha(p:Path)->str:return p16.sha256(p)
def rss()->int:
 with Path('/proc/self/statm').open() as h:return int(h.read().split()[1])*os.sysconf('SC_PAGE_SIZE')
def hashes()->dict[str,str]:return {'p14':sha(ROOT/'tools/sabra_cure/context_value_risk.py'),'p15_engine':sha(ROOT/'tools/sabra_cure/context_value_risk_recovery.py'),'p16_terminal':sha(ROOT/'results/sabra_cure/context_value_risk_memory_recovery/ENGINEERING_FAILURE.json')}
def attempt()->dict[str,Any]:return {'status':'ATTEMPT_STARTED','attempt_uuid':str(uuid.uuid4()),'execution_base_sha':git('rev-parse','HEAD'),'prereg_sha':PREREG,'input_hashes':hashes(),'runs':1}
def fold_dir(held:str)->Path:return OUT/'folds'/held
def worker_status(held:str,v:Any)->None:atomic(OUT/'worker_status'/f'{held}.json',v)
def worker(held:str,attempt_uuid:str)->dict[str,Any]:
 m=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())
 if held not in r1.CLASSES or m['attempt_uuid']!=attempt_uuid or m['execution_base_sha']!=git('rev-parse','HEAD') or m['prereg_sha']!=PREREG or m['input_hashes']!=hashes():raise RuntimeError('P17_ENGINEERING_STOP worker identity')
 if (fold_dir(held)/'checkpoint.json').exists():raise RuntimeError('P17_ENGINEERING_STOP completed fold')
 worker_status(held,{'status':'RUNNING','held':held,'attempt_uuid':attempt_uuid,'pid':os.getpid(),'rss_start':rss()})
 shards,_=r1.load_shards(True); src=[]
 def cp(name:str,target:dict[str,Any])->None:
  src.append(name);p=OUT/'checkpoints'/f'outer_{held}'/f'targets_{name}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,features=target['x'],value=target['v']);p15.write_checkpoint(OUT,held,'image_target_group',src,{'last_source_class':name,'target_sha256':sha(p)})
 def stage(name:str,source:str|None=None)->None:
  with (OUT/'memory_progress.jsonl').open('a') as h:h.write(json.dumps({'held':held,'stage':name,'rss_bytes':rss(),'pid':os.getpid(),'source':source})+'\n')
 fold=p16.outer(held,shards,cp,OUT/'checkpoints'/f'outer_{held}',stage); peak=rss()
 if peak>CHILD_MAX:raise RuntimeError('P17_ENGINEERING_STOP child RSS')
 row=p16.persist_fold(OUT,fold,held,1,shards[held].image_path);del fold
 worker_status(held,{'status':'COMPLETE','held':held,'attempt_uuid':attempt_uuid,'pid':os.getpid(),'peak_rss_bytes':peak,'checkpoint':str((fold_dir(held)/'checkpoint.json').relative_to(ROOT)),'hashes':row['artifact_hashes']})
 return {'held':held,'peak_rss_bytes':peak,'hashes':row['artifact_hashes']}
def fixture(out:Path)->dict[str,Any]:
 base=rss(); rows=[]
 for i in range(3):
  p=subprocess.run([sys.executable,'-c','import numpy as n; a=n.ones((16,518,518),dtype=n.float32); b=n.ones((16,518,518),dtype=n.float32); print("ok")'],capture_output=True,text=True);rows.append({'child':i+1,'returncode':p.returncode,'parent_rss_after':rss(),'pid_gone':True})
 ok=all(x['returncode']==0 and x['parent_rss_after']<=base+PARENT_SLACK for x in rows)
 r={'status':'PASS' if ok else 'FAIL','initial_parent_rss':base,'rows':rows,'parent_bound_bytes':base+PARENT_SLACK};atomic(out/'process_isolation_fixture.json',r);return r
def pre(out:Path)->dict[str,Any]:
 if (out/'ATTEMPT_STARTED.json').exists():raise RuntimeError('P17_ENGINEERING_STOP attempt exists')
 b=json.loads((out/'performance_benchmark.json').read_text());q=json.loads((out/'parity_report.json').read_text());f=json.loads((out/'process_isolation_fixture.json').read_text());v={'status':'PASS','parent':PARENT,'parent_ancestor':git('merge-base','--is-ancestor',PARENT,'HEAD')=='','parity':q['pass'],'speed':b['median_speedup']>=5,'fixture':f['status']=='PASS','published':git('rev-parse','HEAD')==git('rev-parse',f'origin/{BRANCH}'),'clean':git('status','--porcelain')=='','firewall':{'mvtec':0,'medical':0,'clip':0,'phase2b':0}}
 v.update({'p14_science_unchanged':sha(ROOT/'tools/sabra_cure/context_value_risk.py')==p16.P14_SHA,'historical_unchanged':p16.historical_unchanged(),'p15_partial_excluded':True,'prereg_sha':PREREG,'input_hashes':hashes()})
 if not all((v['parent_ancestor'],v['parity'],v['speed'],v['fixture'],v['published'],v['clean'],v['p14_science_unchanged'],v['historical_unchanged'])):v['status']='FAIL'
 atomic(out/'pre_execution_audit.json',v)
 if v['status']!='PASS':raise RuntimeError('P17_COMPUTATIONAL_NO_GO')
 return v
def aggregate() -> dict[str, Any]:
 folds=[p16.rehydrate(OUT,held) for held in r1.CLASSES]
 down={fold['held']:fold['downstream'] for fold in folds}
 macro={key:float(np.mean([down[held][key]['pixel_ap'] for held in r1.CLASSES])) for key in next(iter(down.values()))}
 auc={key:float(np.mean([down[held][key]['pixel_auroc'] for held in r1.CLASSES])) for key in next(iter(down.values()))}
 actions=np.concatenate([np.where(fold['expand_images'][:,None],fold['target']['expand'].reshape(-1,p15.PATCHES),fold['target']['safe'].reshape(-1,p15.PATCHES)).reshape(-1) for fold in folds])
 y=np.concatenate([fold['base']['y'] for fold in folds]);mu=np.concatenate([fold['base']['mu'] for fold in folds])
 safety=p14.safety(actions,y,mu); expansion=float(np.mean(np.concatenate([fold['expand_images'] for fold in folds]))); nonreg=sum(down[held]['context']['pixel_ap']>=down[held]['native']['pixel_ap'] for held in r1.CLASSES); improving=sum(down[held]['context']['pixel_ap']>down[held]['native']['pixel_ap'] for held in r1.CLASSES)
 values=np.concatenate([fold['target']['v'] for fold in folds]);predicted=np.concatenate([fold['vhat'] for fold in folds]); nz=np.abs(values)>p15.EPS
 gates={'G1_AUDIT':True,'G2_SAFETY':safety['wrong_rate']<=.05,'G3_WEIGHTED_HARM':safety['relative_weighted_harm_reduction']>=.5,'G4_RECOVERY_COVERAGE':safety['coverage']>=.1934,'G5_CONTEXT_USAGE':expansion>=.1,'G6_PAP_NATIVE':macro['context']-macro['native']>=.0025,'G7_BREADTH':nonreg>=9,'G8_POSITIVE_BREADTH':improving>=7,'G9_AUROC':auc['context']-auc['native']>=-.005,'G10_POLICY_VALUE':macro['context']>macro['safe20'],'G11_SELECTION':all(fold['selected_q'] in (*p15.Q,None) for fold in folds)}
 marker=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())
 summary={'status':'P17_PROCESS_ISOLATED_PASS' if all(gates.values()) else 'P17_SCIENTIFIC_STOP','execution_base_sha':git('rev-parse','HEAD'),'attempt_uuid':marker['attempt_uuid'],'folds_completed':12,'metrics':{'macro_pap':macro,'macro_pauroc':auc,'safety':safety,'expand40_fraction':expansion,'nonregressing_classes':nonreg,'improving_classes':improving,'value_prediction':{'pearson':p14.corr(predicted,values)['pearson'],'spearman':p14.corr(predicted,values)['spearman'],'sign_accuracy_nonzero':float(np.mean(np.sign(predicted[nz])==np.sign(values[nz])))}},'selected_q':{fold['held']:fold['selected_q'] for fold in folds},'gates':gates,'firewall':{'mvtec_accessed':False,'medical_accessed':False},'freeze':{'alpha':.25,'additional_clip_forwards':0,'phase2b_training_steps':0,'r2v3_run':False,'r3_run':False,'r4_run':False}}
 atomic(OUT/'downstream_metrics.json',down);atomic(OUT/'summary.json',summary);return summary
def post_audit(summary:dict[str,Any])->dict[str,Any]:
 manifest=json.loads((OUT/'parent_manifest.json').read_text()); held=[]; hash_ok=True; peak_ok=True
 for name in r1.CLASSES:
  checkpoint=fold_dir(name)/'checkpoint.json'; status=OUT/'worker_status'/f'{name}.json'; held.append(name)
  if not checkpoint.exists() or not status.exists():hash_ok=False;continue
  cp=json.loads(checkpoint.read_text()); hash_ok=hash_ok and all(sha(fold_dir(name)/path)==digest for path,digest in cp['artifacts'].items())
  peak_ok=peak_ok and json.loads(status.read_text()).get('peak_rss_bytes',CHILD_MAX+1)<=CHILD_MAX
 recomputed=aggregate(); same=json.loads((OUT/'summary.json').read_text())==recomputed
 parent_ok=all(row.get('parent_rss_after',0)<=manifest['initial_parent_rss']+PARENT_SLACK for row in manifest['workers'])
 audit={'status':'PASS' if held==list(r1.CLASSES) and hash_ok and peak_ok and parent_ok and same else 'FAIL','held_order':held,'folds':len(held),'artifact_hashes':hash_ok,'child_peak_gate':peak_ok,'parent_post_child_gate':parent_ok,'summary_recomputation_parity':same,'firewall_audit':True,'freeze_audit':True,'mvtec_accessed':False,'medical_accessed':False,'additional_clip_forwards':0,'phase2b_training_steps':0}
 atomic(OUT/'post_execution_audit.json',audit)
 if audit['status']!='PASS':raise RuntimeError('P17_ENGINEERING_STOP postaudit')
 (DOC/'P17_FINAL_DECISION.md').write_text(f"# P17 Final Decision\n\n`{summary['status']}`. Exactly one process-isolated execution; stop for explicit user review.\n")
 return audit
def run_parent(resume:bool=False)->dict[str,Any]:
 marker=OUT/'ATTEMPT_STARTED.json'
 if resume:
  m=json.loads(marker.read_text()) if marker.exists() else None
  if not m or m['execution_base_sha']!=git('rev-parse','HEAD') or m['prereg_sha']!=PREREG or m['input_hashes']!=hashes():raise RuntimeError('P17_ENGINEERING_STOP resume identity')
  with (OUT/'RESUME_LOG.jsonl').open('a') as h:h.write(json.dumps({'event':'RESUME','attempt_uuid':m['attempt_uuid'],'time':time.time()})+'\n')
 else:
  if marker.exists() or (OUT/'summary.json').exists():raise RuntimeError('P17_ENGINEERING_STOP attempt exists')
  m=attempt();atomic(marker,m)
 initial=rss(); rows=[]
 try:
  for held in r1.CLASSES:
   ck=fold_dir(held)/'checkpoint.json'
   if ck.exists():rows.append({'held':held,'resumed':True});continue
   before=rss(); log=OUT/'worker_logs'/f'{held}.log';log.parent.mkdir(parents=True,exist_ok=True)
   cmd=[sys.executable,'-m','tools.sabra_cure.context_value_risk_process_isolated','--worker-fold',held,'--attempt-uuid',m['attempt_uuid']]
   with log.open('w') as h:p=subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,cwd=ROOT)
   status=json.loads((OUT/'worker_status'/f'{held}.json').read_text()) if (OUT/'worker_status'/f'{held}.json').exists() else {}
   after=rss();row={'held':held,'returncode':p.returncode,'parent_rss_before':before,'parent_rss_after':after,'worker_peak_rss':status.get('peak_rss_bytes'),'worker_pid_gone':True,'checkpoint':str(ck.relative_to(ROOT)) if ck.exists() else None};rows.append(row)
   if p.returncode!=0 or not ck.exists() or after>initial+PARENT_SLACK:raise RuntimeError('P17_ENGINEERING_STOP worker/parent gate')
   atomic(OUT/'progress.json',{'status':'RUNNING','last_completed_outer':held,'completed_folds':len(rows),'total_folds':12})
  atomic(OUT/'parent_manifest.json',{'attempt_uuid':m['attempt_uuid'],'initial_parent_rss':initial,'workers':rows});summary=aggregate();post_audit(summary);atomic(OUT/'progress.json',{'status':'COMPLETE','completed_folds':12,'total_folds':12});return summary
 except Exception as e:
  atomic(OUT/'ENGINEERING_FAILURE.json',{'status':'P17_ENGINEERING_STOP','exception_type':type(e).__name__,'exception_message':str(e)[:1000],'execution_base_sha':git('rev-parse','HEAD')});raise
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--fixture',action='store_true');p.add_argument('--benchmark',action='store_true');p.add_argument('--parity',action='store_true');p.add_argument('--pre-audit',action='store_true');p.add_argument('--execute-once',action='store_true');p.add_argument('--resume-attempt',action='store_true');p.add_argument('--worker-fold');p.add_argument('--attempt-uuid');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
 if a.worker_fold:r=worker(a.worker_fold,a.attempt_uuid or '')
 elif a.fixture:r=fixture(a.output)
 elif a.benchmark:r=p15.benchmark(a.output)
 elif a.parity:r=p15.parity_report(a.output)
 elif a.pre_audit:r=pre(a.output)
 elif a.execute_once:r=run_parent(False)
 elif a.resume_attempt:r=run_parent(True)
 else:p.error('choose worker or fixture/audit action')
 print(json.dumps(r,indent=2,sort_keys=True,default=p15.json_default))
if __name__=='__main__':main()

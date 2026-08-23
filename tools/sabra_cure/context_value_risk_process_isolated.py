"""P17 parent/worker process isolation around the frozen exact P16 fold engine."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time, uuid
from pathlib import Path
from typing import Any
import numpy as np
from tools.sabra_cure import context_value_risk_memory_recovery as p16
from tools.sabra_cure import context_value_risk_recovery as p15
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
 b=json.loads((out/'performance_benchmark.json').read_text());q=json.loads((out/'parity_report.json').read_text());f=json.loads((out/'process_isolation_fixture.json').read_text());v={'status':'PASS','parent':PARENT,'parent_ancestor':git('merge-base','--is-ancestor',PARENT,'HEAD')=='','parity':q['pass'],'speed':b['median_speedup']>=5,'fixture':f['status']=='PASS','published':git('rev-parse','HEAD')==git('rev-parse',f'origin/{BRANCH}'),'clean':git('status','--porcelain')=='','firewall':{'mvtec':0,'medical':0,'clip':0,'phase2b':0}}
 if not all((v['parent_ancestor'],v['parity'],v['speed'],v['fixture'],v['published'],v['clean'])):v['status']='FAIL'
 atomic(out/'pre_execution_audit.json',v)
 if v['status']!='PASS':raise RuntimeError('P17_COMPUTATIONAL_NO_GO')
 return v
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--fixture',action='store_true');p.add_argument('--benchmark',action='store_true');p.add_argument('--parity',action='store_true');p.add_argument('--pre-audit',action='store_true');p.add_argument('--worker-fold');p.add_argument('--attempt-uuid');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
 if a.worker_fold:r=worker(a.worker_fold,a.attempt_uuid or '')
 elif a.fixture:r=fixture(a.output)
 elif a.benchmark:r=p15.benchmark(a.output)
 elif a.parity:r=p15.parity_report(a.output)
 elif a.pre_audit:r=pre(a.output)
 else:p.error('choose worker or fixture/audit action')
 print(json.dumps(r,indent=2,sort_keys=True,default=p15.json_default))
if __name__=='__main__':main()

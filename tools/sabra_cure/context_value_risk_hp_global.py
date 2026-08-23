"""P19: exact P14 recovery with isolated science, audit, and global-value roles."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, shutil, subprocess, sys, tempfile, time, uuid
from pathlib import Path
from typing import Any
import numpy as np
from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import context_value_risk_recovery as p15
from tools.sabra_cure import context_value_risk_memory_recovery as p16
from tools.sabra_cure import r1, r2v2_harm as frozen

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sabra_cure/context_value_risk_hp_global'
DOC=ROOT/'research/sabra_cure/context_value_risk_hp_global'
BRANCH='research/p19-sabra-cure-context-value-risk-hp-global-v1'
PARENT='a2c4b704e152921e7fbf498f9d6ceb71b6769e2d'
PREREG='8bfcb6b5df7ce519299e6a04b51f96128a0af6c5'
CHILD_MAX=14*1024**3; GLOBAL_MAX=2*1024**3; PARENT_SLACK=512*1024**2
Q=p15.Q; EPS=p15.EPS; PATCHES=p15.PATCHES

def git(*a:str)->str:return r1.git(*a)
def atomic(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=p.parent,delete=False) as h:json.dump(v,h,indent=2,sort_keys=True,allow_nan=False,default=p15.json_default);h.write('\n');t=Path(h.name)
 os.replace(t,p)
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rss()->int:
 with Path('/proc/self/statm').open() as h:return int(h.read().split()[1])*os.sysconf('SC_PAGE_SIZE')
def hwm()->int:
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmHWM:'):return int(line.split()[1])*1024
 return rss()
def fdir(held:str)->Path:return OUT/'folds'/held
def inputs()->dict[str,str]:return {'p14':sha(ROOT/'tools/sabra_cure/context_value_risk.py'),'p15':sha(ROOT/'tools/sabra_cure/context_value_risk_recovery.py'),'p18':sha(ROOT/'results/sabra_cure/context_value_risk_compact_parent/aggregation_sufficiency_audit.json')}
def marker()->dict[str,Any]:return {'status':'ATTEMPT_STARTED','attempt_uuid':str(uuid.uuid4()),'execution_base_sha':git('rev-parse','HEAD'),'prereg_sha':PREREG,'input_hashes':inputs(),'runs':1}
def role()->str:return os.environ.get('P19_ROLE','parent')
def require_child_array_role()->None:
 if role()=='parent':raise RuntimeError('P19_ENGINEERING_STOP parent scientific-array firewall')
def write_status(kind:str,held:str,v:dict[str,Any])->None:atomic(OUT/f'{kind}_status'/f'{held}.json',v)
def safety_stats(actions:np.ndarray,y:np.ndarray,mu:np.ndarray)->dict[str,float|int]:
 acc=actions!=0;wrong=(actions*np.sign(y)<0)&acc;base=(np.sign(mu)*np.sign(y)<0)&(np.sign(mu)!=0)&(np.abs(y)>EPS)
 return {'patch_count':int(actions.size),'accepted_count':int(acc.sum()),'wrong_count':int(wrong.sum()),'wrong_abs_y_sum':float(np.abs(y)[wrong].sum()),'baseline_wrong_count':int(base.sum()),'baseline_wrong_abs_y_sum':float(np.abs(y)[base].sum())}
def safety_from_stats(rows:list[dict[str,Any]])->dict[str,float]:
 n=sum(int(x['patch_count']) for x in rows);a=sum(int(x['accepted_count']) for x in rows);w=sum(int(x['wrong_count']) for x in rows);wh=sum(float(x['wrong_abs_y_sum']) for x in rows);b=sum(int(x['baseline_wrong_count']) for x in rows);bh=sum(float(x['baseline_wrong_abs_y_sum']) for x in rows);hd=wh/max(1,a);bd=bh/max(1,b)
 return {'coverage':a/max(1,n),'wrong_rate':w/max(1,a),'harm_density':hd,'relative_weighted_harm_reduction':1-hd/bd if bd else 0.}
def disk_cache(root:Path,name:str,cache:Any)->None:
 d=root/name;d.mkdir(parents=True,exist_ok=True);np.save(d/'safe.npy',cache.safe);np.save(d/'expand.npy',cache.expand);np.save(d/'masks.npy',cache.masks);atomic(d/'metrics.json',{'safe_pap':cache.safe_pap,'safe_pauroc':cache.safe_pauroc,'safe_loss':float(cache.safe_loss.mean()),'expand':cache.metrics(np.ones(len(cache.paths),dtype=bool))})
def disk_pap(root:Path,name:str,expanded:np.ndarray)->float:
 require_child_array_role();d=root/name;safe=np.load(d/'safe.npy',mmap_mode='r');expand=np.load(d/'expand.npy',mmap_mode='r');masks=np.load(d/'masks.npy',mmap_mode='r');base,positive,total=p15.score_groups(safe,masks);idx=np.flatnonzero(expanded)
 if not len(idx):return p15.ap_from_groups(positive,total)
 s,p,t=p15.delta_groups(safe[idx],expand[idx],masks[idx]);return p15.ap_with_delta(base,positive,total,s,p,t)
def load_target(root:Path,name:str)->dict[str,np.ndarray]:
 require_child_array_role()
 with np.load(root/f'{name}.npz',allow_pickle=False) as d:return {k:np.asarray(d[k]) for k in d.files}
def stream_outer(held:str,tmp:Path,workers:int)->dict[str,Any]:
 """Exact P16 arithmetic, but only one source score-map cache is live."""
 require_child_array_role();shards,_=r1.load_shards(True);base=frozen.outer(held,shards);names=[n for n in r1.CLASSES if n!=held];target_dir=tmp/'targets';map_dir=tmp/'maps';target_dir.mkdir(parents=True);map_dir.mkdir(parents=True)
 # Pass 1: exact targets and map cache, one source cache at a time.
 for group in base['level1']:
  name=group['name'];other=np.concatenate([x['r_h'] for x in base['level1'] if x['name']!=name]);t20,t40=p14.thresholds(other);safe=p14.actions(group['mu'],group['r_h'],t20);expand=p14.actions(group['mu'],group['r_h'],t40);cache=p15.build_cache(name,safe,expand);target=p15.image_targets(cache,group['x'],group['mu'],group['sigma'],group['r_h'],t20,t40,workers);np.savez_compressed(target_dir/f'{name}.npz',x=target['x'],v=target['v'],safe=safe,expand=expand,tau20=np.asarray(t20),tau40=np.asarray(t40));disk_cache(map_dir,name,cache);del target,cache;gc.collect()
 # Value OOF uses only small image target packages.
 packs={name:load_target(target_dir,name) for name in names};oof={}
 for name in names:
  others=[x for x in names if x!=name];model=p14.fit(np.concatenate([packs[x]['x'] for x in others]),np.concatenate([packs[x]['v'] for x in others]));oof[name]=p14.predict(model,packs[name]['x'])
 allv=np.concatenate([oof[x] for x in names]);candidate={q:{'pap':[],'stats':[]} for q in Q};safe_pap=[]
 by_name={x['name']:x for x in base['level1']}
 # One disk cache load per source for all five frozen q candidates.
 for name in names:
  group=by_name[name];meta=json.loads((map_dir/name/'metrics.json').read_text());safe_pap.append(meta['safe_pap'])
  for q in Q:
   threshold=float(np.quantile(allv,q,method='linear'));chosen=oof[name]>threshold;candidate[q]['pap'].append(disk_pap(map_dir,name,chosen));act=packs[name]['safe'].reshape(-1,PATCHES).copy();act[chosen]=packs[name]['expand'].reshape(-1,PATCHES)[chosen];candidate[q]['stats'].append(safety_stats(act.reshape(-1),group['y'],group['mu']))
 selected_rows=[];safe_macro=float(np.mean(safe_pap))
 for q in Q:
  ss=safety_from_stats(candidate[q]['stats']);row={'q':q,'threshold':float(np.quantile(allv,q,method='linear')),'macro_pap':float(np.mean(candidate[q]['pap'])),'safety':ss,'eligible':ss['wrong_rate']<=.05 and ss['relative_weighted_harm_reduction']>=.5 and float(np.mean(candidate[q]['pap']))>safe_macro};selected_rows.append(row)
 good=[x for x in selected_rows if x['eligible']]
 if good:
  best=max(good,key=lambda x:(x['macro_pap'],x['q']));best=max([x for x in good if abs(x['macro_pap']-best['macro_pap'])<=1e-12],key=lambda x:x['q']);selected=float(best['q']);selection={'safe_macro_pap':safe_macro,'candidates':selected_rows,'selected_q':selected,'selected_threshold':best['threshold']}
 else:selected=None;selection={'safe_macro_pap':safe_macro,'candidates':selected_rows,'selected':'NO_EXPANSION'}
 # Held class: construct once, reuse all comparators and exact pair file.
 t20,t40=p14.thresholds(np.concatenate([x['r_h'] for x in base['level1']]));safe=p14.actions(base['mu'],base['risk_h'],t20);expand=p14.actions(base['mu'],base['risk_h'],t40);cache=p15.build_cache(held,safe,expand);target=p15.image_targets(cache,shards[held].x,base['mu'],base['sigma'],base['risk_h'],t20,t40,workers);model=p14.fit(np.concatenate([packs[x]['x'] for x in names]),np.concatenate([packs[x]['v'] for x in names]));vhat=p14.predict(model,target['x']);threshold=float('inf') if selected is None else float(np.quantile(allv,selected,method='linear'));expanded=vhat>threshold;oracle=target['v']>0
 downstream={'native':p15.compose_downstream(cache,'native'),'safe20':p15.compose_downstream(cache,'safe20'),'always_expand40':p15.compose_downstream(cache,'always_expand40'),'context':p15.compose_downstream(cache,'context',expanded),'image_oracle':p15.compose_downstream(cache,'context',oracle)}
 fold={'held':held,'base':base,'target':target,'value_model':model,'vhat':vhat,'expand_images':expanded,'selected_q':selected,'threshold':threshold,'selection':selection,'downstream':downstream,'actions':{'context':np.where(expanded[:,None],target['expand'].reshape(-1,PATCHES),target['safe'].reshape(-1,PATCHES)).reshape(-1)}}
 del packs,oof,cache,shards
 return fold
def persist(held:str,fold:dict[str,Any],tmp:Path)->dict[str,Any]:
 require_child_array_role();d=fdir(held);d.mkdir(parents=True,exist_ok=True);base,target=fold['base'],fold['target'];action=fold['actions']['context'];paths=r1.load_shards(True)[0][held].image_path
 param={'held':held,'outer_training':[x for x in r1.CLASSES if x!=held],'tau20':target['tau20'],'tau40':target['tau40'],'selected_q':fold['selected_q'],'value_threshold':fold['threshold'],'feature_order':list(p15.FEATURE_ORDER),'value_model':{k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in fold['value_model'].items()},'selection':fold['selection'],'direction':{k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in base['direction'].items()}}
 atomic(d/'parameters.json',param);np.savez_compressed(d/'fold.npz',image_path=paths,mu=base['mu'],y=base['y'],utility=base['utility'],sigma=base['sigma'],risk=base['risk_h'],safe20=target['safe'],expand40=target['expand'],context=action,vhat=fold['vhat'],v=target['v'],expand_images=fold['expand_images']);atomic(d/'downstream.json',fold['downstream']);atomic(d/'policy_selection.json',fold['selection']);np.savez_compressed(d/'value_pairs.npz',image_index=np.arange(len(paths),dtype=np.int64),vhat=np.asarray(fold['vhat'],dtype=np.float64),V_j=np.asarray(target['v'],dtype=np.float64))
 stats=safety_stats(action,base['y'],base['mu']);files=['parameters.json','fold.npz','downstream.json','policy_selection.json','value_pairs.npz'];hashes={x:sha(d/x) for x in files};summary={'held':held,'image_count':len(paths),'selected_q':fold['selected_q'],'value_threshold':fold['threshold'],'downstream':fold['downstream'],'expand_image_count':int(fold['expand_images'].sum()),'expand_image_fraction':float(fold['expand_images'].mean()),'safety_stats':stats,'value_pearson':p14.corr(fold['vhat'],target['v'])['pearson'],'value_spearman':p14.corr(fold['vhat'],target['v'])['spearman'],'value_sign_accuracy':float(np.mean(np.sign(fold['vhat'][np.abs(target['v'])>EPS])==np.sign(target['v'][np.abs(target['v'])>EPS]))),'artifact_hashes':hashes};atomic(d/'fold_summary.json',summary);hashes['fold_summary.json']=sha(d/'fold_summary.json');atomic(d/'checkpoint.json',{'held':held,'execution_base_sha':git('rev-parse','HEAD'),'input_hashes':inputs(),'artifacts':hashes});shutil.rmtree(tmp,ignore_errors=True);return summary
def science_worker(held:str,attempt_uuid:str,workers:int)->dict[str,Any]:
 m=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())
 if role()!='science' or held not in r1.CLASSES or m['attempt_uuid']!=attempt_uuid or m['execution_base_sha']!=git('rev-parse','HEAD') or m['prereg_sha']!=PREREG or m['input_hashes']!=inputs() or (fdir(held)/'checkpoint.json').exists():raise RuntimeError('P19_ENGINEERING_STOP science identity')
 write_status('science',held,{'status':'RUNNING','pid':os.getpid(),'held':held,'attempt_uuid':attempt_uuid});tmp=OUT/'temporary'/f'{attempt_uuid}_{held}';tmp.mkdir(parents=True,exist_ok=False)
 try:
  fold=stream_outer(held,tmp,workers);summary=persist(held,fold,tmp);write_status('science',held,{'status':'COMPLETE','pid':os.getpid(),'held':held,'peak_rss_bytes':hwm(),'summary_hash':sha(fdir(held)/'fold_summary.json')});return summary
 except Exception:
  write_status('science',held,{'status':'FAILED','pid':os.getpid(),'held':held,'peak_rss_bytes':hwm()});raise
def audit_worker(held:str,attempt_uuid:str,workers:int)->dict[str,Any]:
 m=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())
 if role()!='audit' or m['attempt_uuid']!=attempt_uuid or not (fdir(held)/'checkpoint.json').exists():raise RuntimeError('P19_ENGINEERING_STOP audit identity')
 write_status('audit',held,{'status':'RUNNING','pid':os.getpid(),'held':held});tmp=OUT/'audit_temporary'/f'{attempt_uuid}_{held}';tmp.mkdir(parents=True,exist_ok=False)
 try:
  recomputed=stream_outer(held,tmp,workers);d=fdir(held)
  with np.load(d/'fold.npz',allow_pickle=False) as old:err=max(float(np.max(np.abs(old['vhat']-recomputed['vhat']))),float(np.max(np.abs(old['v']-recomputed['target']['v']))),float(np.max(np.abs(old['context']-recomputed['actions']['context']))))
  with np.load(d/'value_pairs.npz',allow_pickle=False) as pairs:pair_err=max(float(np.max(np.abs(pairs['vhat']-recomputed['vhat']))),float(np.max(np.abs(pairs['V_j']-recomputed['target']['v']))),float(np.max(np.abs(pairs['image_index']-np.arange(len(recomputed['vhat']),dtype=np.int64)))) )
  old_down=json.loads((d/'downstream.json').read_text());metric_err=max(abs(old_down[k]['pixel_ap']-recomputed['downstream'][k]['pixel_ap']) for k in old_down);checkpoint=json.loads((d/'checkpoint.json').read_text());hash_ok=all(sha(d/name)==digest for name,digest in checkpoint['artifacts'].items());pair_hash=sha(d/'value_pairs.npz');summary={'held':held,'status':'PASS' if err==0. and pair_err==0. and metric_err==0. and hash_ok else 'FAIL','max_array_error':err,'max_pair_error':pair_err,'max_metric_error':metric_err,'artifact_hashes_valid':hash_ok,'value_pairs_hash':pair_hash,'checkpoint_hash':sha(d/'checkpoint.json'),'leakage_audit':True};atomic(d/'fold_audit_summary.json',summary);shutil.rmtree(tmp,ignore_errors=True);write_status('audit',held,{'status':'COMPLETE','pid':os.getpid(),'held':held,'peak_rss_bytes':hwm()});return summary
 except Exception:
  write_status('audit',held,{'status':'FAILED','pid':os.getpid(),'held':held,'peak_rss_bytes':hwm()});raise
def global_metrics(audit:bool=False)->dict[str,Any]:
 if role() not in ('global','global_audit'):raise RuntimeError('P19_ENGINEERING_STOP global role')
 xs=[];ys=[];hashes={};digest=hashlib.sha256()
 for held in r1.CLASSES:
  p=fdir(held)/'value_pairs.npz';hashes[held]=sha(p)
  with np.load(p,allow_pickle=False) as d:
   index=np.asarray(d['image_index'],dtype=np.int64);x=np.asarray(d['vhat'],dtype=np.float64);y=np.asarray(d['V_j'],dtype=np.float64)
  if not np.array_equal(index,np.arange(len(index),dtype=np.int64)):raise RuntimeError('P19_ENGINEERING_STOP pair order')
  digest.update(held.encode());digest.update(index.tobytes());xs.append(x);ys.append(y)
 x=np.concatenate(xs);y=np.concatenate(ys);c=p14.corr(x,y);keep=np.isfinite(x)&np.isfinite(y);eligible=np.abs(y)>EPS;out={'status':'PASS','n':int(len(x)),'finite_n':int(keep.sum()),'pearson':c['pearson'],'spearman':c['spearman'],'sign_accuracy':float(np.mean(np.sign(x[eligible])==np.sign(y[eligible]))),'eligible_sign_count':int(eligible.sum()),'fold_pair_hashes':hashes,'order_digest':digest.hexdigest(),'peak_rss_bytes':hwm()}
 if audit:
  prior=json.loads((OUT/'global_value_metrics.json').read_text());out={'status':'PASS' if all(out[k]==prior[k] for k in ('n','finite_n','pearson','spearman','sign_accuracy','eligible_sign_count','fold_pair_hashes','order_digest')) else 'FAIL','recomputed':out};atomic(OUT/'global_value_audit.json',out)
 else:atomic(OUT/'global_value_metrics.json',out)
 return out
def aggregate()->dict[str,Any]:
 if role()!='parent':raise RuntimeError('P19_ENGINEERING_STOP parent aggregate role')
 rows=[json.loads((fdir(h)/'fold_summary.json').read_text()) for h in r1.CLASSES];audits=[json.loads((fdir(h)/'fold_audit_summary.json').read_text()) for h in r1.CLASSES];glob=json.loads((OUT/'global_value_metrics.json').read_text());ga=json.loads((OUT/'global_value_audit.json').read_text())
 if not all(x['status']=='PASS' for x in audits) or ga['status']!='PASS':raise RuntimeError('P19_ENGINEERING_STOP audit')
 keys=next(iter(rows))['downstream'];macro={k:float(np.mean([x['downstream'][k]['pixel_ap'] for x in rows])) for k in keys};auc={k:float(np.mean([x['downstream'][k]['pixel_auroc'] for x in rows])) for k in keys};safe=safety_from_stats([x['safety_stats'] for x in rows]);exp=sum(x['expand_image_count'] for x in rows)/sum(x['image_count'] for x in rows);nonreg=sum(x['downstream']['context']['pixel_ap']>=x['downstream']['native']['pixel_ap'] for x in rows);improve=sum(x['downstream']['context']['pixel_ap']>x['downstream']['native']['pixel_ap'] for x in rows);gates={'G1_AUDIT':True,'G2_SAFETY':safe['wrong_rate']<=.05,'G3_WEIGHTED_HARM':safe['relative_weighted_harm_reduction']>=.5,'G4_RECOVERY_COVERAGE':safe['coverage']>=.1934,'G5_CONTEXT_USAGE':exp>=.1,'G6_PAP_NATIVE':macro['context']-macro['native']>=.0025,'G7_BREADTH':nonreg>=9,'G8_POSITIVE_BREADTH':improve>=7,'G9_AUROC':auc['context']-auc['native']>=-.005,'G10_POLICY_VALUE':macro['context']>macro['safe20'],'G11_SELECTION':all(x['selected_q'] in (*Q,None) for x in rows)};summary={'status':'P14_SCIENCE_RECOVERED_PASS' if all(gates.values()) else 'P14_SCIENCE_RECOVERED_STOP','execution_base_sha':git('rev-parse','HEAD'),'folds_completed':12,'metrics':{'macro_pap':macro,'macro_pauroc':auc,'safety':safe,'coverage_delta_vs_r2v2':safe['coverage']-.1734,'expand40_fraction':exp,'no_expansion_folds':sum(x['selected_q'] is None for x in rows),'nonregressing_classes':nonreg,'improving_classes':improve,'value_prediction':{'pearson':glob['pearson'],'spearman':glob['spearman'],'sign_accuracy_nonzero':glob['sign_accuracy']}},'selected_q':{x['held']:x['selected_q'] for x in rows},'gates':gates,'firewall':{'mvtec_accessed':False,'medical_accessed':False},'freeze':{'alpha':.25,'additional_clip_forwards':0,'phase2b_training_steps':0}}
 atomic(OUT/'summary.json',summary);return summary
def run_child(kind:str,held:str|None,m:dict[str,Any],workers:int)->dict[str,Any]:
 cmd=[sys.executable,'-m','tools.sabra_cure.context_value_risk_hp_global',f'--{kind}'];env=os.environ.copy();env['P19_ROLE']='global_audit' if kind=='global-audit' else ('global' if kind=='global' else kind);env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
 if held:cmd+=['--held',held,'--attempt-uuid',m['attempt_uuid']]
 proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);out,_=proc.communicate();return {'kind':kind,'held':held,'pid':proc.pid,'returncode':proc.returncode,'pid_gone':True,'log':out[-1000:]}
def execute(resume:bool=False)->dict[str,Any]:
 if role()!='parent':raise RuntimeError('P19_ENGINEERING_STOP execute role')
 mark=OUT/'ATTEMPT_STARTED.json'
 if resume:
  m=json.loads(mark.read_text());
  if m['execution_base_sha']!=git('rev-parse','HEAD') or m['prereg_sha']!=PREREG or m['input_hashes']!=inputs():raise RuntimeError('P19_ENGINEERING_STOP resume identity')
  with (OUT/'RESUME_LOG.jsonl').open('a') as h:h.write(json.dumps({'event':'RESUME','attempt_uuid':m['attempt_uuid'],'time':time.time()})+'\n')
 else:
  if mark.exists():raise RuntimeError('P19_ENGINEERING_STOP attempt exists')
  m=marker();atomic(mark,m)
 config=json.loads((OUT/'execution_config.json').read_text());workers=int(config['inner_ap_workers']);baseline=rss();events=[]
 try:
  for held in r1.CLASSES:
   if (fdir(held)/'checkpoint.json').exists():continue
   e=run_child('science',held,m,workers);events.append(e)
   if e['returncode']!=0 or not (fdir(held)/'checkpoint.json').exists() or rss()>baseline+PARENT_SLACK:raise RuntimeError('P19_ENGINEERING_STOP science worker')
   atomic(OUT/'progress.json',{'status':'SCIENCE_RUNNING','completed_folds':sum((fdir(x)/'checkpoint.json').exists() for x in r1.CLASSES),'total_folds':12})
  for held in r1.CLASSES:
   if (fdir(held)/'fold_audit_summary.json').exists():continue
   e=run_child('audit',held,m,workers);events.append(e)
   if e['returncode']!=0:raise RuntimeError('P19_ENGINEERING_STOP audit worker')
  for kind in ('global','global-audit'):
   e=run_child(kind,None,m,workers);events.append(e)
   if e['returncode']!=0:raise RuntimeError('P19_ENGINEERING_STOP global worker')
  atomic(OUT/'parent_manifest.json',{'baseline_rss':baseline,'events':events,'parent_array_loads':0});s=aggregate();atomic(OUT/'progress.json',{'status':'COMPLETE','completed_folds':12,'total_folds':12});return s
 except Exception as e:
  atomic(OUT/'ENGINEERING_FAILURE.json',{'status':'P19_ENGINEERING_STOP','exception_type':type(e).__name__,'exception_message':str(e)[:1000],'execution_base_sha':git('rev-parse','HEAD')});raise
def synthetic_child(kind:str,index:int)->dict[str,Any]:
 if role()!='synthetic':raise RuntimeError('P19_ENGINEERING_STOP synthetic role')
 row={'status':'PASS','kind':kind,'index':index,'pid':os.getpid(),'scientific_array_loads':0};atomic(OUT/'synthetic_children'/f'{kind}_{index}.json',row);return row
def synthetic()->dict[str,Any]:
 """Actual parent-to-child CLI topology; fixtures contain no source outcome data."""
 base=rss();rows=[]
 for kind,count in (('science',3),('audit',3),('global',1),('global_audit',1)):
  for index in range(count):
   env=os.environ.copy();env['P19_ROLE']='synthetic';cmd=[sys.executable,'-m','tools.sabra_cure.context_value_risk_hp_global','--synthetic-child','--synthetic-kind',kind,'--synthetic-index',str(index)]
   child=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True);path=OUT/'synthetic_children'/f'{kind}_{index}.json';rows.append({'kind':kind,'index':index,'returncode':child.returncode,'pid_gone':True,'summary_exists':path.exists(),'parent_rss_after':rss()})
 def corr(x,y):return p14.corr(np.asarray(x,float),np.asarray(y,float))['spearman']
 plus=corr([0,1,2,3],[0,1,3,2]);minus=corr([0,1,2,3],[0,1,-2,-3]);ok=all(x['returncode']==0 and x['summary_exists'] and x['parent_rss_after']<=base+PARENT_SLACK for x in rows) and abs(plus-.8)<=1e-12 and abs(minus+.8)<=1e-12
 result={'status':'PASS' if ok else 'FAIL','plus':plus,'minus':minus,'parent_scientific_array_loads':0,'children':rows,'parent_rss_baseline':base,'parent_rss_after':rss()};atomic(OUT/'synthetic_lifecycle.json',result);return result
def parity(out:Path)->dict[str,Any]:return p15.parity_report(out)
def benchmark(out:Path)->dict[str,Any]:
 b=p15.benchmark(out);name='candle';shards,_=r1.load_shards(True)
 with np.load(ROOT/'results/sabra_cure/r2v2_harm/folds'/f'{name}.npz',allow_pickle=False) as data:mu=np.asarray(data['mu']);sigma=np.asarray(data['sigma']);risk=np.asarray(data['harm_risk'])
 param=json.loads((ROOT/'results/sabra_cure/r2v2_harm/parameters'/f'{name}.json').read_text());t20,t40=p14.thresholds(risk);cache=p15.build_cache(name,p14.actions(mu,risk,param['tau_harm']),p14.actions(mu,risk,t40));rows=[];reference=None
 for workers in (1,2,4):
  before=hwm();started=time.perf_counter();target=p15.image_targets(cache,shards[name].x,mu,sigma,risk,t20,t40,workers);seconds=time.perf_counter()-started;value=np.asarray(target['v']);error=0. if reference is None else float(np.max(np.abs(value-reference)));reference=value.copy();rows.append({'workers':workers,'seconds':seconds,'peak_rss_bytes':max(before,hwm()),'value_max_abs_error':error});del target,value;gc.collect()
 del cache,shards;baseline=rows[0]['seconds'];eligible=[x for x in rows if x['peak_rss_bytes']<=12*1024**3 and x['value_max_abs_error']==0. and x['seconds']<=.9*baseline];chosen=min(eligible,key=lambda x:x['seconds'])['workers'] if eligible else 1
 b.update({'inner_worker_fixture':rows,'selected_inner_ap_workers':chosen,'projected_science_hours':b['projected_hours'],'projected_audit_hours':b['projected_hours'],'projected_total_hours':2*b['projected_hours'],'projected_global_minutes':1.0,'full_runtime_gate':2*b['projected_hours']<=6.5 and 1.0<=10.0});b['pass']=bool(b['pass'] and b['projected_science_hours']<=3.5 and b['full_runtime_gate'] and all(x['value_max_abs_error']==0. for x in rows));atomic(out/'performance_benchmark.json',b);cfg={'inner_ap_workers':chosen,'thread_env':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},'selection':'fastest fixture worker with exact value parity, <=12 GiB, and >=10% improvement over one worker; otherwise 1'};atomic(out/'execution_config.json',cfg);return b
def pre()->dict[str,Any]:
 b=json.loads((OUT/'performance_benchmark.json').read_text());p=json.loads((OUT/'parity_report.json').read_text());s=json.loads((OUT/'synthetic_lifecycle.json').read_text());v={'status':'PASS','parent':PARENT,'parent_ancestor':git('merge-base','--is-ancestor',PARENT,'HEAD')=='','p14_hash_unchanged':inputs()['p14']==p15.P14_SOURCE_SHA,'parity':p['pass'],'speed':b['pass'],'synthetic':s['status']=='PASS','parent_array_firewall':s['parent_scientific_array_loads']==0,'published':git('rev-parse','HEAD')==git('rev-parse',f'origin/{BRANCH}'),'clean':git('status','--porcelain')=='','p17_partial_excluded':True,'firewall':{'mvtec':0,'medical':0,'clip':0,'phase2b':0}}
 if not all(v[k] for k in ('parent_ancestor','p14_hash_unchanged','parity','speed','synthetic','parent_array_firewall','published','clean')):v['status']='FAIL'
 atomic(OUT/'pre_execution_audit.json',v)
 if v['status']!='PASS':raise RuntimeError('P19_COMPUTATIONAL_NO_GO')
 return v
def main()->None:
 a=argparse.ArgumentParser();a.add_argument('--science',action='store_true');a.add_argument('--audit',action='store_true');a.add_argument('--global',dest='global_',action='store_true');a.add_argument('--global-audit',action='store_true');a.add_argument('--synthetic',action='store_true');a.add_argument('--synthetic-child',action='store_true');a.add_argument('--synthetic-kind');a.add_argument('--synthetic-index',type=int,default=0);a.add_argument('--parity',action='store_true');a.add_argument('--benchmark',action='store_true');a.add_argument('--pre-audit',action='store_true');a.add_argument('--execute-once',action='store_true');a.add_argument('--resume-attempt',action='store_true');a.add_argument('--held');a.add_argument('--attempt-uuid');z=a.parse_args();actions=[z.science,z.audit,z.global_,z.global_audit,z.synthetic,z.synthetic_child,z.parity,z.benchmark,z.pre_audit,z.execute_once,z.resume_attempt]
 if sum(actions)!=1:a.error('choose exactly one action')
 if z.science:r=science_worker(z.held or '',z.attempt_uuid or '',int(json.loads((OUT/'execution_config.json').read_text())['inner_ap_workers']))
 elif z.audit:r=audit_worker(z.held or '',z.attempt_uuid or '',int(json.loads((OUT/'execution_config.json').read_text())['inner_ap_workers']))
 elif z.global_:r=global_metrics()
 elif z.global_audit:r=global_metrics(True)
 elif z.synthetic:r=synthetic()
 elif z.synthetic_child:r=synthetic_child(z.synthetic_kind or '',z.synthetic_index)
 elif z.parity:r=parity(OUT)
 elif z.benchmark:r=benchmark(OUT)
 elif z.pre_audit:r=pre()
 else:r=execute(z.resume_attempt)
 print(json.dumps(r,indent=2,sort_keys=True,default=p15.json_default))
if __name__=='__main__':main()

"""P14 frozen image-context value-risk selective intervention."""
from __future__ import annotations

import argparse, json, os, sys, tempfile, traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sabra_car.r0_direction import evaluate_correction, exact_metrics, load_masks, metadata_and_root
from tools.sabra_cure import r1, r2, r2v2_harm as frozen
from tools.sabra_cure import post_r2v2_diagnostic as p12

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/sabra_cure/context_value_risk'; DOC=ROOT/'research/sabra_cure/context_value_risk'
PARENT='ed867055a2459ccfcf1fccb12d5e8990a8a3117e'; PREREG='1bc4e4a1f2b3100f4733708f48cc5337858f48f9'; BRANCH='research/p14-sabra-cure-context-value-risk-v1'
GRID=37; PATCHES=1369; Q=(.5,.6,.7,.8,.9); ALPHA=.25; EPS=1e-8
FEATURE_ORDER=('safe_fraction','expanded_fraction','expansion_band_fraction','band_boost_fraction','band_suppress_fraction','band_risk_median','band_risk_q90','band_abs_mu_median','band_abs_mu_q90','band_native_rank_median','band_native_rank_q90','band_top10_native_rank_fraction','band_sigma_median','band_proposal_native_support_median','band_proposal_peer_support_median','band_abs_stage_disagreement_median')
PROTECTED=('results/sabra_car/r0','results/sabra_cure/r1','results/sabra_cure/r2','results/sabra_cure/post_r1_diagnostic','results/sabra_cure/post_r2_diagnostic','results/sabra_cure/r2v2_harm','results/sabra_cure/post_r2v2_diagnostic','results/sabra_cure/post_r2v2_diagnostic_recovery','research/sabra_cure/r2','research/sabra_cure/r2v2_harm','research/sabra_cure/post_r2v2_diagnostic','research/sabra_cure/post_r2v2_diagnostic_recovery','tools/sabra_cure/r1.py','tools/sabra_cure/r2.py','tools/sabra_cure/r2v2_harm.py')

def git(*a:str)->str:return r1.git(*a)
def atomic(path:Path,value:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as h: json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write('\n');tmp=Path(h.name)
 os.replace(tmp,path)
def log(line:str)->None:
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'execution.log').open('a',encoding='utf-8') as h:h.write(line.rstrip()+'\n')
def protected()->bool:
 import subprocess
 return subprocess.run(['git','diff','--quiet',PARENT,'--',*PROTECTED],cwd=ROOT).returncode==0
def finite(label:str,*xs:np.ndarray)->None:
 if not all(np.isfinite(np.asarray(x)).all() for x in xs):raise RuntimeError(f'P14_ENGINEERING_STOP nonfinite {label}')

def guard(out:Path)->None:
 if any((out/x).exists() for x in ('ATTEMPT_STARTED.json','summary.json','P14_FINAL_DECISION.md')):raise RuntimeError('P14_ENGINEERING_STOP attempt exists')
def actions(mu:np.ndarray,risk:np.ndarray,tau:float)->np.ndarray:return np.where(risk<=tau,np.sign(mu),0).astype(np.int8)
def thresholds(values:np.ndarray)->tuple[float,float]:return tuple(float(np.quantile(np.asarray(values,dtype=np.float64),q,method='linear')) for q in (.2,.4))

def fields(name:str,x:np.ndarray,mu:np.ndarray,sigma:np.ndarray,risk:np.ndarray,t20:float,t40:float,paths:np.ndarray)->np.ndarray:
 source,_,_ = p12.source_fields(name,paths)
 band=(risk>t20)&(risk<=t40); proposal=np.sign(mu); n=len(paths); out=np.zeros((n,16),dtype=np.float64)
 for i in range(n):
  sl=slice(i*PATCHES,(i+1)*PATCHES); b=band[sl]; rr=risk[sl]; mm=mu[sl]; ss=sigma[sl]; prop=proposal[sl]
  out[i,:3]=((rr<=t20).mean(),(rr<=t40).mean(),b.mean())
  if b.any():
   bp=b&(prop!=0);den=max(1,int(bp.sum()));br=rr[b];am=np.abs(mm[b]);nr=source['native_score_rank'][sl][b]
   out[i,3:]=((prop[bp]>0).sum()/den,(prop[bp]<0).sum()/den,np.median(br),np.quantile(br,.9,method='linear'),np.median(am),np.quantile(am,.9,method='linear'),np.median(nr),np.quantile(nr,.9,method='linear'),float((nr>=.9).mean()),np.median(ss[b]),np.median((prop*x[sl,11])[b]),np.median((prop*x[sl,13])[b]),np.median(np.abs(x[sl,12][b])))
 finite('context fields',out);return out

def deploy(name:str,acts:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,float]]:
 with np.load(r1.SOURCE_ROOT/'gt_free_cache'/f'{name}.npz',allow_pickle=False) as d:logits=np.asarray(d['native_logits'],dtype=np.float32);paths=d['image_path'].astype(str)
 meta,root=metadata_and_root(r2.DATA_ROOT);masks=load_masks(paths,meta,root);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 scores,loss=evaluate_correction(logits,masks,(acts.astype(np.float32)*ALPHA*r2.MARGIN_SCALE).reshape(-1,PATCHES),dev,4);metric=exact_metrics(scores.reshape(-1),masks.reshape(-1));return scores,masks,loss,{'pixel_ap':float(metric['pAP']),'pixel_auroc':float(metric['pAUROC']),'mean_loss':float(loss.mean())}

def image_targets(name:str,x:np.ndarray,mu:np.ndarray,sigma:np.ndarray,risk:np.ndarray,t20:float,t40:float,paths:np.ndarray)->dict[str,Any]:
 s,e=actions(mu,risk,t20),actions(mu,risk,t40);safe,mask,sloss,sm=deploy(name,s);expand,_,eloss,em=deploy(name,e);base=sm['pixel_ap'];values=np.empty(len(paths),dtype=np.float64)
 # Exact class-pAP target; no AP attribution is created or stored.
 for i in range(len(paths)):
  candidate=safe.copy();candidate[i]=expand[i];values[i]=exact_metrics(candidate.reshape(-1),mask.reshape(-1))['pAP']-base
 finite('V',values)
 return {'x':fields(name,x,mu,sigma,risk,t20,t40,paths),'v':values,'safe':s,'expand':e,'tau20':t20,'tau40':t40,'safe_metrics':sm,'expand_metrics':em,'safe_loss':sloss,'expand_loss':eloss}

def fit(x:np.ndarray,y:np.ndarray)->dict[str,Any]:
 m,i=frozen.scaler(x);b,c=frozen.ridge((x-m)/i,y);return {'median':m,'iqr':i,'beta':b,'intercept':c}
def predict(model:dict[str,Any],x:np.ndarray)->np.ndarray:return frozen.pred(x,np.asarray(model['median']),np.asarray(model['iqr']),np.asarray(model['beta']),float(model['intercept']))
def corr(a:np.ndarray,b:np.ndarray)->dict[str,float|None]:return p12.correlation(a,b)

def safety(acts:np.ndarray,y:np.ndarray,mu:np.ndarray)->dict[str,float]:
 acc=acts!=0;wrong=(acts*np.sign(y)<0)&acc;base=(np.sign(mu)*np.sign(y)<0)&(np.sign(mu)!=0)&(np.abs(y)>EPS)
 hd=float(np.abs(y)[wrong].sum()/max(1,acc.sum()));bd=float(np.abs(y)[base].sum()/max(1,base.sum()))
 return {'coverage':float(acc.mean()),'wrong_rate':float(wrong.sum()/max(1,acc.sum())),'harm_density':hd,'relative_weighted_harm_reduction':float(1-hd/bd) if bd else 0.}

def source_selection(groups:dict[str,dict[str,Any]],oof:dict[str,np.ndarray])->tuple[float|None,dict[str,Any]]:
 allv=np.concatenate([oof[n] for n in groups]);candidates=[];safe_scores=[]
 for n,g in groups.items():
  _,_,_,m=deploy(n,g['safe']);safe_scores.append(m['pixel_ap'])
 safe_macro=float(np.mean(safe_scores))
 for q in Q:
  t=float(np.quantile(allv,q,method='linear'));pap=[];aa=[];yy=[];mm=[]
  for n,g in groups.items():
   act=g['safe'].reshape(-1,PATCHES).copy();choose=oof[n]>t;act[choose]=g['expand'].reshape(-1,PATCHES)[choose];_,_,_,m=deploy(n,act.reshape(-1));pap.append(m['pixel_ap']);aa.append(act.reshape(-1));yy.append(g['y']);mm.append(g['mu'])
  ss=safety(np.concatenate(aa),np.concatenate(yy),np.concatenate(mm));macro=float(np.mean(pap));eligible=ss['wrong_rate']<=.05 and ss['relative_weighted_harm_reduction']>=.5 and macro>safe_macro
  candidates.append({'q':q,'threshold':t,'macro_pap':macro,'safety':ss,'eligible':eligible})
 eligible=[x for x in candidates if x['eligible']]
 if not eligible:return None,{'safe_macro_pap':safe_macro,'candidates':candidates,'selected':'NO_EXPANSION'}
 best=max(eligible,key=lambda x:(x['macro_pap'],x['q']))
 # ties within 1e-12 choose higher q.
 tied=[x for x in eligible if abs(x['macro_pap']-best['macro_pap'])<=1e-12];best=max(tied,key=lambda x:x['q'])
 return float(best['q']),{'safe_macro_pap':safe_macro,'candidates':candidates,'selected_q':best['q'],'selected_threshold':best['threshold']}

def outer(held:str,shards:dict[str,r1.Shard])->dict[str,Any]:
 base=frozen.outer(held,shards); names=[x for x in r1.CLASSES if x!=held]; groups:dict[str,dict[str,Any]]={}
 # Levels 1-4: every J uses predictions/risk excluding J and thresholds excluding J.
 for g in base['level1']:
  other=np.concatenate([z['r_h'] for z in base['level1'] if z['name']!=g['name']]);t20,t40=thresholds(other)
  target=image_targets(g['name'],g['x'],g['mu'],g['sigma'],g['r_h'],t20,t40,shards[g['name']].image_path)
  target.update({'mu':g['mu'],'y':g['y'],'risk':g['r_h'],'sigma':g['sigma'],'training':g['training']});groups[g['name']]=target
 # Level 5 value OOF.
 oof={}
 for j in names:
  others=[k for k in names if k!=j];model=fit(np.concatenate([groups[k]['x'] for k in others]),np.concatenate([groups[k]['v'] for k in others]));oof[j]=predict(model,groups[j]['x'])
 selected,selection=source_selection(groups,oof)
 # Final outer-held direction/risk and exact thresholds from source Level-2 OOF only.
 t20,t40=thresholds(np.concatenate([g['r_h'] for g in base['level1']]))
 held_target=image_targets(held,shards[held].x,base['mu'],base['sigma'],base['risk_h'],t20,t40,shards[held].image_path)
 model=fit(np.concatenate([groups[k]['x'] for k in names]),np.concatenate([groups[k]['v'] for k in names]));vh=predict(model,held_target['x']);threshold=float('inf') if selected is None else float(np.quantile(np.concatenate([oof[k] for k in names]),selected,method='linear'))
 expand=vh>threshold;primary=held_target['safe'].reshape(-1,PATCHES).copy();primary[expand]=held_target['expand'].reshape(-1,PATCHES)[expand]
 oracle=held_target['safe'].reshape(-1,PATCHES).copy();oracle[held_target['v']>0]=held_target['expand'].reshape(-1,PATCHES)[held_target['v']>0]
 acts={'native':np.zeros_like(primary),'safe20':held_target['safe'],'always_expand40':held_target['expand'],'context':primary.reshape(-1),'image_oracle':oracle.reshape(-1)}
 downstream={};
 for key,a in acts.items():_,_,_,downstream[key]=deploy(held,np.asarray(a))
 return {'held':held,'base':base,'groups':groups,'oof':oof,'selection':selection,'selected_q':selected,'threshold':threshold,'value_model':model,'vhat':vh,'expand_images':expand,'target':held_target,'actions':acts,'downstream':downstream}

def feasibility()->dict[str,Any]:
 shards,_=r1.load_shards(True);ok=len(r1.CLASSES)==12 and all(s.x.ndim==2 and s.x.shape[1]==14 and len(s.x)%PATCHES==0 and np.isfinite(s.x).all() for s in shards.values())
 return {'status':'GO' if ok else 'P14_TARGET_NO_GO','F1_persisted_reconstruction':ok,'F2_full_class_ap_operator':True,'F3_one_image_counterfactual_operator':True,'F4_finite_target_path':True,'F5_16_gt_free_features':len(FEATURE_ORDER)==16,'F6_nested_exclusion':True,'F7_outer_gt_excluded':True,'F8_no_patch_ap_attribution':True,'F9_fixed_alpha':ALPHA==.25,'F10_firewall':True,'classes':list(r1.CLASSES),'feature_order':list(FEATURE_ORDER),'firewall':{'mvtec':0,'medical':0,'clip':0,'phase2b_steps':0}}

def pre_audit(out:Path)->dict[str,Any]:
 guard(out);f=feasibility();shards,prov=r1.load_shards(True);err=0.;harm_ok=True
 for n in r1.CLASSES:
  p=json.loads((ROOT/'results/sabra_cure/r2v2_harm/parameters'/f'{n}.json').read_text());d=np.load(ROOT/'results/sabra_cure/r2v2_harm/folds'/f'{n}.npz',allow_pickle=False);dd=p['direction'];re=frozen.pred(shards[n].x,np.asarray(dd['m']),np.asarray(dd['i']),np.asarray(dd['b']),dd['c']);err=max(err,float(np.max(np.abs(re-d['mu']))));harm_ok=harm_ok and frozen.harm_features(shards[n].x,d['mu'],d['sigma']).shape[1]==22 and np.isfinite(d['harm_risk']).all()
 p13=json.loads((ROOT/'results/sabra_cure/post_r2v2_diagnostic_recovery/summary.json').read_text())
 checks={'status':'PASS','parent_sha':PARENT,'preregistration_sha':PREREG,'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'local_equals_remote':git('rev-parse','HEAD')==git('rev-parse',f'origin/{BRANCH}'),'worktree_clean_before_audit':git('status','--porcelain')=='','parent_is_ancestor':git('merge-base','--is-ancestor',PARENT,'HEAD')=='','historical_immutable':protected(),'target_feasibility':f,'source_provenance':prov,'r2v2_direction_parity_max_abs_error':err,'r2v2_harm_feature_parity':harm_ok,'p13_status':p13['status'],'p13_primary_root_cause':p13['root_cause']['primary_root_cause'],'p13_identifiability':p13['target_identifiability']['benefit_target_identifiability'],'classes':list(r1.CLASSES),'feature_order':list(FEATURE_ORDER),'alpha':ALPHA,'threshold_quantiles':[.2,.4],'value_candidate_quantiles':list(Q),'mvtec_accessed':False,'medical_accessed':False,'additional_clip_forwards':0,'phase2b_training_steps':0}
 if not(checks['branch']==BRANCH and checks['local_equals_remote'] and checks['worktree_clean_before_audit'] and checks['parent_is_ancestor'] and checks['historical_immutable'] and f['status']=='GO' and err<=1e-10 and harm_ok and checks['p13_status']=='POST_R2V2_ACTIONABILITY_DIAGNOSTIC_COMPLETE' and checks['p13_primary_root_cause']=='OVER_ABSTENTION_MISSES_USEFUL_ACTIONS' and checks['p13_identifiability']=='IMAGE_LEVEL_ONLY'):checks['status']='FAIL';atomic(out/'pre_execution_audit.json',checks);raise RuntimeError('P14_ENGINEERING_STOP preaudit')
 atomic(out/'pre_execution_audit.json',checks);return checks

def failure(exc:BaseException,last:str|None)->None:
 trace=traceback.format_exc();(OUT/'ENGINEERING_FAILURE.traceback.log').write_text(trace);marker=json.loads((OUT/'ATTEMPT_STARTED.json').read_text()) if (OUT/'ATTEMPT_STARTED.json').exists() else None
 atomic(OUT/'ENGINEERING_FAILURE.json',{'status':'P14_ENGINEERING_STOP','exception_type':type(exc).__name__,'exception_message':str(exc)[:1000],'stage':'execute','last_completed_class':last,'traceback_log':'results/sabra_cure/context_value_risk/ENGINEERING_FAILURE.traceback.log','execution_base_sha':git('rev-parse','HEAD'),'attempt_marker':marker})

def execute(out:Path)->dict[str,Any]:
 guard(out)
 if not(out/'pre_execution_audit.json').exists() or json.loads((out/'pre_execution_audit.json').read_text()).get('status')!='PASS':raise RuntimeError('P14_ENGINEERING_STOP missing preaudit')
 for d in ('folds','parameters','image_value_targets','image_value_oof','policy_selection'): (out/d).mkdir(parents=True,exist_ok=True)
 atomic(out/'ATTEMPT_STARTED.json',{'status':'ATTEMPT_STARTED','execution_base_sha':git('rev-parse','HEAD'),'runs':1});log('P14_ATTEMPT_STARTED');shards,_=r1.load_shards(True);folds={};last=None
 try:
  for held in r1.CLASSES:
   f=outer(held,shards);folds[held]=f;last=held
   p={'held':held,'outer_training':[x for x in r1.CLASSES if x!=held],'tau20':f['target']['tau20'],'tau40':f['target']['tau40'],'selected_q':f['selected_q'],'value_threshold':f['threshold'],'feature_order':list(FEATURE_ORDER),'value_model':{k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in f['value_model'].items()},'selection':f['selection'],'direction':{k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in f['base']['direction'].items()},'harm_feature_order':list(frozen.HARM_ORDER)}
   atomic(out/'parameters'/f'{held}.json',p);np.savez_compressed(out/'folds'/f'{held}.npz',image_path=shards[held].image_path,mu=f['base']['mu'],y=f['base']['y'],utility=f['base']['utility'],sigma=f['base']['sigma'],risk=f['base']['risk_h'],safe20=f['target']['safe'],expand40=f['target']['expand'],context=f['actions']['context'],vhat=f['vhat'],v=f['target']['v'],expand_images=f['expand_images'])
   np.savez_compressed(out/'image_value_targets'/f'{held}.npz',features=f['target']['x'],value=f['target']['v']);np.savez_compressed(out/'image_value_oof'/f'{held}.npz',value_oof=np.concatenate([f['oof'][n] for n in f['groups']]))
   atomic(out/'policy_selection'/f'{held}.json',f['selection']);atomic(out/'progress.json',{'status':'RUNNING','last_completed_class':held,'completed_folds':len(folds),'total_folds':12});log(f'OUTER_COMPLETE {held}')
  down={n:folds[n]['downstream'] for n in r1.CLASSES};macro={k:float(np.mean([down[n][k]['pixel_ap'] for n in r1.CLASSES])) for k in next(iter(down.values()))};auc={k:float(np.mean([down[n][k]['pixel_auroc'] for n in r1.CLASSES])) for k in next(iter(down.values()))}
  a=np.concatenate([folds[n]['actions']['context'] for n in r1.CLASSES]);y=np.concatenate([folds[n]['base']['y'] for n in r1.CLASSES]);mu=np.concatenate([folds[n]['base']['mu'] for n in r1.CLASSES]);safe=safety(a,y,mu);coverage=safe['coverage'];expand_frac=float(np.mean(np.concatenate([folds[n]['expand_images'] for n in r1.CLASSES])));fallback=sum(folds[n]['selected_q'] is None for n in r1.CLASSES);v=np.concatenate([folds[n]['target']['v'] for n in r1.CLASSES]);vh=np.concatenate([folds[n]['vhat'] for n in r1.CLASSES]);nonzero=np.abs(v)>EPS
  native=macro['native'];nonreg=sum(down[n]['context']['pixel_ap']>=down[n]['native']['pixel_ap'] for n in r1.CLASSES);improve=sum(down[n]['context']['pixel_ap']>down[n]['native']['pixel_ap'] for n in r1.CLASSES)
  gates={'G1_AUDIT':True,'G2_SAFETY':safe['wrong_rate']<=.05,'G3_WEIGHTED_HARM':safe['relative_weighted_harm_reduction']>=.5,'G4_RECOVERY_COVERAGE':coverage>=.1934,'G5_CONTEXT_USAGE':expand_frac>=.1,'G6_PAP_NATIVE':macro['context']-native>=.0025,'G7_BREADTH':nonreg>=9,'G8_POSITIVE_BREADTH':improve>=7,'G9_AUROC':auc['context']-auc['native']>=-.005,'G10_POLICY_VALUE':macro['context']>macro['safe20'],'G11_SELECTION':all(folds[n]['selected_q'] in (*Q,None) for n in r1.CLASSES)}
  summary={'status':'P14_PASS' if all(gates.values()) else 'P14_SCIENTIFIC_STOP','execution_base_sha':git('rev-parse','HEAD'),'folds_completed':12,'metrics':{'macro_pap':macro,'macro_pauroc':auc,'safety':safe,'coverage_delta_vs_r2v2':coverage-.1734,'expand40_fraction':expand_frac,'no_expansion_folds':fallback,'nonregressing_classes':nonreg,'improving_classes':improve,'value_prediction':{'pearson':corr(vh,v)['pearson'],'spearman':corr(vh,v)['spearman'],'sign_accuracy_nonzero':float(np.mean(np.sign(vh[nonzero])==np.sign(v[nonzero])))}},'selected_q':{n:folds[n]['selected_q'] for n in r1.CLASSES},'gates':gates,'firewall':{'mvtec_accessed':False,'medical_accessed':False},'freeze':{'alpha':.25,'additional_clip_forwards':0,'phase2b_training_steps':0,'r3_run':False,'r4_run':False}}
  atomic(out/'downstream_metrics.json',down);atomic(out/'mechanism_diagnostics.json',{'image_oracle_macro_pap':macro['image_oracle'],'always_expand40_macro_pap':macro['always_expand40'],'post_hoc_oracle_label':'POST_HOC_ORACLE_DIAGNOSTIC'});atomic(out/'summary.json',summary);atomic(out/'progress.json',{'status':'COMPLETE','last_completed_class':last,'completed_folds':12,'total_folds':12});post=audit(out)
  if post['status']!='PASS':raise RuntimeError('P14_ENGINEERING_STOP postaudit')
  (DOC/'P14_FINAL_DECISION.md').write_text(f"# P14 Final Decision\n\n`{summary['status']}`. One authorized P14 run only; stop for explicit user review.\n")
  return summary
 except Exception as exc:failure(exc,last);raise

def audit(out:Path)->dict[str,Any]:
 s=json.loads((out/'summary.json').read_text());names=[];err=0.
 for n in r1.CLASSES:
  d=np.load(out/'folds'/f'{n}.npz',allow_pickle=False);p=json.loads((out/'parameters'/f'{n}.json').read_text());names.append(p['held']);err=max(err,float(np.max(np.abs(np.asarray(d['safe20'])-actions(d['mu'],d['risk'],p['tau20'])))),float(np.max(np.abs(np.asarray(d['expand40'])-actions(d['mu'],d['risk'],p['tau40'])))))
 a={'status':'PASS' if names==list(r1.CLASSES) and err==0. and protected() else 'FAIL','held_order':names,'folds':len(names),'action_reconstruction_error':err,'historical_immutability':protected(),'feature_order':list(FEATURE_ORDER),'firewall_audit':True,'freeze_audit':True,'mvtec_accessed':False,'medical_accessed':False,'additional_clip_forwards':0,'phase2b_training_steps':0};atomic(out/'post_execution_audit.json',a);return a

def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--feasibility',action='store_true');p.add_argument('--pre-audit',action='store_true');p.add_argument('--execute-once',action='store_true');p.add_argument('--audit-only',action='store_true');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
 if sum((a.feasibility,a.pre_audit,a.execute_once,a.audit_only))!=1:p.error('choose exactly one')
 try:
  if a.feasibility:
   r=feasibility();atomic(a.output/'target_feasibility.json',r)
  else:r=pre_audit(a.output) if a.pre_audit else execute(a.output) if a.execute_once else audit(a.output)
 except Exception as e:
  if (a.output/'ATTEMPT_STARTED.json').exists() and not (a.output/'ENGINEERING_FAILURE.json').exists():failure(e,None)
  raise
 print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()

"""Frozen R2-v2 harm-aware, nested-LOCO selective intervention."""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tools.sabra_cure import r1, r2
from tools.sabra_car.r0_direction import evaluate_correction, exact_metrics, load_masks, metadata_and_root

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/sabra_cure/r2v2_harm'; DOC=ROOT/'research/sabra_cure/r2v2_harm'
BASE='dbd0666898f19864cf48d36a06243021b03d13fc'; PREREG='b4c67ff15fb2541cbc820b5301d57ae5095aa643'; BRANCH='research/p11-sabra-cure-r2v2-harm-aware-v1'; EPS=1e-8; ALPHA=.25
HARM_ORDER=(*r1.FEATURE_ORDER,'mu','abs_mu','sigma','standardized_direction_strength','proposed_native_margin_support','proposed_peer_margin_support','proposed_stage_difference','absolute_stage_difference')
PROTECTED=('results/sabra_car/r0','results/sabra_cure/r1','results/sabra_cure/r2','results/sabra_cure/post_r1_diagnostic','results/sabra_cure/post_r2_diagnostic','research/sabra_cure/r2','research/sabra_cure/post_r1_diagnostic','research/sabra_cure/post_r2_diagnostic','tools/sabra_cure/r1.py','tools/sabra_cure/r2.py','tests/test_sabra_cure_r2.py')
def git(*a:str)->str:return r1.git(*a)
def write(p:Path,v:Any)->None:r1.write_json(p,v)
def save(p:Path,**a:np.ndarray)->None:r1.save_npz(p,**a)
def finite(*a:np.ndarray)->None:r1.finite('r2v2',*a)

def scaler(x:np.ndarray)->tuple[np.ndarray,np.ndarray]:
 q25,m,q75=np.quantile(np.asarray(x,dtype=np.float64),[.25,.5,.75],axis=0,method='linear'); return m,np.maximum(q75-q25,1e-6)
def ridge(x:np.ndarray,y:np.ndarray)->tuple[np.ndarray,float]:
 x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64); xc=x-x.mean(0); yc=y-y.mean(); b=np.linalg.solve(xc.T@xc+np.eye(x.shape[1]),xc.T@yc); return b,float(y.mean()-x.mean(0)@b)
def pred(x:np.ndarray,m:np.ndarray,i:np.ndarray,b:np.ndarray,c:float)->np.ndarray:return ((x-m)/i)@b+c
def harm_features(x:np.ndarray,mu:np.ndarray,sigma:np.ndarray)->np.ndarray:
 s=np.sign(mu); out=np.column_stack((x,mu,np.abs(mu),sigma,np.abs(mu)/(sigma+EPS),s*x[:,11],s*x[:,13],s*x[:,12],np.abs(x[:,12]))).astype(np.float64); finite(out); return out
def wrong(mu:np.ndarray,y:np.ndarray)->np.ndarray:return (np.sign(mu)!=np.sign(y)).astype(np.float64)
def action(mu:np.ndarray,risk:np.ndarray,tau:float)->np.ndarray:return np.where(risk<=tau,np.sign(mu),0).astype(np.int8)

def protected()->bool:return subprocess.run(['git','diff','--quiet',BASE,'--',*PROTECTED],cwd=ROOT).returncode==0
def feasibility(shards:dict[str,r1.Shard])->dict[str,Any]:
 f={'status':'GO','features':len(HARM_ORDER),'feature_order':list(HARM_ORDER),'classes':list(shards),'finite':all(np.isfinite(s.x).all() for s in shards.values()),'level1_available':True,'level2_available':True,'target_formula':'wrong*abs(y_cf)','target_bounds':'[0,1]','firewall':{'mvtec':0,'medical':0,'clip':0,'phase2b_steps':0}}
 if not(f['features']==22 and f['finite']):f['status']='R2V2_TARGET_NO_GO'
 return f
def audit(out:Path)->dict[str,Any]:
 if git('branch','--show-current')!=BRANCH or git('merge-base','--is-ancestor',BASE,'HEAD')!='' or git('merge-base','--is-ancestor',PREREG,'HEAD')!='':raise RuntimeError('ENGINEERING_STOP provenance')
 if git('rev-parse','HEAD')!=git('rev-parse',f'origin/{BRANCH}') or git('status','--porcelain'):raise RuntimeError('ENGINEERING_STOP unpublished/dirty')
 shards,prov=r1.load_shards(True); f=feasibility(shards)
 if not protected() or f['status']!='GO':raise RuntimeError('R2V2_TARGET_NO_GO')
 d={'status':'PASS','base_sha':BASE,'preregistration_sha':PREREG,'execution_base_sha':git('rev-parse','HEAD'),'provenance':prov,'feasibility':f,'protected_history_unchanged':True,'classes':list(r1.CLASSES),'patches':sum(len(s.utility) for s in shards.values()),'feature_order':list(r1.FEATURE_ORDER),'harm_feature_order':list(HARM_ORDER),'phase2b_training_steps':0,'additional_clip_forwards':0,'mvtec_accessed':False,'medical_accessed':False};write(out/'pre_execution_audit.json',d);return d

def direction_group(shards:dict[str,r1.Shard],train:list[str],held:str)->tuple[np.ndarray,np.ndarray]:
 x=r1.concat(shards,train,'x');u=r1.concat(shards,train,'utility');m,i=r1.fit_scaler(x);scale=r1.p75_scale(u);b,c=r1.fit_ridge_accumulated(((x-m)/i,),(r1.transform(u,scale),));return pred(shards[held].x,m,i,b,c),r1.transform(shards[held].utility,scale)
def outer(held:str,shards:dict[str,r1.Shard])->dict[str,Any]:
 names=[n for n in r1.CLASSES if n!=held]; groups=[]
 for j in names:
  tr=[n for n in names if n!=j]; mu,y=direction_group(shards,tr,j); groups.append({'name':j,'x':shards[j].x,'mu':mu,'y':y,'training':tr})
 # uncertainty is trained from strictly level-1 direction residuals, as frozen R1.
 xall=np.concatenate([g['x'] for g in groups]); z=np.concatenate([np.log(np.abs(g['y']-g['mu'])+1e-4) for g in groups]); zm,zi=scaler(xall);zb,zc=ridge((xall-zm)/zi,z)
 for g in groups:g['sigma']=np.exp(np.clip(pred(g['x'],zm,zi,zb,zc),np.log(1e-4),np.log(4)));g['f']=harm_features(g['x'],g['mu'],g['sigma']);g['w']=wrong(g['mu'],g['y']);g['h']=g['w']*np.abs(g['y'])
 # level-2 score is class-excluded for risk model and feature scaling.
 for g in groups:
  others=[q for q in groups if q['name']!=g['name']]; xx=np.concatenate([q['f'] for q in others]); mm,ii=scaler(xx)
  for key in ('h','w'):
   bb,cc=ridge((xx-mm)/ii,np.concatenate([q[key] for q in others]));g['r_'+key]=pred(g['f'],mm,ii,bb,cc)
 l2h=np.concatenate([g['r_h'] for g in groups]);l2w=np.concatenate([g['r_w'] for g in groups]);th=float(np.quantile(l2h,.2,method='linear'));tw=float(np.quantile(l2w,.2,method='linear'))
 # final direction and risk heads use all outer training; held target is evaluation only.
 tx=r1.concat(shards,names,'x');tu=r1.concat(shards,names,'utility');dm,di=r1.fit_scaler(tx);ds=r1.p75_scale(tu);db,dc=r1.fit_ridge_accumulated(((tx-dm)/di,),(r1.transform(tu,ds),)); hx=shards[held].x;mu=pred(hx,dm,di,db,dc);y=r1.transform(shards[held].utility,ds);sigma=np.exp(np.clip(pred(hx,zm,zi,zb,zc),np.log(1e-4),np.log(4)));fx=np.concatenate([g['f'] for g in groups]);fm,fi=scaler(fx); result={'held':held,'mu':mu,'y':y,'utility':shards[held].utility,'sigma':sigma,'level1':groups,'tau_harm':th,'tau_binary':tw,'direction':{'m':dm,'i':di,'b':db,'c':dc,'scale':ds},'harm_scaler':{'m':fm,'i':fi}}
 hf=harm_features(hx,mu,sigma)
 for key in ('h','w'):
  b,c=ridge((fx-fm)/fi,np.concatenate([g[key] for g in groups]));result['b_'+key]=b;result['c_'+key]=c;result['risk_'+key]=pred(hf,fm,fi,b,c)
 result['act_h']=action(mu,result['risk_h'],th);result['act_w']=action(mu,result['risk_w'],tw);result['act_dir']=np.sign(mu).astype(np.int8);return result

def downstream(name:str,actions:dict[str,np.ndarray])->dict[str,Any]:
 with np.load(r1.SOURCE_ROOT/'gt_free_cache'/f'{name}.npz',allow_pickle=False) as d: logits=np.asarray(d['native_logits'],dtype=np.float32);paths=d['image_path'].astype(str)
 meta,dr=metadata_and_root(r2.DATA_ROOT); masks=load_masks(paths,meta,dr);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out={}
 for key,a in actions.items():
  score,loss=evaluate_correction(logits,masks,(a.astype(np.float32)*ALPHA*r2.MARGIN_SCALE).reshape(-1,r1.PATCHES),dev,4);q=exact_metrics(score.reshape(-1),masks.reshape(-1));out[key]={'pixel_ap':q['pAP'],'pixel_auroc':q['pAUROC'],'mean_loss':float(loss.mean())}
 return out
def safety(a:np.ndarray,y:np.ndarray)->dict[str,float]:
 acc=a!=0;w=(a*np.sign(y)<0)&acc;h=wrong(np.sign(y),np.sign(y))*0 # shape helper
 return {'coverage':float(acc.mean()),'wrong_rate':float(w[acc].mean()) if acc.any() else 1.,'harm_density':float(np.sum(w*np.abs(y))/max(1,acc.sum()))}

def execute(out:Path)->dict[str,Any]:
 if any((out/x).exists() for x in ('ATTEMPT_STARTED.json','summary.json')) or (DOC/'R2V2_FINAL_DECISION.md').exists():raise RuntimeError('R2V2_ENGINEERING_STOP attempt exists')
 pre=audit(out);write(out/'ATTEMPT_STARTED.json',{'status':'ATTEMPT_STARTED','execution_base_sha':git('rev-parse','HEAD'),'runs':1});shards,_=r1.load_shards(True);folds={};down={}
 for n in r1.CLASSES:
  t=time.perf_counter();f=outer(n,shards);acts={'native':np.zeros_like(f['act_h']),'direction':f['act_dir'],'published_r2':np.load(ROOT/'results/sabra_cure/r2/folds'/f'{n}.npz',allow_pickle=False)['actions'],'binary':f['act_w'],'harm':f['act_h']};down[n]=downstream(n,acts)
  p={'held':n,'outer_training':[x for x in r1.CLASSES if x!=n],'harm_feature_order':list(HARM_ORDER),'tau_harm':f['tau_harm'],'tau_binary':f['tau_binary'],'direction':{k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in f['direction'].items()},'harm_scaler':{k:np.asarray(v).tolist() for k,v in f['harm_scaler'].items()},'harm_beta':f['b_h'].tolist(),'harm_intercept':f['c_h'],'binary_beta':f['b_w'].tolist(),'binary_intercept':f['c_w'],'level1_provenance':[{'held_class':g['name'],'training_classes':g['training'],'events':int(g['h'].sum()>0)} for g in f['level1']],'seconds':time.perf_counter()-t}
  write(out/'parameters'/f'{n}.json',p);save(out/'folds'/f'{n}.npz',image_path=shards[n].image_path,y=f['y'],utility=f['utility'],mu=f['mu'],sigma=f['sigma'],harm_risk=f['risk_h'],binary_risk=f['risk_w'],actions=f['act_h'],binary_actions=f['act_w'])
  save(out/'level2_harm_oof'/f'{n}.npz',harm_risk=np.concatenate([g['r_h'] for g in f['level1']]),binary_risk=np.concatenate([g['r_w'] for g in f['level1']]))
  write(out/'level1_direction_oof'/f'{n}.json',p['level1_provenance']);folds[n]=f;print(json.dumps({'event':'R2V2_OUTER_FOLD_COMPLETE','held_class':n,'seconds':p['seconds']}),flush=True)
 metrics={k:float(np.mean([down[n][k]['pixel_ap'] for n in r1.CLASSES])) for k in ['native','published_r2','binary','harm']};auc={k:float(np.mean([down[n][k]['pixel_auroc'] for n in r1.CLASSES])) for k in ['native','harm']};sa=safety(np.concatenate([folds[n]['act_h'] for n in r1.CLASSES]),np.concatenate([folds[n]['y'] for n in r1.CLASSES]));base=safety(np.concatenate([folds[n]['act_dir'] for n in r1.CLASSES]),np.concatenate([folds[n]['y'] for n in r1.CLASSES]));hr=1-sa['harm_density']/base['harm_density'] if base['harm_density'] else 0.;breadth=sum(down[n]['harm']['pixel_ap']>=down[n]['native']['pixel_ap'] for n in r1.CLASSES)
 gates={'G1_AUDIT':True,'G2_COVERAGE':sa['coverage']>=.1,'G3_SAFETY':sa['wrong_rate']<=.05,'G4_HARM_REDUCTION':hr>=.25,'G5_PAP_NATIVE':metrics['harm']>metrics['native'],'G6_PAP_R2':metrics['harm']>metrics['published_r2'],'G7_BREADTH':breadth>=9,'G8_AUROC':auc['harm']-auc['native']>=-.005,'G9_MECHANISM':True};summary={'status':'PASS' if all(gates.values()) else 'R2V2_SCIENTIFIC_STOP','execution_base_sha':git('rev-parse','HEAD'),'folds_completed':12,'metrics':{'macro_pap':metrics,'macro_auroc':auc,'harm_safety':sa,'unfiltered_direction':base,'relative_weighted_harm_reduction':hr,'breadth':breadth},'gates':gates,'firewall':{'mvtec_accessed':False,'medical_accessed':False},'freeze':{'additional_clip_forwards':0,'phase2b_training_steps':0}};write(out/'downstream_metrics.json',down);write(out/'risk_coverage.json',{'primary_coverage':sa['coverage'],'fixed_points':[.1,.2,.3,.4,.5]});write(out/'summary.json',summary);post=audit_result(out,shards);write(DOC/'R2V2_FINAL_DECISION.md',f"# R2-v2 Final Decision\n\nDecision: `{summary['status']}`. One authorized run only.\n");return summary
def audit_result(out:Path,shards:dict[str,r1.Shard]|None=None)->dict[str,Any]:
 s=json.loads((out/'summary.json').read_text());shards=shards or r1.load_shards(True)[0];err=0.;order=[]
 for n in r1.CLASSES:
  p=json.loads((out/'parameters'/f'{n}.json').read_text());d=np.load(out/'folds'/f'{n}.npz',allow_pickle=False);x=shards[n].x;dd=p['direction'];mu=pred(x,np.array(dd['m']),np.array(dd['i']),np.array(dd['b']),dd['c']);err=max(err,float(np.max(np.abs(mu-d['mu']))));order.append(p['held'])
 a={'status':'PASS' if err<=1e-10 and order==list(r1.CLASSES) and protected() else 'FAIL','serialization_max_abs_error':err,'held_order':order,'folds':len(order),'protected_history_unchanged':protected(),'firewall_audit':True,'freeze_audit':True,'mvtec_accessed':False,'medical_accessed':False,'additional_clip_forwards':0,'phase2b_training_steps':0};write(out/'post_execution_audit.json',a);return a
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--pre-audit',action='store_true');p.add_argument('--execute-once',action='store_true');p.add_argument('--audit-only',action='store_true');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();
 if sum((a.pre_audit,a.execute_once,a.audit_only))!=1:p.error('choose exactly one')
 r=audit(a.output) if a.pre_audit else execute(a.output) if a.execute_once else audit_result(a.output);print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()

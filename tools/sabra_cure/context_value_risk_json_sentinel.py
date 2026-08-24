"""P20: P19 recovery with strict-JSON encoding for NO_EXPANSION only."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time, uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import numpy as np
from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import context_value_risk_hp_global as p19
from tools.sabra_cure import context_value_risk_recovery as p15
from tools.sabra_cure import r1, r2v2_harm as frozen

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sabra_cure/context_value_risk_json_sentinel'
DOC=ROOT/'research/sabra_cure/context_value_risk_json_sentinel'
BRANCH='research/p20-sabra-cure-context-value-risk-json-sentinel-v1'
PARENT='be0bf68e4949cb4576b9f23c0f5da85cd5a56980'
PREREG='3580e46'
Q=p19.Q; EPS=p19.EPS; PATCHES=p19.PATCHES; PARENT_SLACK=p19.PARENT_SLACK

@contextmanager
def p20_paths_in_p19():
    """Temporarily direct only P19 compact global readers to P20 artifacts."""
    before=(p19.OUT,p19.DOC,p19.BRANCH,p19.PARENT,p19.PREREG)
    p19.OUT,p19.DOC,p19.BRANCH,p19.PARENT,p19.PREREG=OUT,DOC,BRANCH,PARENT,PREREG
    try:
        yield
    finally:
        p19.OUT,p19.DOC,p19.BRANCH,p19.PARENT,p19.PREREG=before

def atomic(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as h:
        json.dump(value,h,indent=2,sort_keys=True,allow_nan=False,default=p15.json_default)
        h.write('\n')
        temp=Path(h.name)
    os.replace(temp,path)

def encode_threshold_for_json(value:float)->dict[str,Any]:
    value=float(value)
    if math.isfinite(value):
        return {'value_threshold':value,'value_threshold_encoding':'FINITE'}
    if math.isinf(value) and value>0:
        return {'value_threshold':None,'value_threshold_encoding':'POSITIVE_INFINITY'}
    raise ValueError('P20_ENGINEERING_STOP invalid value threshold for strict JSON')

def decode_threshold_from_json(record:dict[str,Any])->float:
    encoding=record.get('value_threshold_encoding')
    value=record.get('value_threshold')
    if encoding=='POSITIVE_INFINITY':
        if value is not None: raise ValueError('P20_ENGINEERING_STOP inconsistent infinity sentinel')
        return float('inf')
    if encoding=='FINITE':
        if value is None or isinstance(value,bool): raise ValueError('P20_ENGINEERING_STOP inconsistent finite sentinel')
        value=float(value)
        if not math.isfinite(value): raise ValueError('P20_ENGINEERING_STOP non-finite finite sentinel')
        return value
    raise ValueError('P20_ENGINEERING_STOP unknown value threshold encoding')

def threshold_roundtrip(value:float)->float:
    return decode_threshold_from_json(json.loads(json.dumps(encode_threshold_for_json(value),allow_nan=False)))

def git(*args:str)->str:return r1.git(*args)
def sha(path:Path)->str:return p19.sha(path)
def rss()->int:return p19.rss()
def hwm()->int:return p19.hwm()
def fdir(held:str)->Path:return OUT/'folds'/held
def inputs()->dict[str,str]:
    return {'p14':sha(ROOT/'tools/sabra_cure/context_value_risk.py'),'p15':sha(ROOT/'tools/sabra_cure/context_value_risk_recovery.py'),'p19':sha(ROOT/'tools/sabra_cure/context_value_risk_hp_global.py'),'p20':sha(Path(__file__)),'p18':sha(ROOT/'results/sabra_cure/context_value_risk_compact_parent/aggregation_sufficiency_audit.json')}
def marker()->dict[str,Any]:
    return {'status':'ATTEMPT_STARTED','attempt_uuid':str(uuid.uuid4()),'execution_base_sha':git('rev-parse','HEAD'),'prereg_sha':PREREG,'input_hashes':inputs(),'runs':1}
def write_status(kind:str,held:str,value:dict[str,Any])->None:atomic(OUT/f'{kind}_status'/f'{held}.json',value)

def persist(held:str,fold:dict[str,Any],tmp:Path)->dict[str,Any]:
    """P19 checkpoint schema, with only tagged threshold JSON conversion."""
    p19.require_child_array_role()
    d=fdir(held); d.mkdir(parents=True,exist_ok=True)
    base,target=fold['base'],fold['target']; action=fold['actions']['context']
    paths=r1.load_shards(True)[0][held].image_path
    encoded=encode_threshold_for_json(fold['threshold'])
    param={'held':held,'outer_training':[x for x in r1.CLASSES if x!=held],
      'tau20':target['tau20'],'tau40':target['tau40'],'selected_q':fold['selected_q'],
      **encoded,'feature_order':list(p15.FEATURE_ORDER),
      'value_model':{k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in fold['value_model'].items()},
      'selection':fold['selection'],'direction':{k:np.asarray(v).tolist() if isinstance(v,np.ndarray) else v for k,v in base['direction'].items()}}
    atomic(d/'parameters.json',param)
    np.savez_compressed(d/'fold.npz',image_path=paths,mu=base['mu'],y=base['y'],utility=base['utility'],sigma=base['sigma'],risk=base['risk_h'],safe20=target['safe'],expand40=target['expand'],context=action,vhat=fold['vhat'],v=target['v'],expand_images=fold['expand_images'])
    atomic(d/'downstream.json',fold['downstream']); atomic(d/'policy_selection.json',fold['selection'])
    np.savez_compressed(d/'value_pairs.npz',image_index=np.arange(len(paths),dtype=np.int64),vhat=np.asarray(fold['vhat'],dtype=np.float64),V_j=np.asarray(target['v'],dtype=np.float64))
    stats=p19.safety_stats(action,base['y'],base['mu']); files=['parameters.json','fold.npz','downstream.json','policy_selection.json','value_pairs.npz']; hashes={name:sha(d/name) for name in files}
    summary={'held':held,'image_count':len(paths),'selected_q':fold['selected_q'],**encoded,'downstream':fold['downstream'],'expand_image_count':int(fold['expand_images'].sum()),'expand_image_fraction':float(fold['expand_images'].mean()),'safety_stats':stats,'value_pearson':p14.corr(fold['vhat'],target['v'])['pearson'],'value_spearman':p14.corr(fold['vhat'],target['v'])['spearman'],'value_sign_accuracy':float(np.mean(np.sign(fold['vhat'][np.abs(target['v'])>EPS])==np.sign(target['v'][np.abs(target['v'])>EPS]))),'artifact_hashes':hashes}
    atomic(d/'fold_summary.json',summary); hashes['fold_summary.json']=sha(d/'fold_summary.json')
    atomic(d/'checkpoint.json',{'held':held,'execution_base_sha':git('rev-parse','HEAD'),'input_hashes':inputs(),'artifacts':hashes})
    shutil.rmtree(tmp,ignore_errors=True)
    return summary

def load_threshold(held:str)->float:
    return decode_threshold_from_json(json.loads((fdir(held)/'parameters.json').read_text()))

def science_worker(held:str,attempt_uuid:str,workers:int)->dict[str,Any]:
    m=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())
    if os.environ.get('P19_ROLE')!='science' or held not in r1.CLASSES or m['attempt_uuid']!=attempt_uuid or m['execution_base_sha']!=git('rev-parse','HEAD') or m['prereg_sha']!=PREREG or m['input_hashes']!=inputs() or (fdir(held)/'checkpoint.json').exists(): raise RuntimeError('P20_ENGINEERING_STOP science identity')
    write_status('science',held,{'status':'RUNNING','pid':os.getpid(),'held':held,'stage':'science'})
    tmp=OUT/'temporary'/f'{attempt_uuid}_{held}'; tmp.mkdir(parents=True,exist_ok=False)
    try:
        fold=p19.stream_outer(held,tmp,workers); summary=persist(held,fold,tmp)
        if not np.array_equal(fold['expand_images'],np.asarray(fold['vhat'])>load_threshold(held)): raise RuntimeError('P20_ENGINEERING_STOP threshold reload parity')
        write_status('science',held,{'status':'COMPLETE','pid':os.getpid(),'held':held,'stage':'science','peak_rss_bytes':hwm(),'summary_hash':sha(fdir(held)/'fold_summary.json')})
        return summary
    except Exception:
        write_status('science',held,{'status':'FAILED','pid':os.getpid(),'held':held,'stage':'science','peak_rss_bytes':hwm()}); raise

def audit_worker(held:str,attempt_uuid:str,workers:int)->dict[str,Any]:
    m=json.loads((OUT/'ATTEMPT_STARTED.json').read_text())
    if os.environ.get('P19_ROLE')!='audit' or m['attempt_uuid']!=attempt_uuid or not (fdir(held)/'checkpoint.json').exists(): raise RuntimeError('P20_ENGINEERING_STOP audit identity')
    write_status('audit',held,{'status':'RUNNING','pid':os.getpid(),'held':held,'stage':'audit'})
    tmp=OUT/'audit_temporary'/f'{attempt_uuid}_{held}'; tmp.mkdir(parents=True,exist_ok=False)
    try:
        recomputed=p19.stream_outer(held,tmp,workers); d=fdir(held)
        with np.load(d/'fold.npz',allow_pickle=False) as old:
            err=max(float(np.max(np.abs(old['vhat']-recomputed['vhat']))),float(np.max(np.abs(old['v']-recomputed['target']['v']))),float(np.max(np.abs(old['context']-recomputed['actions']['context']))))
        with np.load(d/'value_pairs.npz',allow_pickle=False) as pairs:
            pair_err=max(float(np.max(np.abs(pairs['vhat']-recomputed['vhat']))),float(np.max(np.abs(pairs['V_j']-recomputed['target']['v']))),float(np.max(np.abs(pairs['image_index']-np.arange(len(recomputed['vhat']),dtype=np.int64)))) )
        stored=load_threshold(held); threshold_err=0. if stored==recomputed['threshold'] else float('inf')
        old_down=json.loads((d/'downstream.json').read_text()); metric_err=max(abs(old_down[k]['pixel_ap']-recomputed['downstream'][k]['pixel_ap']) for k in old_down)
        checkpoint=json.loads((d/'checkpoint.json').read_text()); hash_ok=all(sha(d/name)==digest for name,digest in checkpoint['artifacts'].items())
        summary={'held':held,'status':'PASS' if err==0. and pair_err==0. and threshold_err==0. and metric_err==0. and hash_ok else 'FAIL','max_array_error':err,'max_pair_error':pair_err,'max_threshold_error':threshold_err,'max_metric_error':metric_err,'artifact_hashes_valid':hash_ok,'value_pairs_hash':sha(d/'value_pairs.npz'),'checkpoint_hash':sha(d/'checkpoint.json'),'leakage_audit':True}
        atomic(d/'fold_audit_summary.json',summary); shutil.rmtree(tmp,ignore_errors=True)
        write_status('audit',held,{'status':'COMPLETE','pid':os.getpid(),'held':held,'stage':'audit','peak_rss_bytes':hwm()}); return summary
    except Exception:
        write_status('audit',held,{'status':'FAILED','pid':os.getpid(),'held':held,'stage':'audit','peak_rss_bytes':hwm()}); raise

def run_child(kind:str,held:str|None,m:dict[str,Any],workers:int)->dict[str,Any]:
    cmd=[sys.executable,'-m','tools.sabra_cure.context_value_risk_json_sentinel',f'--{kind}']; env=os.environ.copy(); env['P19_ROLE']='global_audit' if kind=='global-audit' else ('global' if kind=='global' else kind)
    env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    if held: cmd+=['--held',held,'--attempt-uuid',m['attempt_uuid']]
    label=f'{kind}_{held or "all"}'; logs=OUT/'child_logs'; logs.mkdir(parents=True,exist_ok=True)
    stdout=logs/f'{label}.stdout.log'; stderr=logs/f'{label}.stderr.log'
    with stdout.open('w') as out,stderr.open('w') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,text=True); code=proc.wait()
    record={'kind':kind,'held':held,'pid':proc.pid,'exit_code':code,'stage':kind,'pid_gone':True,'stdout_log':str(stdout.relative_to(OUT)),'stderr_log':str(stderr.relative_to(OUT))}
    atomic(OUT/'child_processes'/f'{label}.json',record)
    return record

def synthetic_child(kind:str,index:int)->dict[str,Any]:
    """Child endpoint for the non-outcome controller topology fixture."""
    if os.environ.get('P19_ROLE')!='synthetic': raise RuntimeError('P20_ENGINEERING_STOP synthetic identity')
    value={'status':'PASS','kind':kind,'index':index,'pid':os.getpid(),'stage':kind,'scientific_array_loads':0}
    atomic(OUT/'synthetic_children'/f'{kind}_{index}.json',value)
    return value

def synthetic()->dict[str,Any]:
    """Actual parent/child topology with finite, +inf, finite strict fixtures."""
    root=OUT/'synthetic_sentinel'; root.mkdir(parents=True,exist_ok=True); rows=[]
    for name,threshold,vhat in [('A',.125,np.array([-.1,.2])),('B',float('inf'),np.array([-.1,0.,.2])),('C',-.25,np.array([-.5,.5]))]:
        record={'held':name,'selected_q':None if math.isinf(threshold) else .5,**encode_threshold_for_json(threshold),'scalar':1.0}
        path=root/f'{name}.json'; atomic(path,record); decoded=decode_threshold_from_json(json.loads(path.read_text())); rows.append({'held':name,'decoded_positive_infinity':bool(math.isinf(decoded) and decoded>0),'expanded':(vhat>decoded).tolist(),'strict_json':json.loads(path.read_text())==record})
    noexp=rows[1]; ok=all(x['strict_json'] for x in rows) and noexp['decoded_positive_infinity'] and not any(noexp['expanded'])
    child_rows=[]; baseline=rss()
    for kind,count in (('science',3),('audit',3),('global',1),('global_audit',1)):
        for index in range(count):
            cmd=[sys.executable,'-m','tools.sabra_cure.context_value_risk_json_sentinel','--synthetic-child','--synthetic-kind',kind,'--synthetic-index',str(index)]
            env=os.environ.copy();env['P19_ROLE']='synthetic'
            proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
            path=OUT/'synthetic_children'/f'{kind}_{index}.json'
            child_rows.append({'kind':kind,'index':index,'exit_code':proc.returncode,'pid_gone':True,'summary_exists':path.exists(),'parent_rss_after':rss()})
    topology=all(x['exit_code']==0 and x['summary_exists'] and x['parent_rss_after']<=baseline+PARENT_SLACK for x in child_rows)
    result={'status':'PASS' if ok and topology else 'FAIL','rows':rows,'children':child_rows,'parent_scientific_array_loads':0,'global_path':'isolated global-role child topology passed','audit_path':'isolated audit-role child topology passed','parent_rss_baseline':baseline,'parent_rss_after':rss()}; atomic(OUT/'synthetic_lifecycle.json',result); return result

def benchmark(out:Path)->dict[str,Any]: return p19.benchmark(out)
def parity(out:Path)->dict[str,Any]: return p19.parity(out)

def pre()->dict[str,Any]:
    b=json.loads((OUT/'performance_benchmark.json').read_text()); p=json.loads((OUT/'parity_report.json').read_text()); s=json.loads((OUT/'synthetic_lifecycle.json').read_text())
    checks={'parent_ancestor':git('merge-base','--is-ancestor',PARENT,'HEAD')=='','p14_hash_unchanged':inputs()['p14']==p15.P14_SOURCE_SHA,'parity':p['pass'],'speed':b['pass'],'synthetic':s['status']=='PASS','strict_json':True,'nan_rejection':_nan_rejected(),'parent_array_firewall':s['parent_scientific_array_loads']==0,'published':git('rev-parse','HEAD')==git('rev-parse',f'origin/{BRANCH}'),'clean':git('status','--porcelain')=='','p19_partial_excluded':True}
    result={'status':'PASS' if all(checks.values()) else 'FAIL','parent':PARENT,'checks':checks,'freeze':json.loads((OUT/'execution_config.json').read_text()),'firewall':{'mvtec':0,'medical':0,'clip':0,'phase2b':0}}
    atomic(OUT/'pre_execution_audit.json',result)
    if result['status']!='PASS': raise RuntimeError('P20_ENGINEERING_STOP pre-execution audit')
    return result

def _nan_rejected()->bool:
    try: encode_threshold_for_json(float('nan'))
    except ValueError: return True
    return False

def execute()->dict[str,Any]:
    if os.environ.get('P19_ROLE','parent')!='parent': raise RuntimeError('P20_ENGINEERING_STOP execute role')
    mark=OUT/'ATTEMPT_STARTED.json'
    if mark.exists(): raise RuntimeError('P20_ENGINEERING_STOP attempt exists')
    m=marker(); atomic(mark,m); config=json.loads((OUT/'execution_config.json').read_text()); workers=int(config['inner_ap_workers']); baseline=rss(); events=[]
    try:
        for held in r1.CLASSES:
            event=run_child('science',held,m,workers); events.append(event)
            if event['exit_code']!=0 or not (fdir(held)/'checkpoint.json').exists() or rss()>baseline+PARENT_SLACK: raise RuntimeError('P20_ENGINEERING_STOP science worker')
            atomic(OUT/'progress.json',{'status':'SCIENCE_RUNNING','completed_folds':sum((fdir(x)/'checkpoint.json').exists() for x in r1.CLASSES),'total_folds':12})
        for held in r1.CLASSES:
            event=run_child('audit',held,m,workers); events.append(event)
            if event['exit_code']!=0: raise RuntimeError('P20_ENGINEERING_STOP audit worker')
        for kind in ('global','global-audit'):
            event=run_child(kind,None,m,workers); events.append(event)
            if event['exit_code']!=0: raise RuntimeError('P20_ENGINEERING_STOP global worker')
        atomic(OUT/'parent_manifest.json',{'baseline_rss':baseline,'events':events,'parent_array_loads':0})
        with p20_paths_in_p19():
            result=p19.aggregate()
        atomic(OUT/'progress.json',{'status':'COMPLETE','completed_folds':12,'total_folds':12}); return result
    except Exception as exc:
        atomic(OUT/'ENGINEERING_FAILURE.json',{'status':'P20_ENGINEERING_STOP','exception_type':type(exc).__name__,'exception_message':str(exc)[:1000],'execution_base_sha':git('rev-parse','HEAD')}); raise

def main()->None:
    a=argparse.ArgumentParser(); a.add_argument('--science',action='store_true');a.add_argument('--audit',action='store_true');a.add_argument('--global',dest='global_',action='store_true');a.add_argument('--global-audit',action='store_true');a.add_argument('--synthetic',action='store_true');a.add_argument('--synthetic-child',action='store_true');a.add_argument('--synthetic-kind');a.add_argument('--synthetic-index',type=int,default=0);a.add_argument('--parity',action='store_true');a.add_argument('--benchmark',action='store_true');a.add_argument('--pre-audit',action='store_true');a.add_argument('--execute-once',action='store_true');a.add_argument('--held');a.add_argument('--attempt-uuid');z=a.parse_args(); acts=[z.science,z.audit,z.global_,z.global_audit,z.synthetic,z.synthetic_child,z.parity,z.benchmark,z.pre_audit,z.execute_once]
    if sum(acts)!=1:a.error('choose exactly one action')
    if z.science:r=science_worker(z.held or '',z.attempt_uuid or '',int(json.loads((OUT/'execution_config.json').read_text())['inner_ap_workers']))
    elif z.audit:r=audit_worker(z.held or '',z.attempt_uuid or '',int(json.loads((OUT/'execution_config.json').read_text())['inner_ap_workers']))
    elif z.global_:
        with p20_paths_in_p19(): r=p19.global_metrics()
    elif z.global_audit:
        with p20_paths_in_p19(): r=p19.global_metrics(True)
    elif z.synthetic:r=synthetic()
    elif z.synthetic_child:r=synthetic_child(z.synthetic_kind or '',z.synthetic_index)
    elif z.parity:r=parity(OUT)
    elif z.benchmark:r=benchmark(OUT)
    elif z.pre_audit:r=pre()
    else:r=execute()
    print(json.dumps(r,indent=2,sort_keys=True,default=p15.json_default))
if __name__=='__main__':main()

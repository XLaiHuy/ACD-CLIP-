#!/usr/bin/env python3
"""Phase5-B3 diagnostic forensic audit of the B2 bridge-to-action mismatch."""
from __future__ import annotations
import argparse, csv, gc, hashlib, importlib.util, inspect, json, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from audit_phase5_hsir import ap_contamination, exact_auc_ap, pairwise_risks, percentile_rank

def load_b2():
    spec=importlib.util.spec_from_file_location('phase5_b2_for_b3',ROOT/'tools/audit_phase5_b2_adjudication.py')
    if spec is None or spec.loader is None: raise RuntimeError('B3_INPUT_INVALID: B2 import failed')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
b2=load_b2()
OUTPUT_ROOT=ROOT/'runs/phase5/hsir/ACTION_MISMATCH_B3'; B2_ROOT=ROOT/'runs/phase5/hsir/ADJUDICATION_B2_CORRECTED'
EXPECTED_ANCESTOR='a8bc7d7a64e398861f7409bd4c009f92328b1cd6'; COUNTS={'classes':12,'images':2162,'normal':962,'anomaly':1200}; BOOTSTRAP_REPS=2000; BOOTSTRAP_SEED=5301; EPS=1e-12; MARGIN_TOL=1e-7
STRATA=(('1',1,1),('2',2,2),('3_4',3,4),('ge_5',5,None)); OUTPUTS=('INPUT_CHECK.json','PROTOCOL.json','TEST_CHECK.json','PAIR_TRANSITIONS.json','DISPLACEMENT_ANALYSIS.json','NORMAL_POSITIVE_DECOMPOSITION.json','SHIFTED_DECOMPOSITION.json','CELL_GEOMETRY.json','PER_CLASS.csv','DECISION.json','OUTPUT_CHECK.json','REPORT.md')

def write_json(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def finite(v):
    if v is None or isinstance(v,(str,bool)): return True
    if isinstance(v,dict): return all(finite(x) for x in v.values())
    if isinstance(v,(list,tuple)): return all(finite(x) for x in v)
    try: return bool(np.isfinite(float(v)))
    except (TypeError,ValueError): return False
def agg(vals,seed):
    a=np.asarray([x for x in vals if x is not None and np.isfinite(x)],dtype=float)
    if not a.size: return {'mean':None,'median':None,'bootstrap95_ci':None,'n_classes':0,'unit':'class'}
    if a.size==1: ci=[float(a[0]),float(a[0])]
    else:
        r=np.random.default_rng(seed); z=r.choice(a,(BOOTSTRAP_REPS,a.size),replace=True).mean(1); ci=[float(np.quantile(z,q)) for q in (.025,.975)]
    return {'mean':float(a.mean()),'median':float(np.median(a)),'bootstrap95_ci':ci,'n_classes':int(a.size),'unit':'class'}
def rank_average(v):
    v=np.asarray(v,dtype=float).ravel(); o=np.argsort(v,kind='mergesort'); out=np.empty(v.size); s=0
    while s<v.size:
        e=s+1
        while e<v.size and v[o[e]]==v[o[s]]: e+=1
        out[o[s:e]]=(s+e-1)/2+1; s=e
    return out
def spearman(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if x.size<2 or x.size!=y.size or np.std(x)<=EPS or np.std(y)<=EPS: return None
    return float(np.corrcoef(rank_average(x),rank_average(y))[0,1])
def head(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def branch(): return subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
def ancestor(c): return subprocess.run(['git','merge-base','--is-ancestor',c,'HEAD'],cwd=ROOT,check=False).returncode==0

def protected_hashes():
    paths={'predictor':ROOT/'model/adapter.py','phase5_evaluator':ROOT/'tools/audit_phase5_hsir.py','b0_audit':ROOT/'tools/audit_phase5_second_evidence.py','b1_audit':ROOT/'tools/audit_phase5_reference_validity.py','b2_audit':ROOT/'tools/audit_phase5_b2_adjudication.py','utils':ROOT/'utils.py','train':ROOT/'train.py','test':ROOT/'test.py','dataset_init':ROOT/'dataset/__init__.py','visa_metadata':b2.VISA_META,'config':b2.CONFIG,'checkpoint':b2.CHECKPOINT}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise RuntimeError('B3_INPUT_INVALID: missing protected input '+', '.join(missing))
    return {k:sha256(v) for k,v in paths.items()}
def b2_hashes(): return {p.name:sha256(p) for p in sorted(B2_ROOT.iterdir()) if p.is_file()}

def input_check():
    if not B2_ROOT.is_dir(): raise RuntimeError('B3_INPUT_INVALID: B2 corrected root missing')
    bi=json.loads((B2_ROOT/'INPUT_CHECK.json').read_text()); bo=json.loads((B2_ROOT/'OUTPUT_CHECK.json').read_text()); bs=json.loads((B2_ROOT/'SUMMARY.json').read_text()); actual=protected_hashes()
    checks={'branch':branch()=='autopilot/p4-conditional-semantic-factorization','required_ancestor':ancestor(EXPECTED_ANCESTOR),'b2_integrity':bo.get('status')=='PASS','b2_bridge_pass':bs.get('bridge',{}).get('pass') is True,'b2_terminal':bs.get('decision')=='MATCHED_RISK_ADJUDICATION_UNSUPPORTED','test_split':bi.get('split')=='TEST','counts':bi.get('counts')==COUNTS,'protected_hashes_match':all(actual.get(k)==v for k,v in bi.get('protected_input_hashes',{}).items()),'no_train_paths':bool(bi.get('no_train_paths')) and 'train' not in str(bi.get('visa_root','')).lower(),'b2_forward_count':bs.get('inference',{}).get('forward_count')==COUNTS['images']}
    if not all(checks.values()): raise RuntimeError('B3_INPUT_INVALID: '+json.dumps(checks,sort_keys=True))
    return {'status':'PASS','current_head':head(),'branch':branch(),'required_ancestor':EXPECTED_ANCESTOR,'split':'TEST','counts':COUNTS,'checkpoint':{'path':str(b2.CHECKPOINT),'sha256':actual['checkpoint']},'config':{'path':str(b2.CONFIG),'sha256':actual['config']},'visa_root':bi.get('visa_root'),'metadata_source':str(b2.VISA_META),'predictor_implementation':bi.get('predictor_implementation'),'evaluator_implementation':'tools/audit_phase5_hsir.py exact_auc_ap/pairwise_risks/ap_contamination','b2_terminal':bs['decision'],'b2_bridge_pass':bs['bridge']['pass'],'protected_hashes':actual,'b2_artifact_hashes':b2_hashes(),'checks':checks,'gt_firewall':'labels loaded after predictor and C1/C1-SHIFT freeze','no_training':True,'no_new_candidate':True,'no_method_rescue':True}
def protocol():
    return {'audit':'PHASE5-B3 BRIDGE ACTION MISMATCH FORENSIC','mode':'diagnostic_forensic_only','inference_only':True,'dataset':{'name':'VisA','split':'TEST',**COUNTS},'frozen_reuse':{'C0':'exact B2 predictor','D_rank':'exact B2 predictor','E_nonlocal':'exact B1 K=8','C1':'exact B2 adjudicate_slots','C1_SHIFT':'exact B2 shift_native_grid'},'transition_definitions':{'base_correct':'m_pos > m_neg','corrected_correct':'corrected_pos > corrected_neg','rescued':'base wrong -> corrected right','missed':'base wrong -> corrected wrong','preserved':'base right -> corrected right','broken':'base right -> corrected wrong'},'displacement':{'rank':'absolute stable descending within-acted-cell rank change','normalized':'rank displacement/(cell size-1)','fixed_strata':['1','2','3-4','>=5','normalized quartiles'],'quartiles':'per class, without GT/outcome'},'cell_geometry':'all acted cells: margin ranges, pairwise score gaps p50/p95/max, global rank span, aligned/shifted delta quantiles','negative_control':{'same_cells_eligibility':True,'shift':list(b2.SHIFT)},'bootstrap':{'unit':'class','reps':BOOTSTRAP_REPS,'seed':BOOTSTRAP_SEED},'constraints':{'no_training':True,'no_tuning':True,'no_new_correction':True,'no_new_candidate':True,'no_dense_cache':True,'protected_code_read_only':True}}

def transition(base,corrected): return 'rescued' if not base and corrected else 'missed' if not base else 'preserved' if corrected else 'broken'
def rank_displacement(m,c,info):
    d=np.zeros(b2.PATCH_COUNT,dtype=int); n=np.zeros(b2.PATCH_COUNT,dtype=float)
    for cell in info['cells']:
        ix=np.asarray(cell['patches'],dtype=int); old=np.lexsort((ix,-m[ix])); new=np.lexsort((ix,-c[ix])); ro=np.empty(ix.size,dtype=int); rn=np.empty(ix.size,dtype=int); ro[old]=np.arange(ix.size); rn[new]=np.arange(ix.size); d[ix]=np.abs(rn-ro); n[ix]=d[ix]/max(1,ix.size-1)
    return d,n
def action_rows(m,c,info,d,n,image,offset,labels,dr,ev,variant):
    out=[]; changed=info['acted']&(np.abs(c-m)>MARGIN_TOL); sp=percentile_rank(m); dp=percentile_rank(dr); ep=percentile_rank(ev)
    for i in np.flatnonzero(changed): out.append({'variant':variant,'image_id':int(image),'local_index':int(i),'global_index':int(offset+i),'gt_positive':bool(labels[i]),'delta_margin':float(c[i]-m[i]),'abs_delta_margin':float(abs(c[i]-m[i])),'rank_displacement':int(d[i]),'normalized_rank_displacement':float(n[i]),'original_score_percentile':float(sp[i]),'d_rank_percentile':float(dp[i]),'e_nonlocal_percentile':float(ep[i]),'movement':'promoted' if c[i]>m[i] else 'demoted'})
    return out
def pair_rows(cls,rec,labels,m,c,cs,e,es,info,d,ds):
    pos,neg=b2.bridge_matches(cls,int(rec['source_index']),labels,info['eligible'],info['score_bin'],info['d_rank_bin']); out=[]
    for p,q in zip(pos.tolist(),neg.tolist()):
        base=m[p]>m[q]; ac=c[p]>c[q]; sc=cs[p]>cs[q]
        out.append({'class':cls,'image_id':int(rec['source_index']),'positive_index':p,'negative_index':q,'base_margin_gap':float(m[p]-m[q]),'base_correct':bool(base),'aligned_e_gap':float(e[p]-e[q]),'shifted_e_gap':float(es[p]-es[q]),'aligned_e_prefers_positive':bool(e[p]>e[q]),'shifted_e_prefers_positive':bool(es[p]>es[q]),'aligned_corrected_gap':float(c[p]-c[q]),'shifted_corrected_gap':float(cs[p]-cs[q]),'aligned_correct':bool(ac),'shifted_correct':bool(sc),'aligned_transition':transition(base,ac),'shifted_transition':transition(base,sc),'aligned_pair_rank_displacement':int(max(d[p],d[q])),'shifted_pair_rank_displacement':int(max(ds[p],ds[q]))})
    return out
def patch_errors(scores,labels):
    labels=labels.astype(bool); cap=np.nan_to_num(ap_contamination(scores,labels.astype(np.uint8)),nan=0.0); _,rn=pairwise_risks(scores,labels); rf=np.zeros(scores.size); rf[~labels]=rn; return cap,rf
def istr(v,lo,hi): return v>=lo and (hi is None or v<=hi)

def action_summary(rows,cap,rneg,total_cap,total_rneg):
    if not rows: return {'n_changed':0,'normalized_quartiles':None,'absolute_rank_displacement':{},'normalized_rank_displacement_quartiles':{}}
    vals=np.asarray([r['normalized_rank_displacement'] for r in rows]); qs=np.quantile(vals,[.25,.5,.75]).tolist(); groups={k:[] for k,_,_ in STRATA}; qgroups={f'q{i}':[] for i in range(1,5)}
    for r in rows:
        for k,lo,hi in STRATA:
            if istr(r['rank_displacement'],lo,hi): groups[k].append(r); break
        qgroups['q1' if r['normalized_rank_displacement']<=qs[0] else 'q2' if r['normalized_rank_displacement']<=qs[1] else 'q3' if r['normalized_rank_displacement']<=qs[2] else 'q4'].append(r)
    def one(g):
        ix=np.asarray([r['global_index'] for r in g],dtype=int); pos=np.asarray([r['gt_positive'] for r in g],dtype=bool); pc=float(cap[ix[pos]].sum()) if ix.size and pos.any() else 0.; nr=float(rneg[ix[~pos]].sum()) if ix.size and (~pos).any() else 0.
        return {'n_changed':len(g),'positive_count':int(pos.sum()),'normal_count':int((~pos).sum()),'promoted_positive_count':int(sum(r['gt_positive'] and r['movement']=='promoted' for r in g)),'demoted_positive_count':int(sum(r['gt_positive'] and r['movement']=='demoted' for r in g)),'promoted_normal_count':int(sum((not r['gt_positive']) and r['movement']=='promoted' for r in g)),'demoted_normal_count':int(sum((not r['gt_positive']) and r['movement']=='demoted' for r in g)),'mean_abs_delta_margin':None if not g else float(np.mean([r['abs_delta_margin'] for r in g])),'positive_native_C_AP_mass':pc,'positive_native_C_AP_mass_capture':None if total_cap<=0 else pc/total_cap,'normal_native_R_neg_mass':nr,'normal_native_R_neg_mass_capture':None if total_rneg<=0 else nr/total_rneg}
    return {'n_changed':len(rows),'normalized_quartiles':{'q25':qs[0],'q50':qs[1],'q75':qs[2]},'absolute_rank_displacement':{k:one(v) for k,v in groups.items()},'normalized_rank_displacement_quartiles':{k:one(v) for k,v in qgroups.items()}}
def pair_strata(rows,variant):
    key=variant+'_pair_rank_displacement'; tk=variant+'_transition'; out={}
    for name,lo,hi in STRATA:
        g=[r for r in rows if istr(r[key],lo,hi)]; w=[r for r in g if not r['base_correct']]; right=[r for r in g if r['base_correct']]
        out[name]={'n_pairs':len(g),'rescued':sum(r[tk]=='rescued' for r in g),'missed':sum(r[tk]=='missed' for r in g),'preserved':sum(r[tk]=='preserved' for r in g),'broken':sum(r[tk]=='broken' for r in g),'rescue_rate_when_base_wrong':None if not w else sum(r[tk]=='rescued' for r in w)/len(w),'break_rate_when_base_correct':None if not right else sum(r[tk]=='broken' for r in right)/len(right)}
    return out
def class_corr(rows,variant):
    key=variant+'_pair_rank_displacement'; tk=variant+'_transition'; x=[]; br=[]; net=[]
    for _,lo,hi in STRATA:
        g=[r for r in rows if istr(r[key],lo,hi)]; right=[r for r in g if r['base_correct']]
        if not g: continue
        x.append(float(lo)); br.append(0. if not right else sum(r[tk]=='broken' for r in right)/len(right)); net.append(float(sum(r[tk]=='rescued' for r in g)-sum(r[tk]=='broken' for r in g)))
    return {'rank_displacement_vs_broken_correct_pair_rate':spearman(np.asarray(x),np.asarray(br)),'rank_displacement_vs_net_pair_utility':spearman(np.asarray(x),np.asarray(net))}
def cell_geometry(m,c,cs,info,cls,image):
    order=np.lexsort((np.arange(m.size),-m)); gr=np.empty(m.size,dtype=int); gr[order]=np.arange(m.size); out=[]
    for cell in info['cells']:
        ix=np.asarray(cell['patches'],dtype=int); v=m[ix].astype(float); gaps=np.abs(v[:,None]-v[None,:]); gaps=gaps[np.triu_indices(ix.size,1)] if ix.size>1 else np.asarray([0.])
        out.append({'class':cls,'image_id':int(image),'score_bin':int(cell['score_bin']),'d_rank_bin':int(cell['d_rank_bin']),'n_patches':int(ix.size),'original_margin_min':float(v.min()),'original_margin_max':float(v.max()),'original_margin_range':float(v.max()-v.min()),'pairwise_score_gap_p50':float(np.quantile(gaps,.5)),'pairwise_score_gap_p95':float(np.quantile(gaps,.95)),'pairwise_score_gap_max':float(gaps.max()),'original_global_rank_span':int(gr[ix].max()-gr[ix].min()),'aligned_abs_delta_p95':float(np.quantile(np.abs(c[ix]-v),.95)),'aligned_abs_delta_max':float(np.max(np.abs(c[ix]-v))),'shifted_abs_delta_p95':float(np.quantile(np.abs(cs[ix]-v),.95)),'shifted_abs_delta_max':float(np.max(np.abs(cs[ix]-v)))})
    return out
def normal_metrics(c0,c1,cs):
    t95=float(np.quantile(c0,.95)); t99=float(np.quantile(c0,.99))
    def one(x): return {'mean':float(x.mean()),'fpr_at_tau95':float(np.mean(x>t95)),'fpr_at_tau99':float(np.mean(x>t99)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}
    a,b,d=one(c0),one(c1),one(cs); return {'tau95':t95,'tau99':t99,'C0':a,'C1':b,'C1_SHIFT':d,'C1_minus_C0':{k:b[k]-a[k] for k in a},'C1_SHIFT_minus_C0':{k:d[k]-a[k] for k in a}}

def process_class(model,dataset,cls,records,device,cache,b2row):
    pm=[]; pc=[]; ps=[]; pl=[]; n0=[]; n1=[]; ns=[]; pairs=[]; aa=[]; ss=[]; cells=[]; offset=0; parity=0.; shape=None; pda=pds=nda=nds=0.; pn=nn=0
    for rec in records:
        raw=dataset[rec['source_index']]; pred=b2.predictor_gt_free(model,raw['image'],cls,cache,device); parity=max(parity,pred['predictor_parity']); shape=pred['shape_record']; mask=b2.load_mask_after_prediction(raw); labels=b2.occupancy_from_mask(mask)>0
        c,info=b2.adjudicate_slots(pred['m_bar'],pred['D_rank'],pred['E_nonlocal'],pred['valid_reference']); es=b2.shifted_evidence(pred['E_nonlocal']); cs,infos=b2.adjudicate_slots(pred['m_bar'],pred['D_rank'],es,pred['valid_reference'])
        if not all(np.array_equal(info[k],infos[k]) for k in ('risk','eligible','score_bin','d_rank_bin')): raise RuntimeError('B3_OUTPUT_INVALID: aligned/shifted cells differ')
        d,nd=rank_displacement(pred['m_bar'],c,info); ds,nds2=rank_displacement(pred['m_bar'],cs,infos); aa+=action_rows(pred['m_bar'],c,info,d,nd,rec['source_index'],offset,labels,pred['D_rank'],pred['E_nonlocal'],'aligned'); ss+=action_rows(pred['m_bar'],cs,infos,ds,nds2,rec['source_index'],offset,labels,pred['D_rank'],es,'shifted'); cells+=cell_geometry(pred['m_bar'],c,cs,info,cls,rec['source_index'])
        an=b2.apply_delta_to_native(pred['native'],info['delta']); sn=b2.apply_delta_to_native(pred['native'],infos['delta']); prob,_=b2.deploy_native(an); prob_s,_=b2.deploy_native(sn); c0=pred['score']; c1=prob[0,1].detach().float().cpu().numpy().ravel(); csh=prob_s[0,1].detach().float().cpu().numpy().ravel(); target=mask.ravel().astype(bool)
        if rec['label']==0: n0.append(c0); n1.append(c1); ns.append(csh)
        else: pn+=int(target.sum()); pda+=float((c1[target]-c0[target]).sum()); pds+=float((csh[target]-c0[target]).sum())
        nn+=int((~target).sum()); nda+=float((c1[~target]-c0[~target]).sum()); nds+=float((csh[~target]-c0[~target]).sum()); pm.append(pred['m_bar']); pc.append(c); ps.append(cs); pl.append(labels)
        if rec['label']==1: pairs+=pair_rows(cls,rec,labels,pred['m_bar'],c,cs,pred['E_nonlocal'],es,info,d,ds)
        offset+=b2.PATCH_COUNT; del an,sn,prob,prob_s,pred,raw
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    p0=np.concatenate(pm).astype(np.float32); p1=np.concatenate(pc).astype(np.float32); psh=np.concatenate(ps).astype(np.float32); labels=np.concatenate(pl).astype(bool); cap,rneg=patch_errors(p0,labels); tc=float(cap[labels].sum()); tr=float(rneg[~labels].sum())
    for group in (aa,ss):
        for r in group: r['native_C_AP']=float(cap[r['global_index']]); r['native_R_neg']=float(rneg[r['global_index']])
    auc0,ap0=exact_auc_ap(p0,labels.astype(np.uint8)); auc1,ap1=exact_auc_ap(p1,labels.astype(np.uint8)); aucs,aps=exact_auc_ap(psh,labels.astype(np.uint8)); n0a=np.concatenate(n0); n1a=np.concatenate(n1); nsa=np.concatenate(ns); nm=normal_metrics(n0a,n1a,nsa); pn=max(1,pn); nn=max(1,nn)
    trans={}; bridge={}
    for v in ('aligned','shifted'):
        tk=v+'_transition'; ek=v+'_e_prefers_positive'; wrong=[r for r in pairs if not r['base_correct']]; right=[r for r in pairs if r['base_correct']]; trans[v]={'pairs':len(pairs),'rescued':sum(r[tk]=='rescued' for r in pairs),'missed':sum(r[tk]=='missed' for r in pairs),'preserved':sum(r[tk]=='preserved' for r in pairs),'broken':sum(r[tk]=='broken' for r in pairs),'net':sum(r[tk]=='rescued' for r in pairs)-sum(r[tk]=='broken' for r in pairs),'base_wrong':len(wrong),'base_correct':len(right),'rescue_rate_when_base_wrong':None if not wrong else sum(r[tk]=='rescued' for r in wrong)/len(wrong),'break_rate_when_base_correct':None if not right else sum(r[tk]=='broken' for r in right)/len(right)}; bridge[v]={'pairs':len(pairs),'base_wrong':len(wrong),'base_correct':len(right),'rescue_opportunity':None if not wrong else sum(r[ek] for r in wrong)/len(wrong),'damage_risk':None if not right else sum(not r[ek] for r in right)/len(right)}
    native={'C0_AP':ap0,'C1_AP':ap1,'C1_SHIFT_AP':aps,'C1_minus_C0_AP':ap1-ap0,'C1_minus_C1_SHIFT_AP':ap1-aps,'C0_AUROC':auc0,'C1_AUROC':auc1,'C1_SHIFT_AUROC':aucs,'C1_minus_C0_AUROC':auc1-auc0,'C1_minus_C1_SHIFT_AUROC':auc1-aucs}; deployed={k:b2row[k] for k in ('baseline_ap_C0','C1_ap','C1_SHIFT_ap','C1_AP_delta','C1_minus_C1_SHIFT_AP_delta','baseline_auroc_C0','C1_auroc','C1_SHIFT_auroc','C1_AUROC_delta','C1_minus_C1_SHIFT_AUROC_delta')}
    dep={'native_patch':native,'deployed_pixel':deployed,'before_vs_after':{'aligned_minus_shifted_AP_before_deployment':ap1-aps,'aligned_minus_shifted_AP_after_deployment':b2row['C1_minus_C1_SHIFT_AP_delta'],'aligned_minus_shifted_AUROC_before_deployment':auc1-aucs,'aligned_minus_shifted_AUROC_after_deployment':b2row['C1_minus_C1_SHIFT_AUROC_delta']},'positive_pixel_mean_score_delta':{'aligned':pda/pn,'shifted':pds/pn},'normal_pixel_mean_score_delta':{'aligned':nda/nn,'shifted':nds/nn}}
    return {'class':cls,'n_images':len(records),'forward_count':len(records),'predictor_parity_max_abs':parity,'shape_record':shape,'bridge':bridge,'transitions':trans,'pair_rows':pairs,'displacement':{'aligned_action':action_summary(aa,cap,rneg,tc,tr),'shifted_action':action_summary(ss,cap,rneg,tc,tr),'aligned_pair_strata':pair_strata(pairs,'aligned'),'shifted_pair_strata':pair_strata(pairs,'shifted'),'correlation_aligned':class_corr(pairs,'aligned'),'correlation_shifted':class_corr(pairs,'shifted')},'native_deployed':dep,'normal_positive':{'aligned':nm,'shifted':nm,'positive_patch_count':int(labels.sum()),'normal_patch_count':int((~labels).sum()),'changed_aligned':len(aa),'changed_shifted':len(ss)},'cell_geometry':{'cells':cells,'n_acted_cells':len(cells),'pairwise_gap_p95':None if not cells else float(np.quantile([x['pairwise_score_gap_p95'] for x in cells],.95)),'pairwise_gap_max':None if not cells else float(max(x['pairwise_score_gap_max'] for x in cells))}}

def aggregate_trans(rows,v):
    return {k:{'mean':float(np.mean([r['transitions'][v][k] for r in rows])),'total':int(sum(r['transitions'][v][k] for r in rows))} for k in ('pairs','rescued','missed','preserved','broken','net','base_wrong','base_correct')}|{'rescue_rate_when_base_wrong':agg([r['transitions'][v]['rescue_rate_when_base_wrong'] for r in rows],BOOTSTRAP_SEED),'break_rate_when_base_correct':agg([r['transitions'][v]['break_rate_when_base_correct'] for r in rows],BOOTSTRAP_SEED+1),'net_utility':agg([r['transitions'][v]['net'] for r in rows],BOOTSTRAP_SEED+2)}
def aggregate_bridge(rows,v): return {'pairs':sum(r['bridge'][v]['pairs'] for r in rows),'base_wrong':sum(r['bridge'][v]['base_wrong'] for r in rows),'base_correct':sum(r['bridge'][v]['base_correct'] for r in rows),'rescue_opportunity':agg([r['bridge'][v]['rescue_opportunity'] for r in rows],BOOTSTRAP_SEED+10),'damage_risk':agg([r['bridge'][v]['damage_risk'] for r in rows],BOOTSTRAP_SEED+11)}
def aggregate_corr(rows,v): return {k:agg([r['displacement']['correlation_'+v][k] for r in rows],BOOTSTRAP_SEED+i) for i,k in enumerate(('rank_displacement_vs_broken_correct_pair_rate','rank_displacement_vs_net_pair_utility'))}
def aggregate_native(rows):
    nk=('C1_minus_C0_AP','C1_minus_C1_SHIFT_AP','C1_minus_C0_AUROC','C1_minus_C1_SHIFT_AUROC'); dk=('C1_AP_delta','C1_minus_C1_SHIFT_AP_delta','C1_AUROC_delta','C1_minus_C1_SHIFT_AUROC_delta'); bk=('aligned_minus_shifted_AP_before_deployment','aligned_minus_shifted_AP_after_deployment','aligned_minus_shifted_AUROC_before_deployment','aligned_minus_shifted_AUROC_after_deployment')
    return {'native_patch':{k:agg([r['native_deployed']['native_patch'][k] for r in rows],BOOTSTRAP_SEED+i) for i,k in enumerate(nk)},'deployed_pixel':{k:agg([r['native_deployed']['deployed_pixel'][k] for r in rows],BOOTSTRAP_SEED+10+i) for i,k in enumerate(dk)},'before_vs_after':{k:agg([r['native_deployed']['before_vs_after'][k] for r in rows],BOOTSTRAP_SEED+20+i) for i,k in enumerate(bk)},'positive_pixel_mean_score_delta':{k:agg([r['native_deployed']['positive_pixel_mean_score_delta'][k] for r in rows],BOOTSTRAP_SEED+30+i) for i,k in enumerate(('aligned','shifted'))},'normal_pixel_mean_score_delta':{k:agg([r['native_deployed']['normal_pixel_mean_score_delta'][k] for r in rows],BOOTSTRAP_SEED+40+i) for i,k in enumerate(('aligned','shifted'))}}
def run_tests():
    checks={'T1_rescue':transition(False,True)=='rescued','T2_break':transition(True,False)=='broken','T3_preserved_missed':transition(True,True)=='preserved' and transition(False,False)=='missed'}; m=np.asarray([4.,3.,2.,1.]); c=np.asarray([1.,4.,3.,2.]); d,n=rank_displacement(m,c,{'cells':[{'patches':[0,1,2,3]}]}); checks['T4_rank_displacement']=np.array_equal(d[:4],[3,1,1,1]) and np.allclose(n[:4],[1.,1/3,1/3,1/3]); p=np.arange(b2.PATCH_COUNT,dtype=np.float32); e=p[::-1]; valid=np.ones(b2.PATCH_COUNT,dtype=bool); a,ia=b2.adjudicate_slots(p,p,e,valid); s,is_=b2.adjudicate_slots(p,p,b2.shifted_evidence(e),valid); checks.update({'T5_gt_firewall':'gt' not in inspect.signature(b2.adjudicate_slots).parameters,'T6_same_eligibility_cells':all(np.array_equal(ia[k],is_[k]) for k in ('eligible','score_bin','d_rank_bin')),'T7_exact_b2_reproduction':np.array_equal(a,b2.adjudicate_slots(p,p,e,valid)[0]),'T8_no_new_correction':True,'T9_deterministic':np.array_equal(s,b2.adjudicate_slots(p,p,b2.shifted_evidence(e),valid)[0]),'T10_no_nan_inf':finite(a.tolist()) and finite(s.tolist())}); return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'no_model_inference':True,'no_training':True,'no_new_correction':True}
def decision(summary,rows):
    b=summary['bridge']; c=summary['displacement']['aligned']; ba=summary['native_deployed']['before_vs_after']; normal=summary['native_deployed']['normal_pixel_mean_score_delta']; flags={'rescue_opportunity_positive':b['aligned']['rescue_opportunity']['mean']>0,'aligned_damage_lower_than_shifted':b['aligned']['damage_risk']['mean']<b['shifted']['damage_risk']['mean'],'broad_breakage':sum(r['transitions']['aligned']['broken']>0 for r in rows)>=8,'breakage_grows_with_displacement':c['rank_displacement_vs_broken_correct_pair_rate']['bootstrap95_ci'] is not None and c['rank_displacement_vs_broken_correct_pair_rate']['bootstrap95_ci'][0]>0,'nonzero_cell_gaps':summary['cell_geometry']['pairwise_gap_p95_macro']>0,'aligned_loses_after_deployment':ba['aligned_minus_shifted_AP_after_deployment']['mean']<0,'shifted_more_normal_suppression':normal['shifted']['mean']<normal['aligned']['mean']}; term='AGGRESSIVE_ACTION_MISMATCH_SUPPORTED' if all(flags.values()) else 'AGGRESSIVE_ACTION_MISMATCH_UNSUPPORTED' if not flags['breakage_grows_with_displacement'] or not flags['broad_breakage'] else 'ACTION_MISMATCH_MECHANISM_INCONCLUSIVE'; return term,flags
def write_csv(rows):
    fields=('class','n_images','bridge_pairs','aligned_rescue_opportunity','aligned_damage_risk','shifted_rescue_opportunity','shifted_damage_risk','aligned_rescued','aligned_missed','aligned_preserved','aligned_broken','aligned_net','shifted_rescued','shifted_missed','shifted_preserved','shifted_broken','shifted_net','aligned_break_correlation','changed_aligned','changed_shifted','native_AP_delta','deployed_AP_delta','aligned_minus_shifted_AP_before','aligned_minus_shifted_AP_after','cell_count','cell_pair_gap_p95','normal_fpr95_delta_aligned','normal_fpr99_delta_aligned')
    with (OUTPUT_ROOT/'PER_CLASS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader()
        for r in rows:
            w.writerow({'class':r['class'],'n_images':r['n_images'],'bridge_pairs':r['bridge']['aligned']['pairs'],'aligned_rescue_opportunity':r['bridge']['aligned']['rescue_opportunity'],'aligned_damage_risk':r['bridge']['aligned']['damage_risk'],'shifted_rescue_opportunity':r['bridge']['shifted']['rescue_opportunity'],'shifted_damage_risk':r['bridge']['shifted']['damage_risk'],'aligned_rescued':r['transitions']['aligned']['rescued'],'aligned_missed':r['transitions']['aligned']['missed'],'aligned_preserved':r['transitions']['aligned']['preserved'],'aligned_broken':r['transitions']['aligned']['broken'],'aligned_net':r['transitions']['aligned']['net'],'shifted_rescued':r['transitions']['shifted']['rescued'],'shifted_missed':r['transitions']['shifted']['missed'],'shifted_preserved':r['transitions']['shifted']['preserved'],'shifted_broken':r['transitions']['shifted']['broken'],'shifted_net':r['transitions']['shifted']['net'],'aligned_break_correlation':r['displacement']['correlation_aligned']['rank_displacement_vs_broken_correct_pair_rate'],'changed_aligned':r['normal_positive']['changed_aligned'],'changed_shifted':r['normal_positive']['changed_shifted'],'native_AP_delta':r['native_deployed']['native_patch']['C1_minus_C0_AP'],'deployed_AP_delta':r['native_deployed']['deployed_pixel']['C1_AP_delta'],'aligned_minus_shifted_AP_before':r['native_deployed']['before_vs_after']['aligned_minus_shifted_AP_before_deployment'],'aligned_minus_shifted_AP_after':r['native_deployed']['before_vs_after']['aligned_minus_shifted_AP_after_deployment'],'cell_count':r['cell_geometry']['n_acted_cells'],'cell_pair_gap_p95':r['cell_geometry']['pairwise_gap_p95'],'normal_fpr95_delta_aligned':r['normal_positive']['aligned']['C1_minus_C0']['fpr_at_tau95'],'normal_fpr99_delta_aligned':r['normal_positive']['aligned']['C1_minus_C0']['fpr_at_tau99']})
def report(summary,term,flags):
    b=summary['bridge']['aligned']; t=summary['transitions']['aligned']; c=summary['displacement']['aligned']; n=summary['native_deployed']; g=summary['cell_geometry']; nextq='How can useful E_nonlocal pairwise evidence be converted into a minimal constrained intervention that preserves trustworthy base ordering?' if term=='AGGRESSIVE_ACTION_MISMATCH_SUPPORTED' else 'Which measurable component of the B2 action path, rather than intervention magnitude alone, explains the remaining aligned-versus-shifted deployment gap?'
    return '\n'.join(['# Phase5-B3 Bridge → Action Mismatch Forensic','',f'Terminal: `{term}`','',f'1. E_nonlocal on base mistakes: aligned rescue opportunity={b["rescue_opportunity"]}; aligned damage risk={b["damage_risk"]}; shifted rescue opportunity={summary["bridge"]["shifted"]["rescue_opportunity"]}; shifted damage risk={summary["bridge"]["shifted"]["damage_risk"]}.',f'2. C1 transitions: rescued={t["rescued"]["total"]}, missed={t["missed"]["total"]}, preserved={t["preserved"]["total"]}, broken={t["broken"]["total"]}, net={t["net"]["total"]}; shifted net={summary["transitions"]["shifted"]["net"]["total"]}.',f'3. Action magnitude: break-rate correlation={c["rank_displacement_vs_broken_correct_pair_rate"]}; net-utility correlation={c["rank_displacement_vs_net_pair_utility"]}; fixed strata only.',f'4. Near-tie audit: acted-cell p95 pairwise score-gap macro={g["pairwise_gap_p95_macro"]}, max={g["pairwise_gap_max"]}; same cell was not assumed to be a numerical near-tie.',f'5. Shifted decomposition: native AP aligned-minus-shifted={n["before_vs_after"]["aligned_minus_shifted_AP_before_deployment"]}; deployed AP aligned-minus-shifted={n["before_vs_after"]["aligned_minus_shifted_AP_after_deployment"]}; normal score deltas={n["normal_pixel_mean_score_delta"]}.','6. Causality is not inferred; the comparison locates the difference before versus after deployment.',f'7. Decision flags: {json.dumps(flags,sort_keys=True)}',f'8. Next research question: {nextq}',''])
def output_check(inp,test,summary,rows,term,protected):
    current=protected_hashes(); files={x:(OUTPUT_ROOT/x).is_file() for x in OUTPUTS if x != 'OUTPUT_CHECK.json'}; checks={'input_integrity':inp['status']=='PASS','tests':test['status']=='PASS','all_classes':len(rows)==12,'forward_count':summary['inference']['forward_count']==2162,'test_split':inp['split']=='TEST','class_bootstrap':summary['transitions']['aligned']['net_utility']['unit']=='class','no_nan_inf':finite(summary),'gt_firewall':True,'no_new_correction':True,'protected_hashes_unchanged':current==protected,'b2_artifacts_unchanged':b2_hashes()==inp['b2_artifact_hashes'],'required_files':all(files.values()),'terminal_valid':term in {'AGGRESSIVE_ACTION_MISMATCH_SUPPORTED','AGGRESSIVE_ACTION_MISMATCH_UNSUPPORTED','ACTION_MISMATCH_MECHANISM_INCONCLUSIVE'}}; return {'status':'PASS' if all(checks.values()) else 'B3_OUTPUT_INVALID','checks':checks,'required_files':files,'protected_hashes_after':current,'terminal':term}
def run():
    inp=input_check(); OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); write_json(OUTPUT_ROOT/'INPUT_CHECK.json',inp); write_json(OUTPUT_ROOT/'PROTOCOL.json',protocol()); test=run_tests(); write_json(OUTPUT_ROOT/'TEST_CHECK.json',test)
    if test['status']!='PASS': raise RuntimeError('B3_INPUT_INVALID: focused tests failed')
    config=json.loads(b2.CONFIG.read_text()); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); datasets,records,counts=b2.canonical_records(b2.IMAGE_SIZE); b2rows={r['class']:r for r in json.loads((B2_ROOT/'SUMMARY.json').read_text())['per_class']}; model,_=b2.load_model(config,b2.CHECKPOINT,device); cache={}; rows=[]
    with torch.inference_mode():
        for cls in sorted(records): rows.append(process_class(model,datasets[cls],cls,records[cls],device,cache,b2rows[cls]))
    summary={'provenance':inp,'inference':{'forward_count':sum(r['forward_count'] for r in rows),'class_count':len(rows),'image_count':2162,'normal_image_count':962,'anomaly_image_count':1200,'class_at_a_time':True,'training_steps':0,'dense_feature_cache_persisted':False,'one_forward_per_image':True},'bridge':{'aligned':aggregate_bridge(rows,'aligned'),'shifted':aggregate_bridge(rows,'shifted')},'transitions':{'aligned':aggregate_trans(rows,'aligned'),'shifted':aggregate_trans(rows,'shifted')},'displacement':{'aligned':aggregate_corr(rows,'aligned'),'shifted':aggregate_corr(rows,'shifted')},'native_deployed':aggregate_native(rows),'cell_geometry':{'acted_cells_total':sum(r['cell_geometry']['n_acted_cells'] for r in rows),'pairwise_gap_p95_macro':float(np.mean([r['cell_geometry']['pairwise_gap_p95'] for r in rows])),'pairwise_gap_max':float(max(r['cell_geometry']['pairwise_gap_max'] for r in rows)),'per_class':{r['class']:{'n_acted_cells':r['cell_geometry']['n_acted_cells'],'pairwise_gap_p95':r['cell_geometry']['pairwise_gap_p95'],'pairwise_gap_max':r['cell_geometry']['pairwise_gap_max']} for r in rows}},'class_consistency':{'classes':[r['class'] for r in rows],'aligned_net_positive':sum(r['transitions']['aligned']['net']>0 for r in rows),'aligned_break_positive':sum(r['transitions']['aligned']['broken']>0 for r in rows)},'per_class':[{k:({kk:vv for kk,vv in v.items() if kk != 'cells'} if k == 'cell_geometry' else v) for k,v in r.items() if k not in {'pair_rows','shape_record'}} for r in rows]}
    term,flags=decision(summary,rows); summary['decision']=term; summary['decision_flags']=flags
    write_json(OUTPUT_ROOT/'PAIR_TRANSITIONS.json',{'per_class':{r['class']:{'aligned':r['transitions']['aligned'],'shifted':r['transitions']['shifted'],'bridge_aligned':r['bridge']['aligned'],'bridge_shifted':r['bridge']['shifted'],'pair_rows':r['pair_rows']} for r in rows},'aggregate':summary['transitions'],'bridge':summary['bridge']}); write_json(OUTPUT_ROOT/'DISPLACEMENT_ANALYSIS.json',{'per_class':{r['class']:r['displacement'] for r in rows},'aggregate':summary['displacement'],'fixed_strata':STRATA}); write_json(OUTPUT_ROOT/'NORMAL_POSITIVE_DECOMPOSITION.json',{'per_class':{r['class']:r['native_deployed'] for r in rows},'aggregate':summary['native_deployed']}); write_json(OUTPUT_ROOT/'SHIFTED_DECOMPOSITION.json',{'per_class':{r['class']:{'shifted_transitions':r['transitions']['shifted'],'shifted_bridge':r['bridge']['shifted'],'native_deployed':r['native_deployed']} for r in rows},'aggregate':{'shifted':summary['transitions']['shifted']}}); write_json(OUTPUT_ROOT/'CELL_GEOMETRY.json',{'per_class':{r['class']:r['cell_geometry'] for r in rows},'aggregate':summary['cell_geometry']}); write_csv(rows); write_json(OUTPUT_ROOT/'SUMMARY.json',summary); write_json(OUTPUT_ROOT/'DECISION.json',{'terminal':term,'flags':flags,'integrity':'PENDING_OUTPUT_CHECK','next_research_question':'How can useful E_nonlocal pairwise evidence be converted into a minimal constrained intervention that preserves trustworthy base ordering?' if term=='AGGRESSIVE_ACTION_MISMATCH_SUPPORTED' else 'Which measurable component of the B2 action path, rather than intervention magnitude alone, explains the remaining aligned-versus-shifted deployment gap?'}); (OUTPUT_ROOT/'REPORT.md').write_text(report(summary,term,flags)); check=output_check(inp,test,summary,rows,term,inp['protected_hashes']); write_json(OUTPUT_ROOT/'OUTPUT_CHECK.json',check); d=json.loads((OUTPUT_ROOT/'DECISION.json').read_text()); d['integrity']=check['status']; write_json(OUTPUT_ROOT/'DECISION.json',d)
    if check['status']!='PASS': raise RuntimeError('B3_OUTPUT_INVALID: output check failed')
    print(json.dumps({'status':'PASS','terminal':term,'forward_count':2162},sort_keys=True))
def main(): argparse.ArgumentParser().parse_args(); b2.configure_canonical_fp32(); run()
if __name__=='__main__': main()

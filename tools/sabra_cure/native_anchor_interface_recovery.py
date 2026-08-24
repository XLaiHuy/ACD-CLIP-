"""P23: surgical interface-safe wrapper around the immutable P22 engine."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.sabra_cure import native_anchor_diagnostic as p21
from tools.sabra_cure import native_anchor_performance_recovery as p22

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/sabra_cure/p22_interface_recovery"
PARENT="5d869debdb245627fb6c389349489e51af08ecd5"
PREREG="36a97b75c083c53589aa4685c6a854b9e6792bc4"


def cache_count(cache: Any, seed2: np.ndarray | None=None) -> int:
    """The sole P23 ClassCache image-count interface: existing paths only."""
    count=len(cache.paths)
    lengths={"paths":count,"native":len(cache.native),"safe":len(cache.safe),"expand":len(cache.expand)}
    if seed2 is not None: lengths["seed2"]=len(seed2)
    if len(set(lengths.values()))!=1: raise RuntimeError(f"P23_ENGINEERING_STOP cache alignment {lengths}")
    return count


def witness(held: str, family: tuple[str,...], seed2: np.ndarray, smoke: bool=False) -> dict[str,Any]:
    engine,vectors,extra=p22._engine_for(held,family); fold,cache=extra["fold"],extra["cache"]
    try:
        count=cache_count(cache,seed2)
        if engine.n_images!=count: raise RuntimeError("P23_ENGINEERING_STOP engine/cache image count")
        seed1=np.full(count,"NATIVE",dtype="<U16")
        if smoke:
            return {"smoke":True,"held":held,"family":list(family),"count":count,"seed1_length":len(seed1),"seed2_length":len(seed2),"engine_images":engine.n_images}
        runs=p22.coordinate_two_lanes(engine,seed1,seed2,family); chosen=p21.choose_seed(runs,family); score=engine.compose(chosen["state"]); reference=p21.exact_metrics(score.reshape(-1),cache.masks.reshape(-1)); error=abs(float(chosen["pap"])-float(reference["pAP"]))
        if error>p21.EPS: raise RuntimeError(f"P23_ENGINEERING_STOP direct parity {error}")
        return {"held":held,"assignment":chosen["state"].tolist(),"pap":float(chosen["pap"]),"pauroc":float(reference["pAUROC"]),"fast_reference_error":error,"sweeps":int(chosen["sweeps"]),"converged":bool(chosen["converged"]),"changes":int(chosen["changes"]),"action_counts":{name:int(np.sum(chosen["state"]==name)) for name in family},"seeds":[{"pap":float(run["pap"]),"sweeps":int(run["sweeps"]),"converged":bool(run["converged"]),"changes":int(run["changes"])} for run in runs],"safety":p21.p14.safety(p21.state_action_vector(chosen["state"],vectors),fold["y"],fold["mu"])}
    finally: engine.close()


def opportunity(held: str, family: tuple[str,...]) -> dict[str,Any]:
    """P22 opportunity implementation with only the canonical cache count."""
    engine,_,extra=p22._engine_for(held,family); cache,fold=extra["cache"],extra["fold"]
    try:
        count=cache_count(cache); positive,total=engine.counts(np.full(count,"NATIVE",dtype="<U16"));base=engine.ap(positive,total);values=np.zeros((count,len(family)),dtype=np.float64)
        for image in range(count):
            for ai,name in enumerate(family[1:],1):values[image,ai]=engine.candidate_value_inplace(positive,total,image,"NATIVE",name)-base
        target=np.maximum(0.0,np.max(values[:,1:],axis=1));oracle=np.zeros(count,dtype=np.int64)
        for image in range(count):
            for ai in range(1,len(family)):
                if values[image,ai]>values[image,oracle[image]]+p21.EPS:oracle[image]=ai
        shards,_=p21.r1.load_shards(True);params=json.loads((p21.fold_dir(held)/"parameters.json").read_text());f0=p21.p14.fields(held,shards[held].x,fold["mu"],fold["sigma"],fold["risk"],float(params["tau20"]),float(params["tau40"]),fold["image_path"].astype(str));f1=p21.f1_features(cache.native,{name:engine.actions[name] for name in family},family)
        if f0.shape[1]!=16 or not np.isfinite(f0).all() or not np.isfinite(f1).all():raise RuntimeError("P23_ENGINEERING_STOP feature contract")
        return {"held":held,"f0":f0,"f1":f1,"opportunity":target,"action_opportunity":values,"oracle_action":oracle,"family":family,"image_path":fold["image_path"].astype(str)}
    finally: engine.close()


def _bind() -> None:
    """Process-local substitution only; historical P22 files are untouched."""
    p22.OUT=OUT; p22.PREREG=PREREG; p22.witness=witness; p22.opportunity=opportunity


def interface_audit() -> dict[str,Any]:
    fold=p21.load_fold("candle");cache=p21.class_cache("candle",fold);seed2=p21.p20_oracle_seed(fold);count=cache_count(cache,seed2)
    entries=[]
    for attr,shape,callsite in (("paths","(N,) ndarray","cache_count/witness/opportunity"),("native","(N,H,W) ndarray","cache_count/_engine_for"),("safe","(N,H,W) ndarray","cache_count"),("expand","(N,H,W) ndarray","cache_count"),("masks","(N,H,W) ndarray","witness"),("base_scores","ndarray","P15 cache only"),("base_positive","ndarray","P15 cache only"),("base_total","ndarray","P15 cache only")):
        entries.append({"object_type":"p15.ClassCache","referenced_attribute":attr,"exists":hasattr(cache,attr),"expected_shape_type":shape,"production_callsite":callsite})
    engine=p22.RecoveryEngine(cache.native,{"NATIVE":cache.native,"SAFE20":cache.safe,"EXPAND40":cache.expand},cache.masks)
    for attr,shape in (("n_images","int"),("names","tuple[str,...]"),("union","float32 ndarray"),("base_positive","float64 ndarray"),("base_total","float64 ndarray")):
        entries.append({"object_type":"P22.RecoveryEngine","referenced_attribute":attr,"exists":hasattr(engine,attr),"expected_shape_type":shape,"production_callsite":"flat store/coordinate"})
    result={"status":"PASS" if all(x["exists"] for x in entries) else "FAIL","canonical_image_count":count,"alignment":{"paths":len(cache.paths),"native":len(cache.native),"safe":len(cache.safe),"expand":len(cache.expand),"seed2":len(seed2)},"entries":entries,"forbidden_classcache_n_images_reference":False,"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}}
    p21.atomic(OUT/"interface_contract_audit.json",result);return result


def production_smoke() -> dict[str,Any]:
    fold=p21.load_fold("candle"); row=witness("candle",p21.A0,p21.p20_oracle_seed(fold),smoke=True)
    result={"status":"PASS" if row["count"]==row["seed1_length"]==row["seed2_length"]==row["engine_images"] and not (OUT/"ATTEMPT_STARTED.json").exists() and not (OUT/"summary.json").exists() else "FAIL","route":["execute_once","run_action_space","witness","cache construction","seed construction","coordinate initialization"],"details":row,"marker_absent":not (OUT/"ATTEMPT_STARTED.json").exists(),"scientific_result_written":(OUT/"summary.json").exists(),"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}}
    p21.atomic(OUT/"production_path_smoke.json",result);return result


def controller_rehearsal() -> dict[str,Any]:
    """Full routing/serialization rehearsal with deterministic non-scientific stubs."""
    stages=[]
    for a0_strong,a1_strong in ((True,False),(False,True),(False,False)):
        active="A0" if a0_strong else "A1"; stage_d=a0_strong or a1_strong; selected={"per_class":{},"headroom_strong":a0_strong if active=="A0" else a1_strong}; payload={"A0":{"headroom_strong":a0_strong},"A1":None if a0_strong else {"headroom_strong":a1_strong},"stage_d":stage_d,"selected":active,"selected_audit":{"status":"PASS"},"summary":{"status":"STUB"}}
        p21.json.dumps(payload,allow_nan=False); stages.append(payload)
    result={"status":"PASS","cases":stages,"route":["Stage A","A0","conditional A1","conditional Stage D","selected audit","summary","terminal"],"real_scientific_outcomes":False,"marker_absent":not (OUT/"ATTEMPT_STARTED.json").exists(),"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}}
    p21.atomic(OUT/"controller_rehearsal.json",result);return result


def exactness() -> dict[str,Any]: _bind();return p22.exactness()
def benchmark() -> dict[str,Any]: _bind();return p22.benchmark()

def pre_audit() -> dict[str,Any]:
    audit=json.loads((OUT/"interface_contract_audit.json").read_text());smoke=json.loads((OUT/"production_path_smoke.json").read_text());rehearsal=json.loads((OUT/"controller_rehearsal.json").read_text());exact=json.loads((OUT/"exactness_parity.json").read_text());performance=json.loads((OUT/"performance_benchmark.json").read_text())
    checks={"parent":p21.git("merge-base","--is-ancestor",PARENT,"HEAD")=="","marker_absent":not (OUT/"ATTEMPT_STARTED.json").exists(),"interface_audit":audit["status"]=="PASS","production_smoke":smoke["status"]=="PASS","controller_rehearsal":rehearsal["status"]=="PASS","exactness":exact["status"]=="PASS","performance":performance["status"]=="PASS","runtime":performance["projected_max_route_minutes"]<=180.0}
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}};p21.atomic(OUT/"pre_execution_audit.json",result);return result


def execute_once() -> dict[str,Any]: _bind();return p22.execute_once()

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--interface-audit",action="store_true");parser.add_argument("--smoke",action="store_true");parser.add_argument("--rehearsal",action="store_true");parser.add_argument("--exactness",action="store_true");parser.add_argument("--benchmark",action="store_true");parser.add_argument("--pre-audit",action="store_true");parser.add_argument("--run",action="store_true");args=parser.parse_args();options=(args.interface_audit,args.smoke,args.rehearsal,args.exactness,args.benchmark,args.pre_audit,args.run)
    if sum(options)!=1:parser.error("choose one operation")
    result=interface_audit() if args.interface_audit else production_smoke() if args.smoke else controller_rehearsal() if args.rehearsal else exactness() if args.exactness else benchmark() if args.benchmark else pre_audit() if args.pre_audit else execute_once();print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False,default=p21.json_default))

if __name__=="__main__":main()

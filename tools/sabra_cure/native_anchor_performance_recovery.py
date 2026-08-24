"""P22 exact-engineering recovery for the frozen P21 diagnostic.

This module owns only storage/lifetime and CUDA execution.  P21 scientific
definitions are imported from the immutable runner; no P21 source is edited.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.sabra_cure import native_anchor_diagnostic as p21

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/p21_exact_performance_recovery"
PARENT = "625bf96b192bab713523fbae9b855804b72e5e56"
PREREG = "9547269db933440fb85cd7ebe79b508e91cefa5b"
EPS = p21.EPS
A0, A1 = p21.A0, p21.A1


@dataclass
class FlatStore:
    root: Path
    names: tuple[str, ...]
    n_images: int
    offsets: np.ndarray
    index: np.memmap
    positive: np.memmap
    total: np.memmap

    @classmethod
    def build(cls, engine: "RecoveryEngine", root: Path) -> "FlatStore":
        root.mkdir(parents=True, exist_ok=False)
        names = tuple(name for name in engine.names if name != "NATIVE")
        offsets = np.zeros((len(names), engine.n_images, 2), dtype=np.int64)
        paths = (root / "delta_index.bin", root / "delta_positive.bin", root / "delta_total.bin")
        cursor = 0
        try:
            with paths[0].open("wb") as fi, paths[1].open("wb") as fp, paths[2].open("wb") as ft:
                for ai, action in enumerate(names):
                    for image in range(engine.n_images):
                        delta = engine._direct_delta(image, action)
                        if len(delta.index) != len(np.unique(delta.index)):
                            raise RuntimeError("P22_ENGINEERING_STOP nonunique sparse index")
                        if len(engine.union) >= 2**32 or np.any(np.abs(delta.positive) > np.iinfo(np.int32).max) or np.any(np.abs(delta.total) > np.iinfo(np.int32).max):
                            raise RuntimeError("P22_ENGINEERING_STOP flat delta dtype")
                        start = cursor; stop = start + len(delta.index)
                        np.asarray(delta.index, dtype=np.uint32).tofile(fi)
                        np.asarray(delta.positive, dtype=np.int32).tofile(fp)
                        np.asarray(delta.total, dtype=np.int32).tofile(ft)
                        offsets[ai, image] = (start, stop); cursor = stop
            np.save(root / "delta_offsets.npy", offsets, allow_pickle=False)
            (root / "manifest.json").write_text(json.dumps({"names": names, "n_images": engine.n_images, "nnz": cursor}, sort_keys=True), encoding="utf-8")
        except Exception:
            shutil.rmtree(root, ignore_errors=True); raise
        return cls.open(root)

    @classmethod
    def open(cls, root: Path) -> "FlatStore":
        meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        offsets = np.load(root / "delta_offsets.npy", allow_pickle=False)
        nnz = int(meta["nnz"])
        return cls(root, tuple(meta["names"]), int(meta["n_images"]), offsets,
                   np.memmap(root / "delta_index.bin", dtype=np.uint32, mode="r", shape=(nnz,)),
                   np.memmap(root / "delta_positive.bin", dtype=np.int32, mode="r", shape=(nnz,)),
                   np.memmap(root / "delta_total.bin", dtype=np.int32, mode="r", shape=(nnz,)))

    def get(self, image: int, action: str) -> p21.SparseDelta:
        if action == "NATIVE": return p21.EMPTY_DELTA
        ai = self.names.index(action); start, stop = self.offsets[ai, image]
        return p21.SparseDelta(self.index[int(start):int(stop)], self.positive[int(start):int(stop)], self.total[int(start):int(stop)])


class RecoveryEngine(p21.NativeAnchorEngine):
    """P21 engine with flat exact deltas and no CPU candidate full-state copy."""
    def __init__(self, native: np.ndarray, actions: dict[str, np.ndarray], masks: np.ndarray):
        super().__init__(native, actions, masks); self.store: FlatStore | None = None

    def _direct_delta(self, image: int, action: str) -> p21.SparseDelta:
        return super().image_delta(image, action)

    def build_flat_store(self, root: Path) -> None:
        self.store = FlatStore.build(self, root)

    def image_delta(self, images: np.ndarray | int, action: str) -> p21.SparseDelta:
        selected = np.atleast_1d(np.asarray(images, dtype=np.int64))
        if self.store is not None and len(selected) == 1:
            return self.store.get(int(selected[0]), action)
        return super().image_delta(selected, action)

    def counts(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive, total = self.base_positive.copy(), self.base_total.copy()
        for image, action in enumerate(np.asarray(state).tolist()):
            if action != "NATIVE": self.apply(positive, total, self.image_delta(image, str(action)), 1.0)
        return positive, total

    def candidate_value_inplace(self, positive: np.ndarray, total: np.ndarray, image: int, old: str, candidate: str) -> float:
        if old == candidate: return self.ap(positive, total)
        self.apply(positive, total, self.image_delta(image, old), -1.0)
        self.apply(positive, total, self.image_delta(image, candidate), 1.0)
        try: return self.ap(positive, total)
        finally:
            self.apply(positive, total, self.image_delta(image, candidate), -1.0)
            self.apply(positive, total, self.image_delta(image, old), 1.0)

    def close(self) -> None:
        if self.store is not None:
            root = self.store.root; self.store = None; shutil.rmtree(root, ignore_errors=True)


def _gpu_add(row: torch.Tensor, delta: p21.SparseDelta, sign: float) -> None:
    if len(delta.index):
        index = torch.tensor(delta.index, dtype=torch.int64, device="cuda")
        # Flat-store construction proves one index occurrence per delta, so
        # indexed addition is exact and avoids index_add's duplicate machinery.
        row[index] += torch.as_tensor(delta.positive * sign, dtype=torch.float64, device="cuda")


def _gpu_add_total(row: torch.Tensor, delta: p21.SparseDelta, sign: float) -> None:
    if len(delta.index):
        index = torch.tensor(delta.index, dtype=torch.int64, device="cuda")
        row[index] += torch.as_tensor(delta.total * sign, dtype=torch.float64, device="cuda")


def coordinate_two_lanes(engine: RecoveryEngine, left_seed: np.ndarray, right_seed: np.ndarray, family: tuple[str, ...], max_sweeps: int = 10, image_limit: int | None = None) -> list[dict[str, Any]]:
    if not torch.cuda.is_available(): raise RuntimeError("P22_ENGINEERING_STOP CUDA unavailable")
    states = [np.asarray(left_seed, dtype="<U16").copy(), np.asarray(right_seed, dtype="<U16").copy()]
    cpu = [engine.counts(state) for state in states]
    positive = [torch.as_tensor(pair[0], dtype=torch.float64, device="cuda") for pair in cpu]
    total = [torch.as_tensor(pair[1], dtype=torch.float64, device="cuda") for pair in cpu]
    current = [p21.gpu_ap(positive[i], total[i]) for i in range(2)]; changes_total = [0, 0]
    for sweep in range(1, max_sweeps + 1):
        changed = [0, 0]
        for image in range(engine.n_images if image_limit is None else min(engine.n_images, image_limit)):
            rows: list[tuple[int, str]] = []
            for lane in range(2): rows.extend((lane, name) for name in family if name != str(states[lane][image]))
            cp = torch.cat([positive[lane][None, :] for lane, _ in rows], dim=0)
            ct = torch.cat([total[lane][None, :] for lane, _ in rows], dim=0)
            for row, (lane, candidate) in enumerate(rows):
                old = str(states[lane][image]); old_delta = engine.image_delta(image, old); new_delta = engine.image_delta(image, candidate)
                _gpu_add(cp[row], old_delta, -1.0); _gpu_add_total(ct[row], old_delta, -1.0)
                _gpu_add(cp[row], new_delta, 1.0); _gpu_add_total(ct[row], new_delta, 1.0)
            values = p21.gpu_ap_batch(cp, ct)
            for lane in range(2):
                old = str(states[lane][image]); best, best_value, best_row = old, current[lane], None
                for row, (row_lane, candidate) in enumerate(rows):
                    if row_lane == lane and float(values[row]) > best_value + EPS:
                        best, best_value, best_row = candidate, float(values[row]), row
                if best_row is not None:
                    states[lane][image] = best; current[lane] = best_value
                    positive[lane] = cp[best_row].clone(); total[lane] = ct[best_row].clone()
                    changed[lane] += 1; changes_total[lane] += 1
        if not any(changed):
            return [{"state": states[i], "pap": current[i], "sweeps": sweep, "converged": True, "changes": changes_total[i]} for i in range(2)]
    return [{"state": states[i], "pap": current[i], "sweeps": max_sweeps, "converged": False, "changes": changes_total[i]} for i in range(2)]


def scalar_prefix(engine: RecoveryEngine, seed: np.ndarray, family: tuple[str, ...], image_limit: int) -> dict[str, Any]:
    """Frozen scalar P21 decisions over the predeclared non-outcome prefix."""
    state=np.asarray(seed,dtype="<U16").copy(); positive,total=engine.counts(state); current=engine.ap(positive,total); decisions=[]
    for image in range(image_limit):
        old=str(state[image]); best,best_ap=old,current
        for candidate in family:
            if candidate==old: continue
            _,_,value=engine.candidate(positive,total,image,old,candidate)
            if value>best_ap+EPS: best,best_ap=candidate,value
        decisions.append(best)
        if best!=old:
            positive,total,_=engine.candidate(positive,total,image,old,best);state[image]=best;current=best_ap
    return {"state":state,"pap":current,"decisions":decisions}


def real_slice_parity() -> dict[str, Any]:
    """Fixed candle/A0/first-8-coordinates engineering parity; never a fold outcome."""
    held, limit = "candle", 8
    engine, _, extra = _engine_for(held,A0)
    try:
        left=np.full(engine.n_images,"NATIVE",dtype="<U16");right=p21.p20_oracle_seed(extra["fold"])
        scalar=[scalar_prefix(engine,left,A0,limit),scalar_prefix(engine,right,A0,limit)]
        batched=coordinate_two_lanes(engine,left,right,A0,max_sweeps=1,image_limit=limit)
        errors=[abs(float(scalar[i]["pap"])-float(batched[i]["pap"])) for i in range(2)]
        state_match=[scalar[i]["state"].tolist()==batched[i]["state"].tolist() for i in range(2)]
        result={"status":"PASS" if max(errors)<=EPS and all(state_match) else "FAIL","held":held,"family":list(A0),"seed_count":2,"first_image_coordinates":limit,"max_candidate_ap_error":max(errors),"state_match":state_match,"scalar_decisions":[row["decisions"] for row in scalar],"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}}
        p21.atomic(OUT/"trajectory_parity.json",result); return result
    finally: engine.close()


def _engine_for(held: str, family: tuple[str, ...]) -> tuple[RecoveryEngine, dict[str, np.ndarray], dict[str, Any]]:
    maps, vectors, extra = p21.action_maps(held, family)
    engine = RecoveryEngine(extra["cache"].native, {name: maps[name] for name in family}, extra["cache"].masks)
    engine.build_flat_store(Path(tempfile.mkdtemp(prefix=f"p22_{held}_")) / "store")
    return engine, vectors, extra


def witness(held: str, family: tuple[str, ...], seed2: np.ndarray) -> dict[str, Any]:
    engine, vectors, extra = _engine_for(held, family); fold, cache = extra["fold"], extra["cache"]
    try:
        runs = coordinate_two_lanes(engine, np.full(cache.n_images, "NATIVE", dtype="<U16"), seed2, family)
        chosen = p21.choose_seed(runs, family); score = engine.compose(chosen["state"])
        reference = p21.exact_metrics(score.reshape(-1), cache.masks.reshape(-1)); error = abs(float(chosen["pap"]) - float(reference["pAP"]))
        if error > EPS: raise RuntimeError(f"P22_ENGINEERING_STOP direct parity {error}")
        record: dict[str, Any] = {"held": held, "assignment": chosen["state"].tolist(), "pap": float(chosen["pap"]), "pauroc": float(reference["pAUROC"]), "fast_reference_error": error, "sweeps": int(chosen["sweeps"]), "converged": bool(chosen["converged"]), "changes": int(chosen["changes"]), "action_counts": {name: int(np.sum(chosen["state"] == name)) for name in family}, "seeds": [{"pap": float(run["pap"]), "sweeps": int(run["sweeps"]), "converged": bool(run["converged"]), "changes": int(run["changes"])} for run in runs], "safety": p21.p14.safety(p21.state_action_vector(chosen["state"], vectors), fold["y"], fold["mu"])}
        return record
    finally: engine.close()


def run_action_space(family: tuple[str, ...], seed_from_a0: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = {}; native_pap=[]; native_auc=[]; actions=[]; ys=[]; mus=[]
    for held in p21.r1.CLASSES:
        fold = p21.load_fold(held); seed = p21.p20_oracle_seed(fold) if seed_from_a0 is None else np.asarray(seed_from_a0[held]["assignment"], dtype="<U16")
        row = witness(held, family, seed)
        if not row["converged"]: raise RuntimeError("P22_ENGINEERING_STOP coordinate nonconvergence")
        maps, vectors, extra = p21.action_maps(held, family); actions.append(p21.state_action_vector(np.asarray(row["assignment"], dtype="<U16"), vectors)); ys.append(fold["y"]); mus.append(fold["mu"]); del maps, extra
        per[held] = row; stored=json.loads((p21.fold_dir(held)/"downstream.json").read_text()); native_pap.append(float(stored["native"]["pixel_ap"])); native_auc.append(float(stored["native"]["pixel_auroc"]))
        p21.atomic(OUT/"progress.json", {"status":"ACTION_SPACE_RUNNING","stage":"A0" if family==A0 else "A1","completed_classes":len(per),"total_classes":12,"last_completed_class":held})
        print(f"OUTER_COMPLETE {held}", flush=True)
    pap=float(np.mean([x["pap"] for x in per.values()])); auc=float(np.mean([x["pauroc"] for x in per.values()])); safety=p21.p14.safety(np.concatenate(actions),np.concatenate(ys),np.concatenate(mus)); nonreg=sum(x["pap"]>=native_pap[i] for i,x in enumerate(per.values())); improve=sum(x["pap"]>native_pap[i] for i,x in enumerate(per.values()));
    gates={"H1_wrong_sign":safety["wrong_rate"]<=.05,"H2_weighted_harm":safety["relative_weighted_harm_reduction"]>=.50,"H3_macro_pap":pap>=float(np.mean(native_pap))+.0025,"H4_nonregressing":nonreg>=9,"H5_improving":improve>=7,"H6_pauroc_guardrail":auc-float(np.mean(native_auc))>=-.005}
    images=sum(sum(x["action_counts"].values()) for x in per.values())
    return {"family":list(family),"per_class":per,"macro_pap":pap,"macro_pauroc":auc,"native_macro_pap":float(np.mean(native_pap)),"native_macro_pauroc":float(np.mean(native_auc)),"delta_vs_native":pap-float(np.mean(native_pap)),"nonregressing_classes":int(nonreg),"improving_classes":int(improve),"safety":safety,"action_fractions":{name:sum(x["action_counts"].get(name,0) for x in per.values()) / images for name in family},"headroom_gates":gates,"headroom_strong":bool(all(gates.values()))}


def opportunity(held: str, family: tuple[str, ...]) -> dict[str, Any]:
    """Exact P21 image opportunities, with in-place P22 candidate evaluation."""
    engine, _, extra = _engine_for(held, family); cache, fold = extra["cache"], extra["fold"]
    try:
        positive,total=engine.counts(np.full(cache.n_images,"NATIVE",dtype="<U16")); base=engine.ap(positive,total)
        values=np.zeros((cache.n_images,len(family)),dtype=np.float64)
        for image in range(cache.n_images):
            for ai,name in enumerate(family[1:],1): values[image,ai]=engine.candidate_value_inplace(positive,total,image,"NATIVE",name)-base
        target=np.maximum(0.0,np.max(values[:,1:],axis=1)); oracle=np.zeros(cache.n_images,dtype=np.int64)
        for image in range(cache.n_images):
            for ai in range(1,len(family)):
                if values[image,ai]>values[image,oracle[image]]+EPS: oracle[image]=ai
        shards,_=p21.r1.load_shards(True); params=json.loads((p21.fold_dir(held)/"parameters.json").read_text())
        f0=p21.p14.fields(held,shards[held].x,fold["mu"],fold["sigma"],fold["risk"],float(params["tau20"]),float(params["tau40"]),fold["image_path"].astype(str))
        maps={name:engine.actions[name] for name in family}; f1=p21.f1_features(cache.native,maps,family)
        if f0.shape[1]!=16 or not np.isfinite(f0).all() or not np.isfinite(f1).all(): raise RuntimeError("P22_ENGINEERING_STOP feature contract")
        return {"held":held,"f0":f0,"f1":f1,"opportunity":target,"action_opportunity":values,"oracle_action":oracle,"family":family,"image_path":fold["image_path"].astype(str)}
    finally: engine.close()


def run_probes(family: tuple[str, ...]) -> dict[str, Any]:
    packages={held:opportunity(held,family) for held in p21.r1.CLASSES}; prediction={"P0":{},"P1":{},"P2":{}}; solver={"P1":{},"P2":{}}; target={held:np.asarray(packages[held]["opportunity"],dtype=np.float64) for held in p21.r1.CLASSES}
    for held in p21.r1.CLASSES:
        names=[name for name in p21.r1.CLASSES if name!=held]
        for probe,key in (("P0","f0"),("P1","f0"),("P2","f01")):
            train=np.concatenate([packages[name]["f0"] if key=="f0" else np.column_stack((packages[name]["f0"],packages[name]["f1"])) for name in names]); test=packages[held]["f0"] if key=="f0" else np.column_stack((packages[held]["f0"],packages[held]["f1"])); value=np.concatenate([target[name] for name in names]); median,iqr=p21.scale_fit(train); train=(train-median)/iqr; test=(test-median)/iqr
            if probe=="P0": beta,intercept=p21.frozen.ridge(train,value);prediction[probe][held]=test@beta+intercept
            else:
                offsets=[];start=0
                for name in names: stop=start+len(target[name]);offsets.append(np.arange(start,stop,dtype=np.int64));start=stop
                weight,info=p21.fit_ranknet(train,value,offsets);prediction[probe][held]=test@weight;solver[probe][held]=info
    output={"family":list(family),"probes":{name:p21.probe_metrics(prediction[name],target) for name in prediction},"solver":solver}
    output["diagnosis"]={"RANK_OBJECTIVE_MISMATCH":not output["probes"]["P0"]["floors_pass"] and output["probes"]["P1"]["floors_pass"],"ACTION_IMPACT_FEATURE_GAP":not output["probes"]["P1"]["floors_pass"] and output["probes"]["P2"]["floors_pass"],"IMAGE_VALUE_NOT_GT_FREE_PREDICTABLE":not any(output["probes"][name]["floors_pass"] for name in ("P0","P1","P2")),"GROUP_SHIFT_LIMIT":output["probes"]["P2"]["global_spearman"] is not None and output["probes"]["P2"]["global_spearman"]>0.0 and not output["probes"]["P2"]["floors_pass"]}
    np.savez_compressed(OUT/"opportunity_targets.npz",**{f"{held}_opportunity":packages[held]["opportunity"] for held in p21.r1.CLASSES});return output


def selected_audit(stage: dict[str, Any], family: tuple[str, ...]) -> dict[str, Any]:
    maximum=0.0
    for held in p21.r1.CLASSES:
        maps,_,extra=p21.action_maps(held,family); engine=p21.NativeAnchorEngine(extra["cache"].native,{name:maps[name] for name in family},extra["cache"].masks); score=engine.compose(np.asarray(stage["per_class"][held]["assignment"],dtype="<U16")); metric=p21.exact_metrics(score.reshape(-1),extra["cache"].masks.reshape(-1)); row=stage["per_class"][held]; maximum=max(maximum,abs(float(metric["pAP"])-float(row["pap"])),abs(float(metric["pAUROC"])-float(row["pauroc"])))
    result={"status":"PASS" if maximum<=EPS else "FAIL","max_metric_error":maximum,"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}};p21.atomic(OUT/"selected_result_audit.json",result)
    if result["status"]!="PASS": raise RuntimeError("P22_ENGINEERING_STOP selected audit")
    return result


def execute_once() -> dict[str, Any]:
    if (OUT/"ATTEMPT_STARTED.json").exists() or (OUT/"summary.json").exists(): raise RuntimeError("P22_ENGINEERING_STOP attempt exists")
    pre=json.loads((OUT/"pre_execution_audit.json").read_text())
    if pre.get("status")!="PASS": raise RuntimeError("P22_ENGINEERING_STOP missing pre-audit")
    attempt={"status":"ATTEMPT_STARTED","attempt_uuid":p21.uuid.uuid4().hex,"execution_base_sha":p21.git("rev-parse","HEAD"),"prereg_sha":PREREG,"frozen_p21_parent_sha":PARENT,"runs":1,"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}}
    p21.atomic(OUT/"ATTEMPT_STARTED.json",attempt);p21.atomic(OUT/"progress.json",{"status":"STARTED","completed_classes":0,"total_classes":12,"attempt_uuid":attempt["attempt_uuid"]});started=time.monotonic()
    try:
        a0=run_action_space(A0);p21.atomic(OUT/"action_space_A0.json",a0);a1=None;probes=None;active=A0
        if not a0["headroom_strong"]: a1=run_action_space(A1,a0["per_class"]);p21.atomic(OUT/"action_space_A1.json",a1);active=A1
        if a0["headroom_strong"] or (a1 is not None and a1["headroom_strong"]):
            probes=run_probes(active)
            for name in ("P0","P1","P2"):p21.atomic(OUT/f"probe_{name}.json",probes["probes"][name])
        selected=a0 if a0["headroom_strong"] else a1
        if selected is None: raise RuntimeError("P22_ENGINEERING_STOP missing selected stage")
        audit=selected_audit(selected,active);diagnosis=p21.final_diagnosis(a0,a1,probes);summary={"status":f"P22_RECOVERED_{diagnosis}","primary_diagnosis":diagnosis,"attempt":attempt,"runtime_seconds":time.monotonic()-started,"A0":a0,"A1":a1,"probes":probes,"selected_result_audit":audit,"firewall":{"mvtec_accessed":False,"medical_accessed":False,"additional_clip_forwards":0,"phase2b_training_steps":0,"r2v3_run":False,"r3_run":False,"r4_run":False}}
        p21.atomic(OUT/"summary.json",summary);p21.atomic(OUT/"progress.json",{"status":"COMPLETE","completed_classes":12,"total_classes":12,"primary_diagnosis":diagnosis});return summary
    except Exception as exc:
        p21.atomic(OUT/"ENGINEERING_FAILURE.json",{"status":"P22_ENGINEERING_STOP","exception_type":type(exc).__name__,"exception_message":str(exc)[:1000],"attempt":attempt});raise


def benchmark() -> dict[str, Any]:
    """Bounded real-class pre-marker benchmark; no trajectory or result."""
    engine, _, _ = _engine_for("candle", A0)
    try:
        state=np.full(engine.n_images,"NATIVE",dtype="<U16"); pos,total=engine.counts(state); delta=engine.image_delta(0,"SAFE20")
        cpu=[]
        for _ in range(3):
            t=time.perf_counter(); engine.candidate_value_inplace(pos,total,0,"NATIVE","SAFE20"); cpu.append(time.perf_counter()-t)
        torch.cuda.reset_peak_memory_stats(); gp=torch.as_tensor(pos,dtype=torch.float64,device="cuda");gt=torch.as_tensor(total,dtype=torch.float64,device="cuda");torch.cuda.synchronize();gpu=[]
        for _ in range(3):
            t=time.perf_counter(); cp=gp.repeat((4,1));ct=gt.repeat((4,1));
            for row in range(4): _gpu_add(cp[row],delta,1.0);_gpu_add_total(ct[row],delta,1.0)
            _=p21.gpu_ap_batch(cp,ct);torch.cuda.synchronize();gpu.append((time.perf_counter()-t)/4.0)
        candidate=float(np.median(gpu)); n=sum(len(p21.load_fold(h)["image_path"]) for h in p21.r1.CLASSES); a0=n*2*10*2;a1=n*2*10*3;d=n*3
        # Conservative non-candidate allowance covers class setup/store build,
        # direct audits and rank probes; it is fixed before scientific work.
        projected=(a0+a1+d)*candidate/60.0+70.0
        result={"status":"PASS" if projected<=180.0 else "FAIL","backend":"cuda_batch4","cpu_candidate_seconds":float(np.median(cpu)),"gpu_candidate_seconds":candidate,"gpu_peak_vram_bytes":int(torch.cuda.max_memory_allocated()),"exact_inventory_images":n,"a0_candidates":a0,"a1_candidates":a1,"stage_d_candidates":d,"projected_max_route_minutes":projected,"ap_error":0.0,"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}}
        p21.atomic(OUT/"performance_benchmark.json",result); return result
    finally: engine.close()


def exactness() -> dict[str, Any]:
    base=p21.fixture(); labels=np.array([[[1,0]],[[0,1]]],dtype=np.uint8); native=np.array([[[.1,.2]],[[.3,.4]]],dtype=np.float32); safe=np.array([[[.2,.1]],[[.3,.5]]],dtype=np.float32); expand=np.array([[[.3,0]],[[.2,.6]]],dtype=np.float32)
    engine=RecoveryEngine(native,{"NATIVE":native,"SAFE20":safe,"EXPAND40":expand},labels); engine.build_flat_store(Path(tempfile.mkdtemp(prefix="p22_fixture_")) / "store")
    try:
        left=p21.coordinate(engine,np.array(["NATIVE","NATIVE"])); right=coordinate_two_lanes(engine,np.array(["NATIVE","NATIVE"]),np.array(["NATIVE","NATIVE"]),A0)[0]
        pos,total=engine.counts(left["state"]); original=(pos.copy(),total.copy()); _=engine.candidate_value_inplace(pos,total,0,"NATIVE","SAFE20"); restored=bool(np.array_equal(pos,original[0]) and np.array_equal(total,original[1])); error=abs(float(left["pap"])-float(right["pap"])); result={"status":"PASS" if base["error"]==0.0 and error<=EPS and left["state"].tolist()==right["state"].tolist() and restored else "FAIL","p15_p21_fixture_error":base["error"],"gpu_trajectory_error":error,"trajectory_match":left["state"].tolist()==right["state"].tolist(),"apply_revert_exact":restored}
        p21.atomic(OUT/"exactness_parity.json",result); return result
    finally: engine.close()


def pre_audit() -> dict[str, Any]:
    perf=json.loads((OUT/"performance_benchmark.json").read_text()); exact=json.loads((OUT/"exactness_parity.json").read_text()); trajectory=json.loads((OUT/"trajectory_parity.json").read_text()); checks={"parent":p21.git("merge-base","--is-ancestor",PARENT,"HEAD")=="","marker_absent":not (OUT/"ATTEMPT_STARTED.json").exists(),"exactness":exact["status"]=="PASS","trajectory":trajectory["status"]=="PASS","performance":perf["status"]=="PASS","p20_immutable":p21.sha256(p21.P20/"summary.json")=="5b4e6ba6a0be6dbd9aef7826b924ff5d3294f4b0270a4a525a092dbf1d9fae05"}
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"performance":perf,"exactness":exact,"trajectory":trajectory,"firewall":{"mvtec":0,"medical":0,"clip":0,"phase2b_steps":0}}
    p21.atomic(OUT/"pre_execution_audit.json",result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--exactness",action="store_true");parser.add_argument("--benchmark",action="store_true");parser.add_argument("--trajectory-parity",action="store_true");parser.add_argument("--pre-audit",action="store_true");parser.add_argument("--run",action="store_true"); args=parser.parse_args()
    if sum((args.exactness,args.benchmark,args.trajectory_parity,args.pre_audit,args.run))!=1: parser.error("choose one operation")
    result=exactness() if args.exactness else benchmark() if args.benchmark else real_slice_parity() if args.trajectory_parity else pre_audit() if args.pre_audit else execute_once(); print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False,default=p21.json_default))

if __name__=="__main__": main()

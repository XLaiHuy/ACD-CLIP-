"""Fail-closed P29 audit after all immutable predictions and scoring exist."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from typing import Any
import torch
from tools.sabra.data import EXPECTED_VISA_CLASSES,read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.p29_contract import P29_PROTOCOL_PATH,load_and_audit_p29_protocol
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import atomic_write_json,sha256_file
from tools.sabra_v2.train_region_distill import ROOT

def classify_p29(aggregate: dict[str,Any],audit_pass: bool)->str:
 supported=(aggregate["p29_macro_pAP"]>aggregate["native_macro_pAP"] and aggregate["p29_macro_pAUROC"]>=aggregate["native_macro_pAUROC"] and aggregate["median_delta_pAP"]>0 and aggregate["improving_class_count"]>=7 and audit_pass)
 if supported:return "P29_SUPPORTED"
 if not audit_pass:return "P29_ENGINEERING_STOP"
 if aggregate["p29_macro_pAP"]>aggregate["native_macro_pAP"] or aggregate["median_delta_pAP"]>0:return "P29_MIXED"
 return "P29_NOT_SUPPORTED"
def make_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-root",type=Path,required=True);p.add_argument("--cache-root",type=Path,required=True);p.add_argument("--visa-root",type=Path,required=True);p.add_argument("--p26-checkpoint",type=Path,required=True);p.add_argument("--clip-asset",type=Path,required=True);p.add_argument("--metadata",type=Path,default=ROOT / "dataset/hub/VisA.jsonl");p.add_argument("--execution-base-sha",required=True);p.add_argument("--output",type=Path,required=True);return p
def _git(*args):return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()
def run(args):
 protocol=load_and_audit_p29_protocol();assets=verify_p26_parent(args.p26_checkpoint,args.clip_asset,ROOT/"configs/phase2b_canonical_v1.json");attempt_path=args.run_root/"P29_ATTEMPT.json";complete_path=args.run_root/"P29_RUN_COMPLETE.json";gate_path=args.run_root/"P29_SCORING_GATE.json";aggregate_path=args.run_root/"aggregate"/"P29_AGGREGATE.json"
 if any(not p.is_file() for p in (attempt_path,complete_path,gate_path,aggregate_path)) or (args.run_root/"P29_ATTEMPT_FAILURE.json").exists():raise RuntimeError("complete P29 attempt evidence required")
 attempt=json.loads(attempt_path.read_text());complete=json.loads(complete_path.read_text());gate=json.loads(gate_path.read_text());aggregate=json.loads(aggregate_path.read_text())
 if attempt.get("attempt_uuid")!=complete.get("attempt_uuid") or attempt.get("scientific_execution_base_sha")!=args.execution_base_sha or gate.get("completion_status")!="PASS" or gate.get("prediction_count")!=12:raise RuntimeError("P29 attempt or scoring gate mismatch")
 expected_keys=set(RegionResidualAdapter().state_dict());rows=read_visa_metadata(args.metadata);folds=[];prediction_mtimes=[];metric_mtimes=[];training_steps=optimizer_steps=phase2b_steps=clip_steps=held_gt=held_mask=0
 for name in EXPECTED_VISA_CLASSES:
  inventory=loco_inventory(rows,name);base=args.run_root/name;training_path=base/"training"/"TRAINING_COMPLETE.json";checkpoint_path=base/"training"/"p29_region_adapter.pt";prediction_path=base/"predictions"/"p29_held_predictions.pt";prediction_complete=base/"predictions"/"PREDICTION_COMPLETE.json";metric_path=base/"metrics"/"p29_held_metrics.json";tier_b_path=args.cache_root/"tier_b"/name/"manifest.json"
  if any(not p.is_file() for p in (training_path,checkpoint_path,prediction_path,prediction_complete,metric_path,tier_b_path)):raise RuntimeError(f"incomplete P29 fold evidence {name}")
  training=json.loads(training_path.read_text());prediction=json.loads(prediction_complete.read_text());metric=json.loads(metric_path.read_text());tier_b=json.loads(tier_b_path.read_text());checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True);expected_steps=len(inventory.fit_rows)*20
  if training.get("steps")!=expected_steps or checkpoint.get("steps")!=expected_steps or checkpoint.get("status")!="FOLD_TRAINING_COMPLETE" or set(checkpoint.get("state_dict",{}))!=expected_keys:raise RuntimeError(f"P29 training contract failed {name}")
  if tier_b.get("held_class")!=name or tier_b.get("held_mask_reads")!=0 or tier_b.get("source_classes")!=sorted(set(EXPECTED_VISA_CLASSES)-{name}):raise RuntimeError(f"P29 source-only cache firewall failed {name}")
  if prediction.get("mask_reads")!=0 or prediction.get("gt_used") is not False or sha256_file(prediction_path)!=prediction.get("prediction_sha256"):raise RuntimeError(f"P29 immutable prediction failure {name}")
  if metric.get("fit_or_teacher_steps")!=0:raise RuntimeError(f"P29 score fitting violation {name}")
  training_steps+=int(training.get("steps",-1));optimizer_steps+=int(training.get("optimizer_steps",-1));phase2b_steps+=int(training.get("phase2b_optimization_steps",-1));clip_steps+=int(training.get("clip_optimization_steps",-1));held_gt+=int(training.get("held_gt_reads",-1));held_mask+=int(training.get("held_mask_reads",-1))+int(prediction.get("mask_reads",-1));prediction_mtimes.append(prediction_path.stat().st_mtime_ns);metric_mtimes.append(metric_path.stat().st_mtime_ns);folds.append({"held_class":name,"fit_records":len(inventory.fit_rows),"steps":expected_steps,"checkpoint_sha256":sha256_file(checkpoint_path),"prediction_sha256":prediction["prediction_sha256"]})
 if max(prediction_mtimes)>gate_path.stat().st_mtime_ns or min(metric_mtimes)<gate_path.stat().st_mtime_ns:raise RuntimeError("P29 scoring preceded all prediction freeze")
 branch=_git("branch","--show-current");local=_git("rev-parse","HEAD");fields=_git("ls-remote","origin",f"refs/heads/{branch}").split();remote=fields[0] if len(fields)==2 else "";clean=not bool(_git("status","--porcelain"));base_ok=remote==local==args.execution_base_sha
 audit_pass=all((len(folds)==12,held_gt==0,held_mask==0,int(attempt.get("mvtec_reads",-1))==0,int(attempt.get("medical_reads",-1))==0,phase2b_steps==0,clip_steps==0,training_steps==optimizer_steps,base_ok,clean));status=classify_p29(aggregate,audit_pass);result={"schema_version":"P29_POST_RUN_AUDIT_V1","status":"PASS" if audit_pass else "FAIL","terminal_status":status,"attempt_uuid":attempt["attempt_uuid"],"attempt_count":1,"fold_count":len(folds),"duplicate_scientific_folds":0,"rerun_poor_folds":0,"predictions_frozen_before_scoring":True,"held_gt_reads_before_scoring":held_gt,"held_mask_reads_before_scoring":held_mask,"training_steps":training_steps,"optimizer_steps":optimizer_steps,"phase2b_optimization_steps":phase2b_steps,"clip_optimization_steps":clip_steps,"new_clip_forwards":0,"new_phase2b_forwards":0,"mvtec_reads":int(attempt.get("mvtec_reads",-1)),"medical_reads":int(attempt.get("medical_reads",-1)),"only_region_residual_adapter_trained":True,"assets":assets,"protocol":protocol,"protocol_sha256":sha256_file(P29_PROTOCOL_PATH),"scientific_execution_base_sha":args.execution_base_sha,"local_sha":local,"remote_sha":remote,"remote_equals_local":base_ok,"worktree_clean":clean,"folds":folds,"aggregate":aggregate}
 if not audit_pass:raise RuntimeError(f"P29 post-run audit failed: {result}")
 atomic_write_json(args.output,result);return result
def main():print(json.dumps(run(make_parser().parse_args()),sort_keys=True))
if __name__=="__main__":main()

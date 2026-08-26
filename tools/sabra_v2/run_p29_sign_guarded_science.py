"""One-shot P29 cache-reuse/train/predict/score runner with a hard scoring barrier."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from tools.sabra.data import EXPECTED_VISA_CLASSES,read_visa_metadata
from tools.sabra_v2.cuda_runtime import build_p27_cuda_environment,probe_cuda_subprocess
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.p29_contract import P29_PROTOCOL_PATH,load_and_audit_p29_protocol,p29_cache_provenance
from tools.sabra_v2.region_cache import atomic_write_json,sha256_file
from tools.sabra_v2.train_region_distill import ROOT

EXACT_HELD_ORDER=tuple(EXPECTED_VISA_CLASSES)
def make_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--visa-root",type=Path,required=True);p.add_argument("--p26-checkpoint",type=Path,required=True);p.add_argument("--clip-asset",type=Path,required=True);p.add_argument("--cache-root",type=Path,required=True);p.add_argument("--run-root",type=Path,required=True);p.add_argument("--metadata",type=Path,default=ROOT / "dataset/hub/VisA.jsonl");p.add_argument("--execution-base-sha",required=True);p.add_argument("--p29-prereg-sha",required=True);p.add_argument("--seed",type=int,default=0);return p
def _utc():return datetime.now(timezone.utc).isoformat()
def _git(*args):return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,capture_output=True).stdout.strip()
def _remote(branch):
 fields=_git("ls-remote","origin",f"refs/heads/{branch}").split()
 if len(fields)!=2:raise RuntimeError("could not resolve P29 remote branch")
 return fields[0]
def _verify_frozen_git(execution_base):
 branch=_git("branch","--show-current");local=_git("rev-parse","HEAD")
 if local!=execution_base:raise RuntimeError("P29 execution-base mismatch")
 if _git("status","--porcelain"):raise RuntimeError("P29 scientific execution requires clean worktree")
 remote=_remote(branch)
 if remote!=local:raise RuntimeError("P29 remote/local mismatch")
 return branch,remote
def _run(module,args,env):
 command=[sys.executable,"-m",module,*args];print(json.dumps({"event":"START","utc":_utc(),"module":module,"command":command}),flush=True);subprocess.run(command,cwd=ROOT,env=env,check=True);print(json.dumps({"event":"COMPLETE","utc":_utc(),"module":module}),flush=True)
def _prediction_gate(root):
 artifacts=[]
 for name in EXACT_HELD_ORDER:
  completion_path=root/name/"predictions"/"PREDICTION_COMPLETE.json";prediction_path=root/name/"predictions"/"p29_held_predictions.pt"
  if not completion_path.is_file() or not prediction_path.is_file():raise RuntimeError(f"all 12 P29 predictions must freeze before scoring; missing {name}")
  completion=json.loads(completion_path.read_text()); observed=sha256_file(prediction_path)
  if completion.get("completion_status")!="COMPLETE" or completion.get("held_class")!=name or completion.get("prediction_sha256")!=observed or prediction_path.stat().st_mode&0o222:raise RuntimeError(f"invalid immutable P29 prediction {name}")
  artifacts.append({"held_class":name,"path":str(prediction_path),"sha256":observed})
 return artifacts
def run(args):
 protocol=load_and_audit_p29_protocol();verify_p26_parent(args.p26_checkpoint,args.clip_asset,ROOT/"configs/phase2b_canonical_v1.json");branch,remote=_verify_frozen_git(args.execution_base_sha);rows=read_visa_metadata(args.metadata)
 if tuple(sorted({str(r["class_name"]) for r in rows}))!=tuple(sorted(EXACT_HELD_ORDER)):raise RuntimeError("exact VisA inventory failed")
 if args.seed!=0:raise RuntimeError("P29 seed must remain 0")
 environment=build_p27_cuda_environment(os.environ);runtime_probe=probe_cuda_subprocess(environment,sys.executable);args.run_root.mkdir(parents=True,exist_ok=True);attempt_path=args.run_root/"P29_ATTEMPT.json"
 if attempt_path.exists() or any((args.run_root/name).exists() for name in EXACT_HELD_ORDER):raise RuntimeError("P29 attempt already consumed or fold artifacts pre-exist")
 gpu=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,uuid","--format=csv,noheader"],check=True,text=True,capture_output=True).stdout.strip();attempt_uuid=str(uuid.uuid4());provenance=p29_cache_provenance(args.metadata)
 attempt={"schema_version":"P29_SCIENTIFIC_ATTEMPT_V1","completion_status":"ATTEMPT_CONSUMED","attempt_uuid":attempt_uuid,"utc_timestamp":_utc(),"scientific_execution_base_sha":args.execution_base_sha,"p29_prereg_sha":args.p29_prereg_sha,"p29_protocol_sha256":sha256_file(P29_PROTOCOL_PATH),"branch":branch,"remote_sha":remote,"p27_terminal_sha":"cdf06234bee861bbe81a7f07e382530f9a66c207","p28r1_terminal_sha":"feb7d5dcf5a0e0b4933e1069288cb4ab30dc8101","p26_sha256":sha256_file(args.p26_checkpoint),"clip_sha256":sha256_file(args.clip_asset),"config_sha256":sha256_file(ROOT/"configs/phase2b_canonical_v1.json"),"cache_provenance":provenance.as_dict(),"gpu":gpu,"metadata_sha256":sha256_file(args.metadata),"held_order":list(EXACT_HELD_ORDER),"seed":0,"training_steps":0,"optimizer_steps":0,"new_clip_forwards":0,"new_phase2b_forwards":0,"mvtec_reads":0,"medical_reads":0,"protocol_audit":protocol,"qualified_cuda_child_probe":runtime_probe}
 atomic_write_json(attempt_path,attempt);common=["--visa-root",str(args.visa_root),"--p26-checkpoint",str(args.p26_checkpoint),"--clip-asset",str(args.clip_asset),"--cache-root",str(args.cache_root),"--metadata",str(args.metadata),"--execution-base-sha",args.execution_base_sha];started=time.perf_counter()
 try:
  for name in EXACT_HELD_ORDER:
   fold=args.run_root/name
   _run("tools.sabra_v2.train_region_distill_p29_cached",["--held-class",name,*common,"--output",str(fold/"training"),"--epochs","20","--batch-size","1","--learning-rate","0.001","--seed","0","--device","cuda","--num-workers","0"],environment)
   _run("tools.sabra_v2.evaluate_region_distill_p29_cached",["--held-class",name,*common,"--adapter-checkpoint",str(fold/"training"/"p29_region_adapter.pt"),"--output",str(fold/"predictions"),"--batch-size","1","--device","cuda","--num-workers","0"],environment)
  predictions=_prediction_gate(args.run_root);atomic_write_json(args.run_root/"P29_SCORING_GATE.json",{"schema_version":"P29_SCORING_GATE_V1","completion_status":"PASS","utc_timestamp":_utc(),"prediction_count":12,"predictions":predictions,"fit_or_teacher_steps_after_gate":0})
  for name in EXACT_HELD_ORDER:
   fold=args.run_root/name;_run("tools.sabra_v2.score_region_distill_p29_frozen",["--held-class",name,"--visa-root",str(args.visa_root),"--predictions",str(fold/"predictions"/"p29_held_predictions.pt"),"--output",str(fold/"metrics"),"--metadata",str(args.metadata)],environment)
  _run("tools.sabra_v2.aggregate_region_distill_p29",["--run-root",str(args.run_root),"--output",str(args.run_root/"aggregate")],environment)
  result={"schema_version":"P29_SCIENTIFIC_RUN_COMPLETE_V1","completion_status":"COMPLETE","attempt_uuid":attempt_uuid,"utc_timestamp":_utc(),"scientific_execution_base_sha":args.execution_base_sha,"actual_scientific_runtime_seconds":time.perf_counter()-started,"fold_count":12,"prediction_count_before_scoring":12,"attempt_count":1,"mvtec_reads":0,"medical_reads":0};atomic_write_json(args.run_root/"P29_RUN_COMPLETE.json",result);_run("tools.sabra_v2.p29_post_run_audit",["--run-root",str(args.run_root),"--cache-root",str(args.cache_root),"--visa-root",str(args.visa_root),"--p26-checkpoint",str(args.p26_checkpoint),"--clip-asset",str(args.clip_asset),"--metadata",str(args.metadata),"--execution-base-sha",args.execution_base_sha,"--output",str(args.run_root/"P29_POST_RUN_AUDIT.json")],environment);return result
 except BaseException as exc:
  atomic_write_json(args.run_root/"P29_ATTEMPT_FAILURE.json",{"schema_version":"P29_ATTEMPT_FAILURE_V1","attempt_uuid":attempt_uuid,"utc_timestamp":_utc(),"error_type":type(exc).__name__,"error":str(exc),"automatic_rerun_forbidden":True});raise
def main():print(json.dumps(run(make_parser().parse_args()),sort_keys=True))
if __name__=="__main__":main()

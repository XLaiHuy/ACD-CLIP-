"""Freeze GT-free P29 held predictions from exact Tier-A cache with batch size one."""
from __future__ import annotations
import argparse, json, os, time, uuid
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.p29_contract import load_and_audit_p29_protocol, p29_cache_provenance
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import TierADataset, atomic_write_json, sha256_file
from tools.sabra_v2.student_forward import forward_region_student
from tools.sabra_v2.train_region_distill import ROOT

def make_parser() -> argparse.ArgumentParser:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True); p.add_argument("--visa-root", type=Path, required=True, help="provenance-only; no file below this root is opened"); p.add_argument("--p26-checkpoint", type=Path, required=True); p.add_argument("--clip-asset", type=Path, required=True); p.add_argument("--cache-root", type=Path, required=True); p.add_argument("--adapter-checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl"); p.add_argument("--execution-base-sha", required=True); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--num-workers", type=int, choices=(0,), default=0); return p

def run(args: argparse.Namespace) -> dict[str, object]:
 if args.batch_size != 1: raise RuntimeError("P29 held prediction batch size must remain exactly one")
 load_and_audit_p29_protocol(); verify_p26_parent(args.p26_checkpoint,args.clip_asset,ROOT / "configs/phase2b_canonical_v1.json")
 provenance=p29_cache_provenance(args.metadata); checkpoint=torch.load(args.adapter_checkpoint,map_location="cpu",weights_only=True)
 if checkpoint.get("status")!="FOLD_TRAINING_COMPLETE" or checkpoint.get("held_class")!=args.held_class or checkpoint.get("cache_provenance")!=provenance.as_dict() or checkpoint.get("p29_execution_base_sha")!=args.execution_base_sha: raise RuntimeError("P29 adapter checkpoint provenance mismatch")
 inventory=loco_inventory(read_visa_metadata(args.metadata),args.held_class); dataset=TierADataset(inventory.held_rows,args.cache_root,provenance); loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
 device=torch.device(args.device); adapter=RegionResidualAdapter().to(device); adapter.load_state_dict(checkpoint["state_dict"],strict=True); adapter.eval(); records=[]; started=time.perf_counter()
 with torch.no_grad():
  for batch in loader:
   seg=batch["seg_features"].permute(1,0,2,3).to(device=device,dtype=torch.float32); native=batch["native_logits"].permute(1,0,2,3).to(device=device,dtype=torch.float32); student=forward_region_student(adapter,seg,native)
   records.append({"image_path": batch["image_path"][0], "class_name":args.held_class, "native_abnormal_probability":student.native_probability[0,1].detach().cpu(), "p29_abnormal_probability":student.deployed_probability[0,1].detach().cpu()})
 args.output.mkdir(parents=True,exist_ok=True); output_path=args.output / "p29_held_predictions.pt"; payload={"schema_version":"P29_IMMUTABLE_HELD_PREDICTIONS_V1","held_class":args.held_class,"gt_used":False,"mask_reads":0,"cache_provenance":provenance.as_dict(),"p29_execution_base_sha":args.execution_base_sha,"adapter_checkpoint_sha256":sha256_file(args.adapter_checkpoint),"records":records}; temporary=output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp"); torch.save(payload,temporary); os.replace(temporary,output_path); output_path.chmod(0o444)
 result={"prediction_path":str(output_path),"prediction_sha256":sha256_file(output_path),"held_class":args.held_class,"records":len(records),"gt_used":False,"mask_reads":0,"prediction_seconds":time.perf_counter()-started,"completion_status":"COMPLETE"}; atomic_write_json(args.output / "PREDICTION_COMPLETE.json",result); return result

def main() -> None: print(json.dumps(run(make_parser().parse_args()),sort_keys=True))
if __name__ == "__main__": main()

"""Score immutable P29 predictions only after the all-fold freeze gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from tools.sabra.data import EXPECTED_VISA_CLASSES, VisaEvaluationDataset, read_visa_metadata
from tools.sabra_car.r0_direction import exact_metrics
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_cache import atomic_write_json,sha256_file
from tools.sabra_v2.train_region_distill import ROOT

def make_parser():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--held-class",choices=EXPECTED_VISA_CLASSES,required=True);p.add_argument("--visa-root",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--metadata",type=Path,default=ROOT / "dataset/hub/VisA.jsonl");return p

def run(args):
 payload=torch.load(args.predictions,map_location="cpu",weights_only=True)
 if payload.get("schema_version")!="P29_IMMUTABLE_HELD_PREDICTIONS_V1" or payload.get("held_class")!=args.held_class or payload.get("gt_used") is not False or payload.get("mask_reads")!=0: raise RuntimeError("P29 scoring requires matching GT-free immutable predictions")
 inventory=loco_inventory(read_visa_metadata(args.metadata),args.held_class); records=payload.get("records");
 if not isinstance(records,list) or len(records)!=len(inventory.held_rows): raise RuntimeError("P29 frozen inventory mismatch")
 by_path={str(r["image_path"]):r for r in records}
 if len(by_path)!=len(records): raise RuntimeError("P29 duplicate frozen paths")
 native=[];p29=[];labels=[];mask_reads=0
 for batch in DataLoader(VisaEvaluationDataset(inventory.held_rows,args.visa_root),batch_size=1,shuffle=False,num_workers=0):
  record=by_path.get(str(batch["image_path"][0]));
  if record is None: raise RuntimeError("missing P29 frozen prediction")
  for key,collection in (("native_abnormal_probability",native),("p29_abnormal_probability",p29)):
   value=record.get(key)
   if not isinstance(value,torch.Tensor) or tuple(value.shape)!=(518,518): raise RuntimeError("P29 frozen map shape mismatch")
   collection.append(value.numpy().astype(np.float32,copy=False).reshape(-1))
  labels.append(batch["mask"][0,0].numpy().astype(np.uint8,copy=False).reshape(-1));mask_reads+=int(batch["label"][0].item())
 n=exact_metrics(np.concatenate(native),np.concatenate(labels)); s=exact_metrics(np.concatenate(p29),np.concatenate(labels)); result={"schema_version":"P29_HELD_METRICS_V1","held_class":args.held_class,"prediction_sha256":sha256_file(args.predictions),"fit_or_teacher_steps":0,"native_metrics":n,"p29_metrics":s,"delta":{k:s[k]-n[k] for k in ("pAP","pAUROC")},"held_mask_file_reads_after_prediction_freeze":mask_reads};args.output.mkdir(parents=True,exist_ok=True);atomic_write_json(args.output / "p29_held_metrics.json",result);return {"metrics_path":str(args.output / "p29_held_metrics.json"),**result}
def main(): print(json.dumps(run(make_parser().parse_args()),sort_keys=True))
if __name__=="__main__":main()

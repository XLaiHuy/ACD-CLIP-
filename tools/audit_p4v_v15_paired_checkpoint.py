#!/usr/bin/env python3
"""Inference-only V1.5 causal replay from retained paired adapter states."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from run_p4v_short64 import build
from run_p4v_v15_paired import delta, evaluate
from utils import configure_canonical_fp32


def load(config, path, device):
    model=build(config,device); state=torch.load(path,map_location="cpu",weights_only=False)
    model.image_adapter.load_state_dict(state["image_adapter"]);model.text_adapter.load_state_dict(state["text_adapter"]);model.h6.load_state_dict(state["h6"])
    return model


def main():
    root=ROOT/"runs/phase4v/v1_5/paired";config=json.loads((ROOT/"runs/phase4/k1/short64_seed0_attempt5/config.json").read_text());manifest=json.loads((ROOT/"runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json").read_text());configure_canonical_fp32();device=torch.device("cuda:0");data=DeterministicVisATrainDataset(manifest,config["img_size"])
    base=load(config,root/"base_adapter_state.pth",device);reports={"BASE":evaluate(base,data,config,"OFF")};del base;torch.cuda.empty_cache()
    candidate=load(config,root/"current_v1_adapter_state.pth",device);reports.update({mode:evaluate(candidate,data,config,mode) for mode in ("ACTIVE","OFF","ZERO_DELTA","ZERO_GATE")})
    identity={key:{"total_delta":delta(reports["ACTIVE"],reports["BASE"])[key],"intervention_effect":delta(reports["ACTIVE"],reports["OFF"])[key],"optimization_drift":delta(reports["OFF"],reports["BASE"])[key]} for key in ("normal_bce","anomaly_bce","pixel_ap_macro","pixel_auc_macro")}
    for row in identity.values():row["reconstruction_error"]=row["total_delta"]-row["intervention_effect"]-row["optimization_drift"]
    primary=("pixel_ap_macro","pixel_auc_macro"); shares={key:abs(identity[key]["optimization_drift"])/max(abs(identity[key]["total_delta"]),1e-12) for key in primary}; classification="V1_HARM_DOMINATED_BY_OPTIMIZATION_DRIFT" if all(shares[key]>=.70 and identity[key]["optimization_drift"]*identity[key]["total_delta"]>=0 for key in primary) else "V1_MIXED_OR_INCONCLUSIVE"
    decision={"decision":classification,"dominance_rule":"at least 70% of absolute primary degradation with consistent sign","optimization_drift_share":shares,"causal_identity":identity,"geometry_summary":reports["ACTIVE"]["geometry"],"next_authorized":"Branch A: gradient-isolated conditional correction" if classification=="V1_HARM_DOMINATED_BY_OPTIMIZATION_DRIFT" else "Branch E stop/inconclusive"}
    report={"decision":classification,"inference_only":True,"provenance":{"config_sha256":_sha256(ROOT/"runs/phase4/k1/short64_seed0_attempt5/config.json"),"manifest_sha256":_sha256(ROOT/"runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"),"base_checkpoint":str(root/"base_adapter_state.pth"),"candidate_checkpoint":str(root/"current_v1_adapter_state.pth")},"reports":reports,"causal_identity":identity,"classification":decision}
    (root/"V1_5_PAIRED_CAUSAL.json").write_text(json.dumps(report,indent=2)+"\n");(root.parent/"V1_5_CAUSAL_DECISION.json").write_text(json.dumps(decision,indent=2)+"\n");print(json.dumps(decision,indent=2))


if __name__=="__main__":main()

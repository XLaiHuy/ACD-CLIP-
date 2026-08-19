from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn

from sabra import trust_v2
from sabra.trust_v2 import visa_audit as audit

ROOT = Path(__file__).resolve().parents[3]
RECOVERY_ROOT = ROOT / "runs/phase5/sabra/TRUST_V2_M4_RECOVERY_V2"
RESULT_ROOT = RECOVERY_ROOT
RESULT_NAMES = {"TRUST_V2_MODEL_AUDIT.json", "PCRR_DISAGREEMENT_AUDIT.json", "DECISION.json", "NEED_C1_FROZEN_MODEL.json", "READINESS_AUDIT.json"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"ARTIFACT_PATH_COLLISION {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.tmp.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def fit_full(features: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int8)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values.copy(), copy=False)
    model = LogisticRegression(class_weight="balanced", solver="lbfgs", C=1.0, max_iter=1000, random_state=0)
    model.fit(scaled, labels)
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "logistic_coef": model.coef_.tolist(),
        "logistic_intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
        "n_features_in": int(model.n_features_in_),
        "random_seed": 0,
        "sklearn_version": sklearn.__version__,
        "configuration": {"class_weight": "balanced", "solver": "lbfgs", "C": 1.0, "max_iter": 1000, "random_state": 0},
    }


def run() -> None:
    for name in ("TRUST_V2_FROZEN_MODEL.json", "EXTERNAL_VALIDATION_FREEZE.json", "FREEZE_VERIFICATION.json"):
        if (RECOVERY_ROOT / name).exists():
            raise RuntimeError(f"ARTIFACT_PATH_COLLISION {RECOVERY_ROOT / name}")
    decision = json.loads((RESULT_ROOT / "DECISION.json").read_text())
    model_audit = json.loads((RESULT_ROOT / "TRUST_V2_MODEL_AUDIT.json").read_text())
    pcrr_audit = json.loads((RESULT_ROOT / "PCRR_DISAGREEMENT_AUDIT.json").read_text())
    readiness = json.loads((RESULT_ROOT / "READINESS_AUDIT.json").read_text())
    selected_name = str(decision["selected_model"])
    if selected_name not in {"M1_E_Credibility", "M2_E_Credibility_S9_R9", "M3_E_Credibility_S9_R9_S16_R16"}:
        raise RuntimeError("TRUST_V2_CANDIDATE_NOT_ELIGIBLE")
    if decision["terminal"] != "TRUST_V2_DEVELOPMENT_ELIGIBLE":
        raise RuntimeError("TRUST_V2_CANDIDATE_NOT_ELIGIBLE")
    records, manifest = audit.load_gt_free_records()
    gt, occupancy, metadata = audit.load_patch_targets(records)
    del occupancy
    target = gt.reshape(-1).astype(np.int8)
    arrays = {"E": audit._flat(records, "baseline_pgm").astype(np.float32), "peer_coherence": audit._flat(records, "peer_coherence").astype(np.float32), "query_support_mean": audit._flat(records, "query_support_mean").astype(np.float32), "peer_eigen_entropy": audit._flat(records, "peer_eigen_entropy").astype(np.float32), "stage_query_profile_disagreement": audit._flat(records, "stage_query_profile_disagreement").astype(np.float32), "S9": audit._flat(records, "S9").astype(np.float32), "R9": audit._flat(records, "R9").astype(np.float32), "S16": audit._flat(records, "S16").astype(np.float32), "R16": audit._flat(records, "R16").astype(np.float32)}
    order = model_audit["model_feature_order"][selected_name]
    trust_features = np.column_stack([arrays[name] for name in order])
    trust_parameters = fit_full(trust_features, target)
    old_fields = ("native_logits", "margin_within_image_rank", "robust_margin_normalization", "D_rank", "deployment_sensitivity")
    c1_records: list[dict[str, Any]] = []
    for class_name in audit.EXPECTED_VISA_CLASSES:
        class_records = [record for record in records if record["class_name"] == class_name]
        with np.load(audit.OLD_CACHE_ROOT / f"{class_name}.npz", allow_pickle=False) as old:
            old_index = {path: index for index, path in enumerate(old["image_path"].astype(str))}
            for record in class_records:
                index = old_index[record["image_path"]]
                c1_records.append({key: np.array(old[key][index], copy=True) for key in old_fields} | {"class_name": class_name, "image_path": record["image_path"]})
    import torch
    data_root = Path(os.environ.get("ACDCLIP_DATA_ROOT", "/workspace/data"))
    if (data_root / "VisA_20220922").is_dir():
        data_root = data_root / "VisA_20220922"
    signed, _, _, oracle_parity = audit.need_oracle(c1_records, metadata, data_root, torch.device("cuda"))
    c1_target = (signed.reshape(-1) > 1e-8).astype(np.int8)
    c1_order = ["margin_within_image_rank", "robust_margin_normalization", "D_rank", "deployment_sensitivity"]
    c1_features = np.column_stack([audit._flat(c1_records, name) for name in c1_order])
    need_parameters = fit_full(c1_features, c1_target)
    need_parameters["feature_order"] = c1_order
    need_parameters["utility_target"] = "signed utility > 1e-8"
    need_parameters["oracle_parity"] = oracle_parity
    common = {"status": "CANDIDATE_FROZEN", "study": "TRUST_V2_M4_RECOVERY_V2", "created_from_commit": git("rev-parse", "HEAD"), "selected_model": selected_name, "selected_model_feature_order": order, "D_rel_retained": False, "PCRR_STATUS": pcrr_audit["PCRR_STATUS"], "M4_diagnostic_feature_order": pcrr_audit["M4_feature_order"], "M4_definition": pcrr_audit["M4_definition"], "cache_manifest_sha256": readiness["manifest_sha256"], "cache_shards_sha256": readiness["cache_shard_sha256"], "implementation_hashes": readiness["source_and_asset_sha256"], "result_artifact_sha256": {name: sha256(RESULT_ROOT / name) for name in ["DECISION.json", "TRUST_V2_MODEL_AUDIT.json", "PCRR_DISAGREEMENT_AUDIT.json", "NEED_C1_FROZEN_MODEL.json"]}}
    trust_freeze = common | {"trust_model_parameters": trust_parameters, "formulas": {"Trust": "selected VisA-frozen logistic model on the exact feature order", "M4": "selected_non_PCRR_model + D_rel, diagnostic-only", "D_rel": "abs(PGM_baseline_rank - PCRR_baseline_rank)", "p9": "exact ninth candidate in frozen B1 ordering", "p16": "exact sixteenth candidate in frozen B1 ordering"}, "metrics": {"selected_model": model_audit["metrics"][selected_name], "selected_effect": decision["selected_effect"], "statuses": decision["statuses"]}}
    external_freeze = common | {"trust_model": trust_freeze, "need_c1_model_parameters": need_parameters, "authority_formula": "A2_N_T_v2 = C1 * T_v2; comparator A1_N_E_cal = C1 * M0_E OOF", "evaluation_protocol": {"MVTec": "same GT-free relational construction; frozen VisA scaler/model/Need; GT evaluation only", "no_tuning": True, "medical": "forbidden"}, "external_decision_rules": {"trust_supported": "mean >= 0.010, median >= 0.005, at least 2/3 classes positive, no catastrophic tail", "trust_promising": "mean >= 0.005, median >= 0, at least 60 percent classes positive, no catastrophic tail", "trust_weak": "mean > 0, median >= 0, at least 50 percent classes non-negative, no catastrophic tail", "authority_not_falsified": True}, "source_hashes": readiness["source_and_asset_sha256"]}
    verification = {"status": "CANDIDATE_FROZEN", "freeze_root": str(RECOVERY_ROOT), "files": ["TRUST_V2_FROZEN_MODEL.json", "EXTERNAL_VALIDATION_FREEZE.json", "NEED_C1_FROZEN_MODEL.json"], "need_c1_parameters_embedded_in": "EXTERNAL_VALIDATION_FREEZE.json", "MVTec_reads": 0, "medical_reads": 0, "created_from_commit": git("rev-parse", "HEAD")}
    atomic_json(RECOVERY_ROOT / "TRUST_V2_FROZEN_MODEL.json", trust_freeze)
    atomic_json(RECOVERY_ROOT / "EXTERNAL_VALIDATION_FREEZE.json", external_freeze)
    atomic_json(RECOVERY_ROOT / "FREEZE_VERIFICATION.json", verification)
    print(json.dumps({"selected_model":selected_name,"D_rel_retained":False,"PCRR_STATUS":pcrr_audit["PCRR_STATUS"],"MVTec_reads":0,"medical_reads":0}, sort_keys=True))


if __name__ == "__main__":
    run()

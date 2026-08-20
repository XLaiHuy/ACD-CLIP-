# LAB_20E_READY_V2 run contract

All commands below run from the repository root after setup and explicit lab authorization. Use a new collision-free `RUN_ID`; no command performs checkpoint selection from MVTec or Medical.

## State machine

```text
LAB PRECHECK
  -> 20e TRAIN (VisA train only)
  -> adapter_20.pth created
  -> checkpoint identity/integrity verification
  -> FREEZE adapter_20.pth
  -> VisA post-training validation (metrics only)
  -> MVTec post-training frozen benchmark (FAST; no tuning)
  -> sealed Medical final evaluation (explicit authorization)
  -> STOP
```

## Preflight and training

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)'
python tools/sabra/lab_preflight.py --dataset visa --root "$VISA_ROOT"
python tools/sabra/lab_train.py preflight --config runs/phase5/sabra/LAB_20E_READY_V2/TRAIN20E_FINAL_CONFIG.json
python tools/sabra/lab_train.py train --config runs/phase5/sabra/LAB_20E_READY_V2/TRAIN20E_FINAL_CONFIG.json --run-id "$RUN_ID"
python tools/sabra/checkpoint_identity.py --checkpoint "$ACDCLIP_RUN_ROOT/$RUN_ID/checkpoints/adapter_20.pth" --expected-epoch 20
```

The preflight is disposable infrastructure validation, not epoch 1. Training uses VisA only; MVTec and Medical are never discovered by the training loader.

## Resume

```bash
python tools/sabra/lab_train.py resume --config runs/phase5/sabra/LAB_20E_READY_V2/TRAIN20E_FINAL_CONFIG.json --checkpoint "$ACDCLIP_RUN_ROOT/$RUN_ID/checkpoints/adapter_<EPOCH>.pth" --run-id "${RUN_ID}_resume"
```

## Post-training evaluation

Freeze `adapter_20.pth` before any command below. Each command must pass that exact checkpoint and writes a unique validation root.

```bash
python tools/sabra/lab_eval.py visa --checkpoint "$ACDCLIP_RUN_ROOT/$RUN_ID/checkpoints/adapter_20.pth" --data-root "$VISA_ROOT" --output-root "$ACDCLIP_RUN_ROOT/$RUN_ID/validation/visa"
python tools/sabra/lab_eval.py mvtec --checkpoint "$ACDCLIP_RUN_ROOT/$RUN_ID/checkpoints/adapter_20.pth" --data-root "$MVTEC_ROOT" --output-root "$ACDCLIP_RUN_ROOT/$RUN_ID/validation/mvtec"
```

The VisA command uses the existing `test.py` metric implementation. The historical `tools/sabra/trust_v2/visa_audit.py` remains historical and is not an adapter_20 evaluator. MVTec is a previously observed external benchmark; its GT-free manifest remains before GT evaluation and its historical adapter_5 result is not overwritten.

## Sealed Medical final evaluation

Only after adapter_20 and the evaluation source are frozen, and only with explicit authorization:

```bash
python tools/prepare_phase4_medical_splits.py --output-root "$ACDCLIP_RUN_ROOT/$RUN_ID/validation/medical_manifests" --data-root "$MEDICAL_ROOT" --val-ratio 0.30 --seed 0
python tools/sabra/lab_eval.py medical --checkpoint "$ACDCLIP_RUN_ROOT/$RUN_ID/checkpoints/adapter_20.pth" --data-root "$MEDICAL_ROOT" --manifest-root "$ACDCLIP_RUN_ROOT/$RUN_ID/validation/medical_manifests" --output-root "$ACDCLIP_RUN_ROOT/$RUN_ID/validation/medical" --allow-medical-evaluation
```

Medical results are evaluation-only and cannot feed training, checkpoint selection, thresholding, or architecture changes.

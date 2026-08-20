Lab run commands

After setup and explicit lab authorization, use a new RUN_ID. The wrapper
refuses to reuse an output root.

GPU/data preflight:
    python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)'
    python tools/sabra/lab_preflight.py --dataset visa --root "$VISA_ROOT"

Bounded one-step infrastructure preflight:
    python tools/sabra/lab_train.py preflight --config runs/phase5/sabra/LAB_20E_READY/TRAIN20E_FINAL_CONFIG.json

This runs one VisA batch through forward, loss, backward, optimizer step, and
checkpoint write/read in disposable output. It is not epoch 1.

Full 20e:
    python tools/sabra/lab_train.py train --config runs/phase5/sabra/LAB_20E_READY/TRAIN20E_FINAL_CONFIG.json --run-id sabra_p1_v83_seed0_$(date -u +%Y%m%dT%H%M%SZ)

Resume:
    python tools/sabra/lab_train.py resume --config runs/phase5/sabra/LAB_20E_READY/TRAIN20E_FINAL_CONFIG.json --checkpoint "$ACDCLIP_RUN_ROOT/<RUN_ID>/checkpoints/adapter_<EPOCH>.pth" --run-id <NEW_RESUME_RUN_ID>

VisA validation:
    python tools/sabra/trust_v2/visa_audit.py
This existing frozen held-out-class audit is not an automatic checkpoint
selector; freeze adapter_20.pth before invoking.

MVTec benchmark after checkpoint freeze:
    python -m sabra.trust_v2.mvtec_external --data-root "$MVTEC_ROOT" --output-root "$ACDCLIP_RUN_ROOT/<RUN_ID>/validation/mvtec" --backend fast

Future Medical evaluation after final model/evaluation freeze:
    python <separate-medical-evaluator> --checkpoint "$ACDCLIP_RUN_ROOT/<RUN_ID>/checkpoints/adapter_20.pth" --medical-root "$MEDICAL_ROOT" --allow-medical-evaluation

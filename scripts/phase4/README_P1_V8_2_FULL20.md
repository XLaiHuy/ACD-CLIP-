# P1-v8.2 20-Epoch Training & Testing User Guide

**Repository**: `/home/ai4/caohuy/ACD-CLIP-phase4`  
**Branch**: `phase4-progress1-cops-dynamic-prompt`  
**Commit HEAD**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`  
**Pre-Launch Audit Status**: `PRELAUNCH_READY`

---

## 1. Overview & Protocol Principles

This guide documents the execution of the P1-v8.2 20-epoch training and testing protocol.

### Core Scientific Principles:
1. **No Validation Set**: The training runs for exactly 20 epochs without early stopping or validation-based hyperparameter tuning.
2. **Canonical Final Model**: **Epoch 20** is the canonical final checkpoint.
3. **Trajectory Reporting (Epochs 10–20)**: Epochs 10 through 20 are evaluated on 100% of the official test split strictly to report performance trajectory over time. Test metrics **MUST NOT** be used to select a "best epoch" or alter model parameters.

---

## 2. Resource Requirements & Measured Headroom

From controlled pre-launch memory probes on an NVIDIA GeForce RTX 5060 Ti (16 GiB VRAM):

- **Peak Reserved GPU Memory (Train)**: **5,198 MB (~5.2 GB / 32.8% of VRAM)**
- **GPU VRAM Headroom**: **`67.18%`** (Comfortably exceeds the $\ge 15\%$ safety threshold)
- **Peak Reserved GPU Memory (Test)**: **2,872 MB (~2.9 GB / 18.1% of VRAM)**
- **GPU Test Headroom**: **`81.87%`**
- **Host RSS Memory**: **~3.7 GB** (Stable across accumulation windows, no host RAM inflation)

---

## 3. Settings Comparison Matrix (P1-v8.2 vs Stable Prior Phases)

| Setting | Phase-2B | P1-v7 Full | P1-v8.2 Final | Parity Status |
|---|---|---|---|---|
| Dataset & Split | VisA | VisA | VisA | `SAME_AS_STABLE` |
| Image Size | 518 | 518 | 518 | `SAME_AS_STABLE` |
| Backbone | ViT-L-14-336 | ViT-L-14-336 | ViT-L-14-336 | `SAME_AS_STABLE` |
| Pretrained Source | openai | openai | openai | `SAME_AS_STABLE` |
| $N$ Groups | 4 | 3 | 3 | `INTENTIONAL_CHANGE` |
| DFG Mode | mlp | attn | attn | `SAME_AS_STABLE` |
| SS2D Fusion | N/A | weight_residual | weight_residual | `SAME_AS_STABLE` |
| Microbatch Size | 1 | 1 | 1 | `SAME_AS_STABLE` |
| Grad Accum Steps | 1 | 6 | 6 | `SAME_AS_STABLE` |
| Effective Batch Size | 1 | 6 | 6 | `SAME_AS_STABLE` |
| Precision | bf16 | bf16 | bf16 | `SAME_AS_STABLE` |
| Grad Checkpointing | true | true | true | `SAME_AS_STABLE` |
| DataLoader Workers | 4 | 2 | 2 | `INTENTIONAL_CHANGE` (RAM safety) |
| Optimizer / LR | Adam / 1e-3 | Adam / 1e-3 | Adam / 1e-3 | `SAME_AS_STABLE` |
| Scheduler | StepLR(0.9) | StepLR(0.9) | StepLR(0.9) | `SAME_AS_STABLE` |
| Grad Clip Norm | 1.0 | 1.0 | 1.0 | `SAME_AS_STABLE` |

---

## 4. How to Launch Execution

To launch preflight verification, 20-epoch training, epochs 10–20 test evaluation, and metrics aggregation:

```bash
cd /home/ai4/caohuy/ACD-CLIP-phase4
bash scripts/phase4/run_p1_v8_2_full20_train_then_test.sh --execute
```

---

## 5. Monitoring & Failure Diagnosis

### 5.1 Monitoring Progress
During execution, monitor resource logs and training progress:

- **Training Log**: `tail -f runs/phase4/p1_v8_2_full20_seed0/train.log`
- **Resource Monitor Log**: `tail -f runs/phase4/p1_v8_2_full20_seed0/resource_monitor.log`

### 5.2 Failure Exit Codes & Diagnostic Files
If execution halts, inspect `runs/phase4/p1_v8_2_full20_seed0/exit_code.txt` and `failure_reason.txt`:

| Exit Code | Reason Label | Action / Meaning |
|---|---|---|
| `0` | `SUCCESS` | Pipeline completed successfully. |
| `137` | `HOST_OOM_OR_SIGKILL` | Killed by OS (OOM Killer). Inspect host RAM usage. |
| `143` | `SIGTERM` | Process received SIGTERM termination signal. |
| `42` | `H6_STRUCTURAL_GATE_ABORT` | Aborted by H6 structural gate due to non-finite loss or severe collapse. |
| `1` | `CUDA_OOM` / `DATALOADER_WORKER_FAILURE` | Insufficient VRAM or worker crash. Check `train.log`. |

---

## 6. How to Resume Execution

If training or testing is interrupted, invoke the resume script:

```bash
cd /home/ai4/caohuy/ACD-CLIP-phase4
bash scripts/phase4/resume_p1_v8_2_full20_train_then_test.sh
```

- Training resumes automatically from the latest complete `adapter_${EPOCH}.pth` checkpoint.
- Completed test epochs (`test_epoch_${EPOCH}/metrics.json`) are preserved and will not be re-evaluated.

# P1-v8.2 Pre-Launch OOM, Memory, and Settings Audit Report (Resolved)

**Repository**: `/home/ai4/caohuy/ACD-CLIP-phase4`  
**Branch**: `phase4-progress1-cops-dynamic-prompt`  
**Commit HEAD**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`  
**Date**: 2026-08-07  
**Final Pre-Launch Decision**: `PRELAUNCH_READY`

---

## 1. Executive Summary & Resolution of Blockers

- **Final Pre-Launch Decision**: `PRELAUNCH_READY`
- **Metric Audit Status**: `METRICS_READY` (17/17 metric unit tests passed)
- **Loss Calibration Status**: `READY_FOR_ITERATION_D` (Approved lambdas: $\lambda_{\text{route}} = 0.02356$, $\lambda_{\text{factor}} = 0.28360$, $\lambda_{\text{actual}} = 0.21270$)
- **Preflight Verification**: `PREFLIGHT CHECK PASSED`
- **Unit Test Suite**: 39 passed in 2.79s (0 failures)
- **Execution Confirmation**: **NO full training, NO full testing, NO validation, and NO epochs 10–20 evaluation were executed.**

### Resolved Configuration Blockers:
1. **`center_spread` Flags**: Added `--h6_local_factor_mode center_spread`, `--h6_local_center_mix 0.05`, and `--h6_local_factor_spread 0.10` to `train.py` and `test.py` parsers and launch scripts, replacing default `legacy_mix`.
2. **DFG Beta Schedule**: Fixed as `--dfg_beta_schedule fixed` with $\beta = 0.10$, maintaining 100% exact parity with the 120-image Candidate-1 calibration setup.
3. **Progress Version String**: Explicitly confirmed `--h6_progress_version P1-v8-minimal` as the authoritative valid choice in `train.py` parser for all P1-v8 variants.
4. **Test RAM Safety Safeguard**: Added `--external_exact_pixel_metrics` to `run_p1_v8_2_full_test_e10_e20.sh` for disk-backed chunked pixel AUROC/AP evaluation across 100% of VisA test split.

---

## 2. Resource Snapshot & Hardware Capacity

- **GPU Model**: NVIDIA GeForce RTX 5060 Ti (Driver: 580.173.02, CUDA: 13.0)
- **Total VRAM**: 16,311 MiB (~16.0 GiB)
- **Idle VRAM**: 889 MiB (15.4 GiB free)
- **Host RAM**: 31 GiB Total (18 GiB Available, 11 GiB Free)
- **Swap**: 8.0 GiB (4.7 GiB Free)
- **Storage**: 363 GiB Free on `/dev/nvme0n1p2`
- **Shared Memory `/dev/shm`**: 16 GiB Total (16 GiB Free, 1% used)

---

## 3. Controlled Memory Probes & Measured Headroom

### 3.1 Controlled Train Memory Probe (6 Microbatches = 1 Accumulation Window)
- **Setup**: `batch_size = 1`, `grad_accum_steps = 6`, `precision = bf16`, `img_size = 518`, Candidate-1 settings with `--h6_local_factor_mode center_spread`.
- **Peak Reserved GPU Memory**: **5,198.0 MB (32.8% of VRAM)**
- **Measured GPU Headroom**: **`67.18%`** (Requirement $\ge 15.0\%$ ✅)
- **Host RSS Memory**: Start 841.2 MB $\to$ End 3,682.9 MB (stable, no linear per-microbatch leak)
- **Optimizer Step**: Successfully executed at microbatch 6 with finite gradients.
- **Train Gate Status**: `PASSED`

### 3.2 Controlled Test Memory Probe (2 Test Samples)
- **Setup**: 2 test samples evaluated through test pipeline.
- **Peak Reserved GPU Memory**: **2,872.0 MB (18.1% of VRAM)**
- **Measured GPU Headroom**: **`81.87%`** (Requirement $\ge 15.0\%$ ✅)
- **Post-Cleanup VRAM**: 113.0 MB (Buffers cleanly released)
- **Test Gate Status**: `PASSED`

---

## 4. Authoritative Resolved CLI Launch Commands

### Exact Resolved Train Command:
```bash
python train.py \
  --save_path runs/phase4/p1_v8_2_full20_seed0 \
  --dataset VisA \
  --img_size 518 \
  --epoch 20 \
  --batch_size 1 \
  --cuda_device 0 \
  --grad_accum_steps 6 \
  --num_workers 2 \
  --seed 0 \
  --precision bf16 \
  --n_groups 3 \
  --image_adapt_weight 0.2 \
  --text_adapt_weight 0.2 \
  --lora_rank 16 \
  --lora_alpha 2.0 \
  --conv_lora_rank 8 \
  --conv_lora_alpha 2.0 \
  --conv_kernel_size_list 3 5 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule fixed \
  --h6_progress 1 \
  --h6_progress_version P1-v8-minimal \
  --h6_global_text_mode hard_anchor \
  --h6_local_factor_mode center_spread \
  --h6_local_center_mix 0.05 \
  --h6_local_factor_spread 0.10 \
  --h6_prediction_routing dense \
  --h6_num_factors 4 \
  --h6_top_k 2 \
  --h6_bank_dim 256 \
  --h6_router_dim 128 \
  --no-h6_expert_enabled \
  --no-h6_load_bias_enabled \
  --no-h6_cluster_responsibility \
  --lambda_h6_route 0.023563732085236152 \
  --lambda_h6_factor_role 0.2836047825589712 \
  --lambda_h6_actual_local 0.2127045363418866 \
  --lambda_h6_balance 0.0 \
  --lambda_h6_center 0.0
```

### Exact Resolved Test Command (for Epoch `${EPOCH}`):
```bash
python test.py \
  --save_path runs/phase4/p1_v8_2_full20_seed0 \
  --epochs "${EPOCH}" \
  --dataset VisA \
  --img_size 518 \
  --batch_size 1 \
  --cuda_device 0 \
  --num_workers 2 \
  --n_groups 3 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule fixed \
  --h6_progress 1 \
  --h6_progress_version P1-v8-minimal \
  --h6_global_text_mode hard_anchor \
  --h6_local_factor_mode center_spread \
  --h6_local_center_mix 0.05 \
  --h6_local_factor_spread 0.10 \
  --h6_prediction_routing dense \
  --external_exact_pixel_metrics \
  --h6_num_factors 4 \
  --h6_top_k 2 \
  --h6_bank_dim 256 \
  --h6_router_dim 128 \
  --no-h6_expert_enabled \
  --no-h6_load_bias_enabled \
  --no-h6_cluster_responsibility
```

---

## 5. Command for Manual Launch

### Master Pipeline Launcher (To be run by user when ready):
```bash
cd /home/ai4/caohuy/ACD-CLIP-phase4
bash scripts/phase4/run_p1_v8_2_full20_train_then_test.sh --execute
```

---

## 6. Final Explicit Confirmation

**NO training, NO testing, NO validation, and NO evaluation of epochs 10–20 were executed during this task.**
Preflight checks and controlled memory probes verified readiness and headroom statically and experimentally under strict isolation constraints.

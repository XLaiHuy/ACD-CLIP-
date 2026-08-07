# P1-v8.2 20-Epoch Script Build & Calibration Report

**Repository**: `/home/ai4/caohuy/ACD-CLIP-phase4`  
**Branch**: `phase4-progress1-cops-dynamic-prompt`  
**Commit HEAD**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`  
**Date**: 2026-08-07  
**Final Build Decision**: `SCRIPTS_READY_NOT_RUN`  
**Metric Audit Decision**: `METRICS_READY`  
**Calibration Status**: `READY_FOR_ITERATION_D` (APPROVED)  
**Preflight Status**: `PREFLIGHT_CHECK_PASSED`  
**Prelaunch Audit Status**: `PRELAUNCH_READY`

---

## 1. Approved Calibration Summary (Iteration C Windowed Accumulation)

- **Images Evaluated**: 120 images from VisA training split
- **Gradient Accumulation Window**: `grad_accum_steps = 6` (20 accumulation windows)
- **Calibrated Lambdas**:
  - `lambda_h6_route`: `0.0235637` (1.5% target share)
  - `lambda_h6_factor_role`: `0.2836048` (2.0% target share)
  - `lambda_h6_actual_local`: `0.2127045` (1.5% target share)
- **Split-Half Stability**:
  - Route Lambda diff: `0.32%` ($\le 20.0\%$ ✅)
  - Factor Role Lambda diff: `6.46%` ($\le 20.0\%$ ✅)
  - Actual Local Lambda diff: `6.46%` ($\le 20.0\%$ ✅)
- **Total Auxiliary Share**: Mean `5.00%`, P95 `6.36%` ($\le 8.5\%$ ✅)
- **Config Update**: Approved values written to `configs/phase4/p1_v8_2_candidate1.json` with `"calibration_decision": "READY_FOR_ITERATION_D"`.

---

## 2. Resolved Configuration Blockers

1. **`center_spread` Flags**: Added `--h6_local_factor_mode center_spread`, `--h6_local_center_mix 0.05`, and `--h6_local_factor_spread 0.10` to `train.py` and `test.py` parsers and launch scripts, replacing default `legacy_mix`.
2. **DFG Beta Schedule**: Fixed as `--dfg_beta_schedule fixed` with $\beta = 0.10$, maintaining 100% exact parity with the 120-image Candidate-1 calibration setup.
3. **Progress Version String**: Explicitly confirmed `--h6_progress_version P1-v8-minimal` as the authoritative valid choice in `train.py` parser for all P1-v8 variants.
4. **Test RAM Safety Safeguard**: Added `--external_exact_pixel_metrics` to `run_p1_v8_2_full_test_e10_e20.sh` for disk-backed chunked pixel AUROC/AP evaluation across 100% of VisA test split.

---

## 3. Preflight Verification Status

Running `tools/preflight_p1_v8_2_full20.py --stage all`:
```text
======================================================================
PREFLIGHT CHECK FOR P1-V8.2 FULL 20-EPOCH (ALL STAGE)
======================================================================
[OK] Config loaded: configs/phase4/p1_v8_2_candidate1.json
[OK] Config SHA-256: e112dd9a09fa948c70f217c213ceb02bc0048a485c093f02fcf77943936fffa1
[OK] Dataset hub file verified: dataset/hub/VisA.jsonl
======================================================================
PREFLIGHT CHECK PASSED
======================================================================
```

---

## 4. Metric Audit Findings & Test Results

- **Image Anomaly Score**: $\text{pred\_image} = 0.9 \times P(\text{abnormal}) + 0.1 \times \max(\text{pixel\_preds})$
- **Pixel Anomaly Score**: Continuous score map `seg_pred` output by `vision_text_fusion_gate_seg`.
- **Score Direction**: Larger score strictly means "more anomalous".
- **Metric Unit Tests (`tests/test_p1_v8_2_metrics.py`)**: 17 unit tests passed cleanly (0 failed).
- **Prelaunch Unit Tests (`tests/test_p1_v8_2_prelaunch.py`)**: 22 unit tests passed cleanly (0 failed).

---

## 5. Generated Scripts and Tools

1. `tools/preflight_p1_v8_2_full20.py`: Preflight validation tool.
2. `scripts/phase4/run_p1_v8_2_full20_train.sh`: 20-epoch training launcher (`adapter_1.pth` ... `adapter_20.pth`).
3. `scripts/phase4/run_p1_v8_2_full_test_e10_e20.sh`: Full-test launcher for epochs 10–20 on 100% of test split.
4. `tools/summarize_p1_v8_2_full_test_epochs.py`: Metric aggregator across epochs 10–20 without best-epoch selection.
5. `scripts/phase4/run_p1_v8_2_full20_train_then_test.sh`: Master orchestrator script.
6. `scripts/phase4/resume_p1_v8_2_full20_train_then_test.sh`: Optional resume launcher.
7. `scripts/phase4/README_P1_V8_2_FULL20.md`: Comprehensive user guide.

---

## 6. Static Syntax-Check Results

- `bash -n scripts/phase4/run_p1_v8_2_full20_train.sh`: PASSED
- `bash -n scripts/phase4/run_p1_v8_2_full_test_e10_e20.sh`: PASSED
- `bash -n scripts/phase4/run_p1_v8_2_full20_train_then_test.sh`: PASSED
- `bash -n scripts/phase4/resume_p1_v8_2_full20_train_then_test.sh`: PASSED
- `py_compile tools/preflight_p1_v8_2_full20.py`: PASSED
- `py_compile tools/summarize_p1_v8_2_full_test_epochs.py`: PASSED

---

## 7. Manual Execution Command (To be run by user when ready)

```bash
cd /home/ai4/caohuy/ACD-CLIP-phase4
bash scripts/phase4/run_p1_v8_2_full20_train_then_test.sh --execute
```

---

## 8. Explicit Confirmation

**NO training, NO testing, NO validation, and NO generated launcher script was executed during this task.**
Preflight verification passed statically. Controlled probes verified VRAM and host RAM safety. The master script is ready for manual user launch.

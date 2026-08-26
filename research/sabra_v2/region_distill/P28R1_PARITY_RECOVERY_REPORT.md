# P28R1 PARITY RECOVERY REPORT

## Scope

This is an engineering-only recovery of the failed P28 student replay. No new P28 scientific attempt was created before parity qualification. No held GT or mask was read; no model was trained; no optimizer, CLIP, or Phase2B forward was run.

## Original failure

The failed P28 replay reported candle native `max_abs=0.0` and student `max_abs=0.00022339820861816406`, exceeding the frozen tolerance `0.00002`.

## Forensic result

The first divergence is `adapter_region_residual`. P27 immutable evaluation used `--batch-size 1`, while the failed P28 replay defaulted to batch size 4. The adapter GPU path is numerically batch-size dependent. The identified engineering root-cause class is `OTHER_EXACTLY_IDENTIFIED_ENGINEERING_CAUSE`.

The candle batch-4 versus batch-1 forensic maximum errors were: region `0.0014760494232177734`, patch `0.001348257064819336`, corrected stage logits `0.0006742477416992188`, post-blur `0.00058746337890625`, resized `0.00058746337890625`, stage mean `0.0005869865417480469`, and final map `0.00022339820861816406`. Batch-size-1 replay versus the immutable native and student maps was exactly zero.

## Minimal recovery change

The P28 replay path now fixes its replay batch size to the P27 immutable evaluation batch size (`1`) and rejects any other requested batch size. P27 scientific code, artifacts, checkpoints, metrics, protocol, teacher semantics, region size, architecture, and deployment semantics were not changed.

## Qualification

The cache-only qualification replayed all 12 frozen held classes sequentially using Tier-A `seg_features`, Tier-A `native_logits`, the frozen P27 adapter checkpoint, and immutable P27 prediction artifacts. Native and student maps passed at max absolute error `0.0` for all 12 classes. The unchanged tolerance was `0.00002`. Qualification runtime was approximately 126.77 seconds. GT reads, mask reads, new CLIP forwards, new Phase2B forwards, training steps, and optimizer steps were all zero.

Per-class evidence is in `P28R1_PARITY_TABLE.csv`; boundary evidence is in `P28R1_PARITY_FORENSIC.json`; machine-readable qualification evidence is in `P28R1_PARITY_QUALIFICATION.json`.

## Status before scientific continuation

Parity is recovered and eligible for a separately labeled `P28R1 — RECOVERED MECHANISM DIAGNOSTIC`. No mechanism conclusion is made by this engineering report.

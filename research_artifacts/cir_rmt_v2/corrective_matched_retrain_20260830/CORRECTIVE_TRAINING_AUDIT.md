# Corrective matched retrain audit

Status: PASS for both VisA training runs and the completed exact Medical matrix.

## Identity and protocol

The matched pair uses source VisA at `/home/ai4/caohuy/data/VisA_20220922`, seed 0, the frozen ViT-L/14 336px asset (SHA256 `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`), FP32 with AMP and TF32 disabled, micro/effective batch size 6, and one accumulation step. Parent canonical config SHA256 is `d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb`; CIR canonical config SHA256 is `064e8acd4369645f631030b5d60abf8615e878b50e9caff6a4a8b2439b64f81c`. The architecture freeze SHA256 is `f6de6ee8f1998f591c077efeff50fa9741a9f8bad34603ba145ec54ef961ba86`.

The protocol ledger has 46 rows: every training, precision, optimizer, scheduler, loss, prompt, DFG, source, checkpoint, and evaluator field is `MATCHED`; the only intentional difference is the CIR/RMT mechanism and its architecture/config identity.

## Training completion

Parent Phase2B and corrected CIR both completed epochs 1–20 with candidate checkpoints E10, E12, E14, E16, E18, and E20. Every candidate checkpoint is present, finite, and has 113 optimizer-state entries. Global optimizer steps are identical at each candidate: 3610, 4332, 5054, 5776, 6498, and 7220.

The three parameter groups are `image_adapter`, `text_adapter`, and `soft_prompt`. Adam settings are betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `0`; base learning rates are `1e-3`, `5e-4`, and `1e-4`. The image/text ratio is 2:1 and the prompt base rate is 0.1× image. The prompt is frozen for epochs 1–3 and trainable from epoch 4. Gradient clipping is 1.0 per training step. The loss is `cls_loss + seg_loss + 0.001 * kg_loss + 0.0 * k_loss`.

Both runs use `StepLR(step_size=1, gamma=0.9)`. The corrected trainer calls `scheduler.step()` once after each epoch and before the epoch history row and candidate checkpoint save. Candidate checkpoint scheduler states are last epochs 10, 12, 14, 16, 18, and 20 with step counts 11, 13, 15, 17, 19, and 21. Parent and CIR candidate optimizer learning rates match exactly at every candidate, including the constant prompt group after its freeze policy.

| candidate | image LR | text LR | prompt LR | scheduler last_epoch | scheduler _step_count |
|---:|---:|---:|---:|---:|---:|
| E10 | 3.486784401e-4 | 1.743392200e-4 | 9.000000000e-5 | 10 | 11 |
| E12 | 2.824295365e-4 | 1.412147682e-4 | 9.000000000e-5 | 12 | 13 |
| E14 | 2.287679245e-4 | 1.143839623e-4 | 9.000000000e-5 | 14 | 15 |
| E16 | 1.853020189e-4 | 9.265100944e-5 | 9.000000000e-5 | 16 | 17 |
| E18 | 1.500946353e-4 | 7.504731765e-5 | 9.000000000e-5 | 18 | 19 |
| E20 | 1.215766546e-4 | 6.078832730e-5 | 9.000000000e-5 | 20 | 21 |

The training regression suite passed (`49 passed`); the new schedule/checkpoint tests passed (`7 passed`). The earlier mixed test failure was the pre-existing missing `p5f_geometry` dependency and was not part of this run. The real CIR smoke passed with finite RMT transport and active RMT. Resume behavior is covered by the regression tests; no resume was needed for either completed run.

## Source decomposition gate

The source matrix completed atomically with 18 valid cells and verified hashes: P, corrected CIR alpha 0 (C0/native deployment), and corrected CIR alpha .5 (C05/RMT deployment), for E10/E12/E14/E16/E18/E20. The compact result is `corrected_source_decomposition.csv`. C0 is generally competitive with or above P in source AP, while C05−C0 is near zero; this is an inference effect conditional on the corrected CIR representation, not a clean training comparison against P.

The first source harness attempt failed before a scientific cell because it passed the CIR config to the parent constructor (`KeyError: img_size`). It produced no completed cell and is excluded from the scientific archive. The resumed source run used the parent config for P and passed.

## Medical admission and status

The bounded preflight used E10 CIR on 24 balanced Brain images across four batches, full 518×518 pixel spools, exact metrics after worker/model teardown, and the same frozen forward/evaluator path. It passed with `SAFE` admission, bounded VRAM/RSS, clean worker shutdown, a 1024 soft FD limit with 43 FDs after teardown, and approximately 178.8 GiB free disk before/after. The conservative full Brain cell spool estimate is approximately 14.9 GiB. This was a resource gate, not a target result.

The full Medical matrix completed atomically: 108/108 P/C0/C05 method×epoch×target cells have hashed JSON cells and journal entries. No target tuning, MVTec training, precision change, resolution change, or automatic retry was used.

The compact decomposition is corrected_medical_decomposition.csv; its 36 target rows preserve all six targets at all six candidate epochs, and its six macro rows are unweighted across targets. Colon image AUROC/AP are blank because the frozen evaluator defines those image metrics as undefined for Colon targets.

One harness metadata caveat is preserved rather than hidden: the unmodified evaluation runner writes the CLI CIR config hash into P-cell JSON metadata. P was constructed with parent_config and the parent checkpoint, and the compact ledger records the correct parent/CIR config identities. This is a metadata-labeling defect only; no score or forward-path mismatch was found.

## Post-corrective diagnosis status

The scheduler confound is removed: parent and CIR candidate optimizer states, three-group learning rates, StepLR state, global steps, precision, and checkpoint timing match at E10/E12/E14/E16/E18/E20. Corrected C0 is not close to P on Medical pixel transfer: C0 pixel AUROC/AP trails P at every macro epoch, while C05−C0 is practically zero. The complete post-corrective case classification and bottleneck evidence are in POST_CORRECTIVE_DIAGNOSIS.md, BOTTLENECK_FINGERPRINT.json, and CASE_A_K_DECISION.json.

## Audit conclusion

Training equivalence is PASS. The scheduler correction is included in the already-pushed prerequisite commit 042174cdc63d9cb635566a1dae5b774056045383; no additional architecture, loss, optimizer, RMT, deployment, or training change is included in this result snapshot. The current evidence supports PHASE2B_REPRESENTATION_PRESERVATION; it does not support attributing the residual Medical gap to inference RMT alone.

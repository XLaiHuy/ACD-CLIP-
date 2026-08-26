# P30R1 Implementation Report

Decision: `PASS_TO_STAGE2_PROTOCOL`.

`P30R1 implementation is engineering-qualified. Scientific Stage 2 remains unstarted.`

This is an engineering/mechanism qualification only. No P30R1 scientific Stage 2, score, execution marker, Stage 3, or full experiment was started.

## Scope and frozen inputs

The authoritative preregistration, preflight, and design note were read before implementation. `P30R1_RESEARCH_REPORT.md` was absent; the preregistration/preflight/design artifacts were present, internally consistent, and hash-validated, so implementation proceeded as directed. The run used:

- branch `research/p29r1-fast-objective-forensic-v1`;
- base commit `3cf91fb4325b9d3aea3b8a65d131cdea14d8ceba`;
- preregistration SHA-256 `ceff5944e3602f8b640e7ace02a5b99962244f71eeefe6cb4882c4cf3af92a43`;
- formulation hash `290aae42e04d9faae5a10b929eb58aa0da066b5dbd248b3fee40f20e9094781c`;
- immutable P27 cache root `/workspace/p27r1_cache_v1` and the frozen P26/CLIP/config provenance.

## Frozen formulation implemented

`tools/sabra_v2/p30r1_objective.py` implements exactly one loss:

```text
student: [3,B,9,9], teacher: [B,9,9]
t_bar = teacher / C, s_bar = student / C
a_t = stop_gradient(sqrt(mean(t_bar^2 over 243 coordinates) + eps^2))
z_t = t_bar / a_t, z_s = s_bar / a_t
L = mean SmoothL1(z_s, z_t; beta=1.0)
```

The frozen constants are `C=4.960109710693359`, `eps=0.01`, beta `1.0`, and mean reduction over `B*243`. The teacher is detached and staged across the three stages; the same teacher-derived denominator is used for both sides; there is no student self-normalization, no zero-target filtering, and no auxiliary loss.

The cached trainer retains the frozen AdamW schedule (`lr=0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`), batch size `1`, seed `0`, FP32, and deterministic warn-only policy. Its engineering CLI exposes only smoke/microprofile/profile modes and never a scientific stage.

## P29 value-only equivalence audit

P29's `value` component is exactly `SmoothL1(student/C, teacher/C)` with mean reduction over the staged `3*B*9*9` tensor, when given the same staged teacher. P30R1 is therefore not identical to P29 `value-only`, not equivalent up to a fixed constant, and not merely a reduction-semantics variant.

The exact distinction is the per-sample teacher RMS denominator `a_t = sqrt(mean((teacher/C)^2)+eps^2)`, detached and applied to both student and teacher. This denominator is data-dependent rather than fixed: it removes raw cross-sample teacher-scale weighting while preserving the student/teacher radial ratio and its restoring gradient. P30R1 also accepts the frozen `[B,9,9]` teacher and stages it inside the objective. It is a genuine mechanism distinction from P29's value term, while remaining a deliberately clean single-objective causal test; no novelty claim is made.

## Production/reference parity

The production function was compared with the frozen preflight reference on ordinary random, scale-mismatch, zero-teacher, near-zero-teacher, and heavy-tail-corruption cases. All loss, normalized-output, denominator, and student-gradient maximum absolute errors were `0.0`, within `rtol=1e-6`, `atol=1e-7`; teacher gradients were absent in every case.

The deterministic preflight also passed all 11 synthetic checks. Its zero-target gradient had L2 `1.2933186292648315` and positive restoring dot product `57.73551940917969`; the near-zero cases were finite and below the fixed gradient bounds. The mixed nonzero-scale max/median gradient ratio was `59.09512710571289`, below the frozen bound `100`.

## Files changed

- Added the isolated objective, frozen-contract audit, engineering-only cached trainer, and engineering runner: `p30r1_objective.py`, `p30r1_contract.py`, `train_region_distill_p30r1_cached.py`, and `run_p30r1_engineering.py`.
- Added/retained the preregistration, preflight evidence, and design note; no scientific result artifact was created.
- Added P30R1 objective, execution-contract, and preflight tests.
- Added optional `load_source_mask`/`load_native_logits` switches to `region_cache.py`; defaults remain unchanged, and the P30R1 trainer uses them only to avoid unused tensor copies.
- Added the cache regression test for omitted mask/native-logit fields.
- No P30 objective, P30 trainer, adapter architecture, student-forward implementation, deployment/inference path, or existing P30 evidence was modified.

## Tests and exact production-path smoke

Syntax/import compilation and `git diff --check` passed. The P30R1 tests passed `17/17`. The relevant regression group passed `59/59` with one pre-existing `pkg_resources` deprecation warning.

The smoke used the real CLI runner and cached trainer path:

```text
runner -> CachedSourceDataset -> Tier-A seg features/Tier-B teacher -> RegionResidualAdapter
       -> P30R1 objective -> backward -> AdamW -> checkpoint -> strict reload
```

The one-step smoke completed one optimizer step with finite loss and gradients. Student parameter delta was L2 `0.013964357747584784` (max absolute `0.000999999581836164`); teacher delta was exactly `0.0`. The smoke checkpoint reloaded strictly and produced a finite adapter-only `[3,1,9,9]` future-forward probe. The smoke used no held ground truth or mask.

The source-cache audit found 9 exact-zero teacher records out of 2162 unique source records (`0.004162812210915819`). They remain in the dataset and the code has no filtering branch. The particular one-step shuffled smoke prefix observed zero exact-zero records, which is recorded in the training completion JSON rather than being treated as evidence that zeros are absent.

## Gradient, data-access, and provenance audit

Across smoke, microprofile, and profile: loss/gradients were finite, no adapter gradient elements were missing, the teacher scale was detached, and all teacher parameters remained unchanged. Counts were:

- new CLIP forwards: `0`;
- new Phase2B forwards: `0`;
- teacher forwards: `0`;
- held GT reads: `0`;
- held mask reads: `0`;
- cache rebuilds: `0`;
- source masks/native logits loaded by the P30R1 trainer: `false`/`false`.

The engineering path trained only against the 1962 candle fit cache records and recorded the 200 held records as not read. The cache provenance, metadata, P26 checkpoint, CLIP asset, config, and parent execution hashes are retained in the qualification JSON. No MVTec or Medical data path was invoked.

## Speed qualification

Timing used CUDA events, one final synchronization, five omitted warmup steps for the full profile, and excluded DataLoader wait. The P30 baseline median was `0.006899061845615506` seconds/step; the P29 baseline median was `0.010768339969217777`.

The 5-step microprofile measured a median of `0.004944896221160889` seconds/step, `0.0002170879989862442` seconds objective time, and an objective fraction of `4.390142669875477%`. Its mean/p90 were startup-outlier sensitive (`0.07248117275238038`/`0.20808986072540287` seconds), so the warmed profile was the gate measurement; its median was already below the hard 15% overhead limit.

The warmed profile ran 5 warmup plus 40 measured steps (45 optimizer steps). Median end-to-end step time was `0.004393984079360962` seconds, p90 `0.0050072447776794435`, and mean `0.006552517700195312`. This is `-36.31041179673685%` versus P30 and `-59.19534401846949%` versus P29. Objective-only median time was `0.00022779200226068496` seconds, `5.184179053598525%` of the end-to-end median. Inference overhead is `0%` because inference/deployment code was unchanged.

## Deviations and incidents

There were no preregistration deviations and no qualification-run incidents. The implementation review removed an unnecessary per-step host synchronization from the exact-zero diagnostic and a duplicated timing assignment before the qualification run; these changes only affect engineering instrumentation, not the objective or data semantics. No cache was rebuilt and no scientific process was launched.

Detailed per-run completion JSONs and checkpoint hashes are under `P30R1_ENGINEERING_QUALIFICATION_20260826/`. The compact handoff artifacts are `P30R1_ENGINEERING_QUALIFICATION.json` and `P30R1_SPEED_PROFILE.json`.

## Final decision

All required engineering gates passed: frozen objective correspondence, production/reference parity, zero/near-zero gradient behavior, relevant regressions, exact cached CLI smoke, checkpoint reload, gradient/teacher freeze audit, data-access audit, microprofile, warmed 40-step profile, and unchanged inference path.

Scientific performance is unknown and was not assessed. Scientific Stage 2 remains completely unstarted.

`P30R1 implementation is engineering-qualified. Scientific Stage 2 remains unstarted.`

`PASS_TO_STAGE2_PROTOCOL`

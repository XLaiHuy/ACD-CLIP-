# Phase5 portable research handoff — 2026-08-18

This is a backup/handoff, not a new experiment. It is the durable context for
continuation without chat history. No model/image/mask/GT work was performed:
`NEW_MODEL_FORWARDS=0`, `NEW_GT_MASK_READS=0`, `NEW_IMAGE_READS=0`,
`TRAINING_STEPS=0`, `MEDICAL=false`.

## Authoritative state

- Repository: `XLaiHuy/ACD-CLIP-`
- Handoff branch: `handoff/phase5-20260818-portable`
- Forensic branch: `autopilot/p5-fr1ce1-a-final-forensic-reconciliation`
- Forensic HEAD: `3f41c7e6ee4db86c67781edd3c71c80fbc1daa72`
- Protocol: `f9a2b83753e577a60854ac21e03626bda775c22e`
- Historical P5FR1CE1 result: `2ef784ff91b91e3b2c2c880dfaa74c02e94445d2`
- Evaluator recovery implementation: `f6570348988d257347af8b833ba08ff67ad102fe`
- Evaluator recovery freeze: `0e4278da12b79122d926d6b4ab080cf85980edb9`
- GT-free source: `cd0d22072d34804494e38fa95bc2e9d338bead7c`
- Frozen geometry implementation: `64a36d2df78ffac690f35c170770721ed1069fe1`
- Reconciliation commit: none.

Do not rewrite these branches/commits. Do not begin P5-G.

## Research goal and hard domain protocol

Primary goal: industrial-only development/training followed eventually by
frozen external zero-shot medical anomaly detection. Design/train/tune only
industrial; freeze before medical; medical is external zero-shot only; never
retrain, fine-tune, or tune thresholds after seeing medical; learned
reliability must be industrial-only; prefer relational/domain-relative
evidence; reliability and abstention matter under domain shift.

Core principle: **learn the evidence, bound the authority**. Phase5 remains
evidence/reliability research.

## Audited architecture

Visual stages `[8,16,24]`; text stages `[4,8,12]`; patch grid `37 x 37 = 1369`;
projected patch dimension `768`; native stage logits `[3,1,1369,2]`.
D_rank is the anomaly-minus-normal margin per stage, converted to an
within-image average-tie percentile, then population standard deviation across
the three stages. Deployment is native per-stage logits → Gaussian blur
`k7,sigma1` → bilinear `518x518`, `align_corners=True` → mean logits over
three stages → softmax.

`ARCHITECTURE=PASS`; `DEPLOYMENT_PARITY=PASS`;
`H6=ENABLED_BUT_INACTIVE_FOR_PHASE2B`; `D_RANK=PASS`; `B1=PASS`;
`COMMON_GEOMETRY=PASS`. H6 is not supplied to the authoritative Phase2B
predictor, so there is no foundational native-logit mismatch.

## B1 reference

Candidate peers satisfy `D_rank < image median` and every stage anomaly
percentile `<0.5`; spatial exclusion is Chebyshev distance `>3`; `K=8`.
Search uses the normalized mean of three normalized stage features. Ties are
descending cosine similarity then ascending patch index. The same eight peer
IDs are reused stage-wise. Stage reference is normalized mean peer feature;
`N_g=1-cosine(query_g,reference_g)` and `B1=mean_g N_g`. Fewer than eight
peers means invalid/abstain/evidence zero. B1 peers are trusted-ish Normal-like
peers, not proven Normal.

## Closed directions

Do not casually reopen factor/router terminal redesign, CoPS-like text
selector, Phase4V visual intervention, stage rescue/simple arbitration, local
multiscale analytic/CNN rescue, B2 full reranking, B3.1 adjacent-only,
positive-only tie projection, strict epsilon/gamma forensic sweeps, GT-trained
actionability, D0 global/Hodge relation graph, immediate spatial smoothing,
MLP/LoRA/MoE rescue of weak evidence, HRIP query-adaptive soft peer weighting
as primary direction, or CSRC as current primary direction. Existing Phase4/
Phase5 plans and audits document these closures; current CSRC is last and
fails all usefulness/alignment gates.

## Four-family study

PCRR has 8 configs, CSRC 8, ASR 6, PGM 4: 26 total. Canonical zero-tune
configs are `pcrr_witness_local_mean_mean`,
`csrc_spearman_average_tie_all_three_mean`, `asr_machine_rank_mean`, and
`pgm_sum_whitened_mean`. PCRR, CSRC, ASR, and PGM implementation statuses are
all `PASS`.

| Fold | Holdout classes |
|---|---|
| FOLD_0 | carpet, bottle, cable |
| FOLD_1 | grid, capsule, hazelnut |
| FOLD_2 | leather, metal_nut, pill |
| FOLD_3 | tile, screw, transistor |
| FOLD_4 | wood, toothbrush, zipper |

PCRR is a peer-relative witness-rank transform; CSRC is cross-stage rank
inconsistency; ASR is a centered-peer affine-subspace residual; PGM is a
centered-subspace whitened-coordinate statistic. All are geometry-only and
parameter-free.

## Mandatory numerical result table

Intervals are committed class-bootstrap 95% intervals. “Historical OOF” is
the frozen min-max selection output.

### PCRR

| Quantity | Zero-tune | Historical OOF |
|---|---:|---:|
| Config | `pcrr_witness_local_mean_mean` | fold selections below |
| matched win | `0.671687 [0.614059, 0.727298]` | `0.667476 [0.612919, 0.720437]` |
| B1 matched win | — | `0.627975 [0.550413, 0.700152]` |
| delta vs B1 | `+0.043713 [0.003528, 0.087292]` | `+0.039502 [-0.004733, 0.090019]` |
| aligned minus shifted | `+0.211391 [0.158111, 0.265575]` | `+0.208268 [0.156585, 0.260019]` |
| C_AP | `-0.002243 [-0.008458, 0.003110]` | `-0.000964 [-0.008484, 0.005941]` |
| R_pos | `+0.013889 [0.003825, 0.024242]` | `+0.013753 [0.003219, 0.024429]` |
| R_neg | `-0.018601 [-0.027297, -0.010450]` | `-0.018751 [-0.027953, -0.010833]` |

G1=true, G2=false, G3=true, G4=false; raw p `0.068725586`; Holm
`0.249023438`. Historical folds:
`pcrr_witness_local_mean_median`, `pcrr_pooled_peer_pairs_mean_mean`,
`pcrr_pooled_peer_pairs_mean_mean`, `pcrr_pooled_peer_pairs_mean_mean`,
`pcrr_witness_local_mean_median`.

### CSRC

| Quantity | Zero-tune | Historical OOF |
|---|---:|---:|
| Config | `csrc_spearman_average_tie_all_three_mean` | fold selections below |
| matched win | `0.512130 [0.481909, 0.547055]` | `0.499968 [0.485194, 0.516282]` |
| B1 matched win | — | `0.627975 [0.550413, 0.700152]` |
| delta vs B1 | `-0.115844 [-0.202722, -0.028565]` | `-0.128007 [-0.209762, -0.044549]` |
| aligned minus shifted | `+0.013697 [-0.018107, 0.056519]` | `+0.007886 [-0.006502, 0.023804]` |
| C_AP | `-0.022846 [-0.034360, -0.012762]` | `-0.023146 [-0.034037, -0.012788]` |
| R_pos | `-0.008095 [-0.023346, 0.006326]` | `-0.012137 [-0.028047, 0.004139]` |
| R_neg | `-0.036889 [-0.054946, -0.021879]` | `-0.033957 [-0.051194, -0.020784]` |

G1=false, G2=false, G3=false, G4=false; raw p `0.994873047`; Holm `1.0`.
Historical folds: `csrc_spearman_average_tie_adjacent_max`,
`csrc_spearman_average_tie_all_three_max`, then
`csrc_spearman_average_tie_adjacent_max` for FOLD_2/FOLD_3/FOLD_4.

### ASR

| Quantity | Zero-tune | Historical OOF |
|---|---:|---:|
| Config | `asr_machine_rank_mean` | fold selections below |
| matched win | `0.604951 [0.555041, 0.652102]` | `0.608074 [0.564462, 0.650823]` |
| B1 matched win | — | `0.627975 [0.550413, 0.700152]` |
| delta vs B1 | `-0.023023 [-0.065996, 0.014128]` | `-0.019901 [-0.061237, 0.017112]` |
| aligned minus shifted | `+0.117202 [0.065176, 0.169946]` | `+0.113078 [0.066036, 0.157393]` |
| C_AP | `-0.010854 [-0.019625, -0.003671]` | `-0.011721 [-0.019548, -0.004295]` |
| R_pos | `-0.000524 [-0.012892, 0.012039]` | `-0.002108 [-0.016563, 0.011073]` |
| R_neg | `-0.026354 [-0.038393, -0.015563]` | `-0.025519 [-0.036016, -0.016656]` |

G1=true, G2=false, G3=true, G4=false; raw p `0.817077637`; Holm `1.0`.
Historical folds: `asr_energy_95_mean`, `asr_energy_95_median`,
`asr_energy_99_median`, `asr_energy_95_median`, `asr_energy_95_median`.

### PGM

| Quantity | Zero-tune | Historical OOF |
|---|---:|---:|
| Config | `pgm_sum_whitened_mean` | fold selections below |
| matched win | `0.674022 [0.613286, 0.734492]` | `0.673324 [0.613195, 0.731519]` |
| B1 matched win | — | `0.627975 [0.550413, 0.700152]` |
| delta vs B1 | `+0.046048 [-0.005011, 0.101467]` | `+0.045350 [-0.003272, 0.102869]` |
| aligned minus shifted | `+0.215457 [0.155545, 0.274829]` | `+0.214076 [0.155707, 0.272627]` |
| C_AP | `+0.002873 [-0.004115, 0.010031]` | `+0.002594 [-0.004177, 0.009745]` |
| R_pos | `+0.015435 [0.003143, 0.027410]` | `+0.014688 [0.002309, 0.027492]` |
| R_neg | `-0.013078 [-0.019470, -0.007497]` | `-0.013341 [-0.019750, -0.007811]` |

G1=true, G2=false, G3=true, G4=false; raw p `0.062255859`; Holm
`0.249023438`. Historical folds: `pgm_max_whitened_mean`,
`pgm_sum_whitened_mean`, `pgm_max_whitened_mean`, `pgm_sum_whitened_mean`,
`pgm_max_whitened_mean`.

## Final certification, selection, and statistics

```text
FINAL_CERTIFICATION=SCIENTIFIC_RESULT_NOT_CERTIFIABLE
ZERO_TUNE_CERTIFICATION=CERTIFIED
OOF_CERTIFICATION=NOT_CERTIFIABLE
```

The ambiguity is raw-margin min-max versus ordinal config-rank normalization.
Exactly two selections change under ordinal sensitivity: CSRC FOLD_1
`csrc_spearman_average_tie_all_three_max` →
`csrc_spearman_average_tie_adjacent_mean`; ASR FOLD_2
`asr_energy_99_median` → `asr_energy_95_median`. PCRR changes 0; PGM changes
0; total 2. Both interpretations produce ranking
`PGM > PCRR > ASR > CSRC`, fully eligible `[]`, winner `NONE`.

Bootstrap: PASS, 2,000 class repetitions, seeds 5101–5107. Exact one-sided
sign-flip: PASS, 32,768 assignments. Holm: PASS and unchanged:
PCRR `0.249023438`, PGM `0.249023438`, ASR `1.0`, CSRC `1.0`. Sensitivity
changed some ASR/CSRC scalar intervals and raw p-values but no gate,
eligibility, ranking, or winner outcome.

## Defect ledger

- S1 `S1-PROV-001`: `gt_read=true` is set only after successful evaluation; the historical failed invocation was mask-free, with no numerical effect.
- S2 `S2-RESEARCH-001`: config sensitivity measures within-selected-config mean-minus-median rather than cross-config robustness.
- S2 `S2-RESEARCH-002`: R_neg best-config sensitivity uses maximum mean although safer R_neg is more negative. Gates are unaffected; historical research ranking `PCRR > PGM > ASR > CSRC` is not certified robustness.
- S3 `S3-SELECT-001`: selection semantics are ambiguous and result-affecting.
- S3 `S3-CHECK-001`: historical output checker repeats min-max and cannot independently detect the ambiguity.
- S4: none. S5: none.

No foundational evidence-construction defect or predictor/deployment defect was
detected.

## Robust conclusions and next hypothesis

PGM is strongest empirically under both interpretations but is not a
statistically supported winner: historical delta `+0.045350`, CI
`[-0.003272,0.102869]`, raw p `0.062255859375`, Holm p `0.2490234375`,
G2=false, G4=false. PCRR has alignment-grounded signal and certified zero-tune
evidence but no full support. ASR reduces negative-risk capture but fails full
usefulness. CSRC is weakest and closed as current primary direction, without
claiming the concept impossible in general. No family satisfies G0-G4 plus
multiplicity significance; no final MVTec config exists.

Next hypothesis, not started: parameter-free peer geometry, especially PGM and
PCRR, may contain alignment-grounded evidence. Study evidence authority,
reliability, and abstention:

```text
Phase2B/B1 + canonical PGM relational rarity + canonical PCRR peer calibration
    -> industrial-only reliability/authority mechanism
    -> use / suppress / abstain
```

Freshly preregister ordinal/min-max semantics, ties, corrected research
robustness, industrial dev/validation boundaries, untouched industrial
validation, and no medical access. Do not call PGM a selected winner.

## Earlier Phase5 results

### HRIP / E0RC

Operational and G0 were PASS; G1=PASS, G2=FAIL, G3=PASS, G4=FAIL.
HRIP matched win `0.6932912850`, CI `[0.6060009350,0.7773992426]`,
supportive `10/12`; B1 matched `0.7005508780`, CI
`[0.6160589830,0.7790048563]`; HRIP-minus-B1 `-0.00725959294`, CI
`[-0.052018198,0.0328305343]`, positive `8/12`. Aligned-minus-shifted
`+0.2405858566`, CI `[0.1720132350,0.2961238718]`, aligned better `11/12`.
C_AP `-0.01590103315`, CI `[-0.03908224835,0.00499908072]`; R_pos
`-0.00514820744`, CI `[-0.04970847046,0.03639588178]`; R_neg
`-0.0206419190`, CI `[-0.02797440574,-0.01474099187]`.
Terminal `HRIP_NOT_BETTER_THAN_B1_CENTROID`; E1_AUTHORIZED=false;
CANDIDATE=NONE. Do not rescue HRIP with tau, stage attention, hard-NN, or
learned weighting.

### Earlier B1 evidence

Coverage 100%; supportive 11/12; matched win approximately `0.6990785`, CI
`[0.6132139,0.7832789]`; shifted approximately `0.4270157`; fraction A1
oracle recovered approximately `0.52035`; posthoc GT contamination approximately
`0.0173%`. B1 peers are trusted-ish, not proven Normal.

## Commit map

`1d636895e9f6f299d14f92ae150aebf4e3b4fb80` P5 reconciled head;
`02a56c8997c9c5f14ecab485da84d0d644d82ebd` P5-F protocol;
`a2c6d51b859a8259291fe06e0195bbde70ac5816` P5-F implementation;
`859d4e2c8074f05ffb2056e4efe5fc38657763bb` P5-F invalid audit;
`ca65e8bddaca5acb56b8918cd5f95270fa95dbb2` P5-FR1 protocol/freeze;
`64a36d2df78ffac690f35c170770721ed1069fe1` P5-FR1 implementation;
`529096949afae2c6bad2b73d976a6f8cf0455e3d` P5-FR1 invalid audit;
`779aace6f0decdb6bb4de7cdb5df4c96a22e53ff` P5-FR1C freeze;
`ecef06b5ad7d69426b1cabdcfe4869f7615f9101` P5-FR1C provenance fix;
`cd0d22072d34804494e38fa95bc2e9d338bead7c` P5-FR1C GT-free freeze;
`0e4278da12b79122d926d6b4ab080cf85980edb9` recovery freeze;
`f6570348988d257347af8b833ba08ff67ad102fe` recovery implementation;
`2ef784ff91b91e3b2c2c880dfaa74c02e94445d2` result;
`f9a2b83753e577a60854ac21e03626bda775c22e` forensic protocol;
`3f41c7e6ee4db86c67781edd3c71c80fbc1daa72` forensic audit. Reconciliation:
none.

## External assets, checkpoint, and runtime

MVTec images/masks are not committed. See
`handoff/EXTERNAL_ASSET_MANIFEST.{md,json}` for archive SHA
`61124d44b1e62ad0dc64e1b6111c7ffcfda20cd36a92f68e14df0a8016cf477b`, metadata
SHA `3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c`,
canonical identity SHA `c0ace7f629a636db6393aca7bebe1b37a6a9f5673ff59ff8b6800484642faa34`,
1,725 records, and 15 classes.

Required Phase2B checkpoint: SHA256
`a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34`,
56,451,915 bytes. It is already remote in Git LFS at
`origin/artifacts/p5-runtime-inputs`, commit
`316ce9d4a9ddf742cf17f1c98c5011891c90ab08`, path
`runs/phase4v/v1_7/readiness_full/adapter_5.pth`; it is not duplicated here.
Config SHA: `377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4`.

Large caches remain only on the rental filesystem and are documented, not
deleted, in `handoff/MANUAL_TRANSFER_REQUIRED.md`. Known all-config aggregate
hash: `3f6bc75c3c3a0eeed7a72b457d1e060a9aed55186e905b0ed1e9ad639be23b67`.

Read this file first on another machine, then `NEXT_MACHINE_START.md`, then
the committed `runs/phase5/hsir/P5FR1CE1A_FINAL_FORENSIC/FINAL_REPORT.md` and
its JSON audits. Do not rerun old evaluators, derive 1,725 records, reopen
medical, or begin P5-G.

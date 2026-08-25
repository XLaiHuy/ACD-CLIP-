# SABRA FINAL — P26 ARCHITECTURE FREEZE

## 1. Exact final architecture

`SABRA-FINAL-NATIVE-PHASE2B-V1` is the frozen Phase2B detector with native
postprocessing and no active SABRA/CURE correction. The policy always returns
`KEEP`; correction coverage is zero. The complete machine-readable contract is
`SABRA_FINAL_CONFIG.json`.

## 2. Why retained components remain

Frozen Phase2B is the common, fully specified source detector used throughout
the valid lineage. Native fallback has no dependence on an unvalidated
benefit/value selector. The learned Phase2B image adapter, text adapter, and
soft prompt are retained inside the verified checkpoint.

## 3. Why dropped components were removed

R1 magnitude failed its MAE gate; R2 and R2-v2 did not establish positive pAP
breadth; P14 image value was weak; P23 coarse image actions missed the required
headroom; and P25R3 validly found the frozen patch-benefit formulation not
identifiable. Direction and harm-risk remain supported scientific findings but
are disabled because safety without identifiable benefit is not a complete
deployable correction policy.

## 4. Exact parent evidence

The terminal parent is P25R3 commit
`c8f505aa69b581afffead83db9b146df53179ce4`. See
`P26_EVIDENCE_LEDGER.json` and `P26_COMPONENT_ADJUDICATION.md` for the exact
lineage and claim boundaries.

## 5. Final inference path

RGB image -> bicubic 518x518 -> CLIP normalization -> frozen Phase2B visual and
text forward -> three native stage logits -> per-stage Gaussian blur k7/sigma1
-> bilinear 518x518 (`align_corners=True`) -> mean logits across stages ->
softmax -> anomaly channel 1. No sidecar runs and no logit correction is made.

## 6. Required checkpoints and artifacts

The required artifacts are the canonical Phase2B config, producing config,
epoch-5 Phase2B adapter, and OpenAI CLIP ViT-L/14@336px asset. Exact paths,
sizes, hashes, Git/LFS status, and origins are in `P26_REQUIRED_ARTIFACTS.json`
and `P26_CHECKPOINT_MANIFEST.json`.

## 7. Reproduction command

```bash
python tools/sabra_cure/run_sabra_final.py --check-only
python tools/sabra_cure/run_sabra_final.py --dry-run
```

`--run` is governance-locked until a later explicit authorization.

## 8. New-machine restore command

```bash
bash scripts/restore_p26_sabra_final.sh
```

## 9. External-validation firewall

MVTec was not read in P26. Medical is forbidden. P26 performs zero CLIP
forwards, zero Phase2B steps, zero new fits, and creates no scientific attempt
marker. `EXTERNAL_VALIDATION_AUTHORIZED = FALSE`.

## 10. Next allowed action

Explicit user review, then machine restoration. Only a later explicit
authorization may permit one untouched MVTec external validation using the
unchanged P26 architecture freeze.

# Phase 4 Conditional Semantic Factorization

## Authorized scope

Stage 0 starts from commit 3c07202808c04b8d8480291a285c2bff947241c0
on branch autopilot/p4-conditional-semantic-factorization. It uses a fresh
OpenAI CLIP initialization only. No Phase1, Phase2B, or Phase4 checkpoint is
loaded.

Stage 0 implements one image-conditioned Normal/Abnormal context pair, the
explicit CLS24.detach -> VAE -> class_semantic -> class_to_context bridge,
the split compute_dfg_weights/apply_dfg_weights interface, and the
predictor-aligned abnormal residual:

    delta = L_dyn^A - stopgrad(L_base^A)
    L_final^N = L_base^N
    L_final^A = L_base^A + 0.05 * delta

There is one context-conditioning path. ACT is fixed to one (no ACT module).
The legacy Router, Center-Spread, expert, factor-role, responsibility, and
PCGrad Router paths are inactive and frozen in P4-CSF-K1.

K greater than one, OT, long training, E20, and medical evaluation are outside
Stage 0 and must not begin unless the zero-step decision is PASS.

## Verification

### Legacy test triage

`tests/test_h6_adapter_contract.py::test_adapter_legacy_and_phase4_visual_contracts`
contains an epoch-3 assertion requiring exactly two nonzero routing entries.
It fails unchanged on both the Stage 0 worktree and authoritative commit
`3c07202808c04b8d8480291a285c2bff947241c0`: the legacy default produces
dense-four probabilities at epoch 3. This is therefore classified as
`PRE_EXISTING_LEGACY_TEST_FAILURE`; its assertion is preserved and it is not
part of the P4-CSF-K1 production path.

Run:

    bash scripts/phase4/run_p4_semantic_interface_zero_step.sh

The audit uses one real VisA TRAIN sample, a fresh OpenAI CLIP model, strict
FP32, TF32 disabled, no autocast, and zero optimizer steps. The detailed result
is written to:

    runs/phase4/stage0/semantic_interface_zero_step.json

The only successful Stage 0 decision label is:

    PHASE4_INTERFACE_CLEANUP_PASS

On PHASE4_INTERFACE_CLEANUP_FAIL, stop before Stage 1.

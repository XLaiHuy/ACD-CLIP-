# Cross-trajectory Anchor reference audit

Classification: `UNKNOWN` for the strict same-trajectory test.

The RA Anchor reference is historical H2 E1:

```text
/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_1.pth
SHA256 da893014cdd9ca643f632cedbf5d43fc57eb3acee343e4795bc7de9aa12c3074
```

The new R/RA/RCA arms start from a newly seeded shared E0:

```text
/home/ai4/caohuy/ACD-CLIP-cir-dfg-rmt-v2/runs/h2_anchor_cir_master_20260901/common/e0.pth
SHA256 119ba08eb8aa8107f47bf0a62ccc1c9ee643cd1f395331a527b1c975ea1d3eca
seed 0
```

The exact new R E1 model-state checkpoint was not saved (R saved only candidate E10), so the requested historical-E1 versus new-R-E1 image-adapter L2/cosine/fixed-input comparison cannot be completed. Historical H2 also lacks a recorded seed, Python/NumPy/torch RNG state, or run-local resume state. These facts prevent certification that the historical E1 and new R trajectory are the same; they strongly motivate treating the reference as a cross-run provenance risk, but the strict classification remains `UNKNOWN` rather than an unmeasured numeric claim.

The historical H2 E10 used for the oracle comparison was retrospectively selected using Medical results. Choosing its E1 as the Anchor reference occurred before the new Medical evaluation, but it is still indirectly tied to a target-selected historical trajectory. That is a contamination risk for a strict paper claim. Future Anchor tests must derive E1 from the same shared E0 trajectory before any Medical access, and must save the E1 state/hash.

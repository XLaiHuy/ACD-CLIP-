# BF16 Seed0 disambiguation contract reconstruction

`LAST_FULL_TRAIN_AUTHORIZED=YES`; the sole allowed additional full run is
`A_BF16_Seed0_E15`. This record reconstructs its non-precision contract from
the frozen Seed0 protocol/configuration and current frozen BF16 Seed1 script.

| Contract item | Historical Seed0 A | New BF16 Seed0 A |
|---|---|---|
| Seed, source, image size, ordering | 0, VisA JSONL SHA `468463…2642`, 518, deterministic | exact match |
| Model/adapters | ViT-L-14-336; 3 groups; image/text weights .2; Conv-LoRA r8/a2/k3,5; text LoRA r16/a2 | exact match |
| DFG/SS2D | attn d256 tau8; SS2D weight-residual beta .1 warmup010 | exact match |
| Prompt | hybrid; alpha max .2; freeze 3; ctx 4 phrase init | exact match |
| A / Anchor | lambda .0021633926715180626; family cap .10; shared fresh E1 reference | exact match |
| Optimization | Adam; image/text/prompt LR .001/.0005/.00005; StepLR gamma .9; kg/k .01/.002; clip 1.0; batch 6 | exact match |
| Horizon/evaluator | E15 primary; source-first; frozen checkpoint before fixed target evaluation | exact match |
| Precision | FP16 autocast, GradScaler, legacy local FP32 islands | **intentional difference:** BF16 autocast, GradScaler off, no legacy islands |

The historical checkpoint is unavailable for direct source replay and the
historical E2–E15 trajectory had two non-finite gradient skips. The fresh
BF16 Seed0 run must therefore begin at a fresh BF16 E1, not resume an FP16
checkpoint, and must pass a 100–200-step source-only validity gate before its
single E15 training allocation is consumed.

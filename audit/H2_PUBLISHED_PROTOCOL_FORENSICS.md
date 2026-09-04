# H2 published-protocol forensics

## Scope and decision

This is a read-only audit of the already completed H2 artifacts. It does not
rerun training or target evaluation and does not change the frozen scientific
contract.

`PREVIOUS_REPLICATION_PROTOCOL_AUDIT=PASS` means that the Seed-1 and Seed-2
jobs were launched with the frozen H/A protocol and the provenance artifacts
are internally consistent. It does not mean that those trajectories are
valid: both seeds failed the hard training-validity gate.

`PUBLISHED_PROTOCOL_MATCH=PARTIAL`.

The internal H2 contract is well documented and the current H/A Seed-0
comparison uses one matched evaluator. The external ACD-CLIP README gives
headline N=3 values but does not preserve enough implementation detail to
establish exact equivalence. The historical H2 evaluator also used stride 4
and per-class rounding, whereas the frozen current evaluator uses raw exact
pixel scores at stride 1. Therefore published-gap arithmetic is contextual,
not a claim of a controlled replication.

## Field-by-field audit

| Area / setting | H2 evidence | Tag relative to the frozen internal contract | Tag relative to the public README/paper | Consequence |
|---|---|---|---|---|
| CLIP backbone | `ViT-L-14-336` in `TRAINING_CONTRACT_TABLE.csv`, config, and logs | `EXACT_MATCH` | `EXACT_MATCH` for the named backbone | No known backbone mismatch internally |
| Pretrained checkpoint | CLIP SHA256 `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` | `EXACT_MATCH` | `UNKNOWN` because the public table does not identify a checkpoint hash | External equivalence is not provable |
| Input resolution | `img_size=518` | `EXACT_MATCH` | `UNKNOWN` for the full published training/evaluation path | README exposes a test default, not the complete experiment record |
| Patch geometry | ViT-L-14-336 with H2 geometry frozen | `EXACT_MATCH` | `UNKNOWN` | No public geometry manifest was found |
| N / group interpretation | H2 `n_groups=3`; public results label N=3 but README test default is 4 | `EXACT_MATCH` | `UNKNOWN` | The N=3 mapping is plausible but not fully documented externally |
| Image adaptation | weight `.2`; Conv-LoRA rank 8, alpha 2, kernels 3 and 5 | `EXACT_MATCH` | `UNKNOWN` | Public table does not specify all training settings |
| Text adaptation | weight `.2`; LoRA rank 16, alpha 2 | `EXACT_MATCH` | `UNKNOWN` | Public table does not specify the text branch |
| Source dataset | VisA | `EXACT_MATCH` | `UNKNOWN` as a complete training-data contract | Exact sample/sampling manifest is not in the public table |
| Batch size | 6 in the H/A logs | `EXACT_MATCH` | `UNKNOWN` | Not listed in the public results table |
| Epoch horizon | E15 primary and E20 secondary | `EXACT_MATCH` | `UNKNOWN` | Public N values are not an epoch specification |
| Optimizer | Adam, separate image/text/prompt groups, zero weight decay | `EXACT_MATCH` | `UNKNOWN` | Betas and epsilon are not documented in the public table |
| Learning rates | image `.001`, text `.0005`, prompt `.00005` | `EXACT_MATCH` | `UNKNOWN` | External LR schedule cannot be reconstructed from README |
| Scheduler | StepLR, step size 1, gamma `.9`, stepped after epoch | `EXACT_MATCH` | `UNKNOWN` | A meaningful external mismatch cannot be excluded |
| AMP and clipping | AMP plus GradScaler, clip norm 1.0, checkpointing | `EXACT_MATCH` | `UNKNOWN` | Public numerical policy is not recorded |
| Prompt construction | hybrid hard/soft, phrase `a photo of a`, context 4, alpha 0/.05/.1/.2, freeze 3 | `EXACT_MATCH` | `UNKNOWN` | Public prompt details are incomplete |
| Segmentation/classification objective | main classification plus segmentation | `EXACT_MATCH` | `UNKNOWN` | Normalization and exact implementation are not specified publicly |
| KG regularizer | `.01` | `EXACT_MATCH` | `UNKNOWN` | Public loss decomposition is unavailable |
| K regularizer | detached-W_K term `.002` | `EXACT_MATCH` | `UNKNOWN` | Public detach semantics are unavailable |
| Safe Anchor | A adds lambda `0.0021633926715180626`, family cap rho `.10` | `INTENTIONAL_DIFFERENCE` from H and from original ACD-CLIP | `INTENTIONAL_DIFFERENCE` | A is an H2 factorial candidate, not the original published baseline |
| CIR | absent in H/A | `EXACT_MATCH` for the H/A comparison | `INTENTIONAL_DIFFERENCE` only for the four-arm H2 design | No CIR is included in the H/A result |
| DFG | attention dim 256, tau 8, SS2D weight-residual, beta warmup to `.1`, FP32 residual | `EXACT_MATCH` | `UNKNOWN` | Public implementation details are not sufficient |
| Train/eval mode | adapter BatchNorm semantics explicitly preserved | `EXACT_MATCH` | `UNKNOWN` | Public mode semantics are not stated |
| Test resizing/interpolation | frozen current evaluator; exact replay records the active implementation | `EXACT_MATCH` internally | `UNKNOWN` | Cannot claim public equivalence |
| Pixel stride | current exact evaluator stride 1; historical oracle stride 4 | `IMPLEMENTATION_DIFFERENCE` across H2 reports | `UNKNOWN` relative to public numbers | This alone changes pixel metrics modestly |
| Smoothing and score construction | current evaluator manifest and parity audit | `EXACT_MATCH` internally | `UNKNOWN` | Public smoothing/aggregation details are incomplete |
| Stage/DFG aggregation | current exact evaluator and config | `EXACT_MATCH` internally | `UNKNOWN` | External aggregation is not fully recoverable |
| AUROC/AP implementation | raw exact arrays, macro outputs, no target tuning | `EXACT_MATCH` internally | `UNKNOWN` | Tie handling and implementation details are not fully public |
| Rounding | current raw exact; historical per-class rounding | `IMPLEMENTATION_DIFFERENCE` across H2 reports | `UNKNOWN` | Do not compare rounded published values as exact |
| Dataset/class averaging | H2 manifests and result summaries | `EXACT_MATCH` internally | `UNKNOWN` | Public macro construction is not fully specified |

## Evaluator evidence

`audit/H2_ORACLE_EVALUATOR_PARITY.md` reproduces the historical parser's
published Phase-2B class rows and reports the current exact full-resolution
reference. Its bounded Brain replay agrees with the exact reference. This
supports evaluator correctness inside H2 and documents a small historical
stride/rounding difference. It does not prove that the public ACD-CLIP
implementation used the same evaluator.

## Interpretation

The current Seed-0 H/A contrast is internally matched. The comparison to
historical H2 is `APPROXIMATELY_MATCHED`, and the comparison to public ACD-CLIP
is also only approximate for the macro rows. A protocol/evaluator mismatch is
therefore a plausible contributor to an external gap, but it cannot explain
the invalid Seed-1/Seed-2 trajectories and cannot be used to claim a causal
explanation for Medical AP without stored score maps or a valid replication.

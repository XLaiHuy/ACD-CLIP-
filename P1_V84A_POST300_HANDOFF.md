# P1-v8.4-A post-300 handoff

Snapshot date: 2026-08-09 UTC. This handoff resumes the P1-v8.4-A forensic
calibration; it is not authorization to restart or broaden the research.

## Repository and provenance

- Repository: `/workspace/ACD-CLIP-p1v84a`
- Branch: `autopilot/p1-v84a-forensic-calibration`
- Previous authoritative base HEAD: `1b88c1e45896a2eb25b2b84264152c7cffff4004`
- Remote: `https://github.com/XLaiHuy/ACD-CLIP-`
- GPU: NVIDIA GeForce RTX 3080 Ti 12 GB; driver 590.48.01
- Python 3.11.10; PyTorch 2.5.1+cu124; CUDA 12.4
- OpenAI CLIP: `model/ViT-L-14-336px.pt`
- Required OpenAI CLIP SHA256:
  `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
- Data root: `/workspace/data/med_visa/data`
- Protocol: VisA train, seed 0, image 518, batch 1, grad accumulation 6,
  FP32, TF32 off, AMP off, gradient checkpointing on, fixed rho 0.05.

The materialized OpenAI checkpoint, VisA data, virtual environment, and model
checkpoints are machine-local and are deliberately excluded from Git.

## Completed runs

### Fresh 8-batch P1-v8.4-A

Status: **PASS** (8/8 batches, 2 optimizer steps).

- ACT probability before the first update: `0.5`
- ACT output weight norm initially/before step 1: `0`
- ACT output weight norm after step 1: `0.0027642201`
- ACT probability after step 1: `0.5010200739`
- ACT-head gradient before step 1: `0.0006116156`
- Post-step upstream ACT gradient: `0.0013633764` on batch 7, support 754
- MAIN exact-change maximum: `0`
- Residual definition, local reconstruction, and surgery reconstruction
  maximum errors: `0`

This proves that ACT is initialized as an exact no-op, leaves zero after an
optimizer step, and has a reachable upstream gradient path.

### Fresh 300-batch P1-v8.4-A attempt 1

Status: runtime/optimization **PASS**, scientific gate **NOT PASS**.

- 300/300 batches, 50 optimizer steps, runtime `325.007 s`
- No OOM and no NaN/Inf
- Peak GPU allocation: `4,119,058,432` bytes (about 4.1 GB decimal)
- MAIN exact-change: `0`; fixed rho 0.05
- Cumulative `G_local = 0.0030561143` (`0.3056%`)
- Cumulative `G_multi = 0.0027038413` (`0.2704%`)
- Router-supervised patches: `0`; informative fraction: `0`
- ACT mean `0.35722464` (normal `0.35723680`, anomaly `0.35516283`)
- ACT positive/ambiguous/negative fractions: `0 / 0.88031161 / 0.11968837`
- Absolute factor correlation mean `0.99941915`; absolute effective rank
  `1.003176`

No 1e or 3e was authorized.

## Single post-300 forward-only audit

The completed attempt-1 checkpoint was replayed once over the same seeded
VisA protocol to recover missing regional and per-patch decision evidence.
The audit constructed no optimizer, ran no backward, performed zero optimizer
steps, and mutated no model state. State hashes before and after were exactly
equal:

`12bc277e11fadcf2578f3ae7154c7cd785d16c437df80e90f5a87ed3c17be4f0`

Residual-definition and routed-correction reconstruction errors were exactly
zero.

| Region | Base | Residual BestSingle | Residual Oracle | SoftRouted | Oracle gain vs Base | All-harm |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 0.0155802639 | 0.0154780708 | 0.0154582476 | 0.0156750455 | 0.7831% | 0.000747% |
| Anomaly | 3.1753354 | 3.1632133 | 3.1614382 | 3.1690018 | 0.4377% | 0% |

Residual candidates remain useful in both regions. True residual semantics
eliminated the old anomaly all-harm pathology and must not be reverted.

## Teacher diagnoses

### ACT teacher

**PROVEN: `ACT_GAIN_GATE_SCALE_MISMATCH`.**

- Current positive threshold: `0.02`
- Observed positive best-gain maximum: `0.0109934807`
- Current positive support: `0%`
- Frozen-replay positive-gain p90: `0.00240325927734375`
- At p90, positive support is `9.9834%` overall, `9.5258%` normal, and
  `87.7360%` anomaly.

The p90 value is only a candidate future single-variable experiment, not a
finalized canonical configuration. The existing negative boundary (`gain <=
0`) supplies only `0.0007425%` negative support in the frozen replay and no
anomaly negatives. This unresolved positive/negative target-design issue must
be reviewed before any corrected 300B launch.

### Router teacher

**PROVEN: `CANONICAL_ROUTER_TEACHER_EFFECTIVELY_UNIFORM`.**

At canonical `tau_utility=0.05` and `entropy_threshold=0.98`, admitting
positive-gain candidates still produces zero informative patches. Mean teacher
entropy is `0.99942398` and mean maximum probability is `0.26095831`.

The no-training sensitivity grid is diagnostic only. For example, tau 0.02
with entropy threshold 0.98 exposes nonzero support, but no Router training
configuration change was authorized or made.

## Factor specialization

Frozen final-state audit:

- `G_local = 0.0059475508` (`0.5948%`)
- `G_multi = 0.0041860105` (`0.4186%`)
- Residual effective rank: `1.2322123`
- Dominant factor winner share: about `98.9%` overall
- Anomaly-positive winner shares: F1 `0%`, F2 `48.2151%`, F3 `34.9833%`,
  F4 `16.8015%`

Classification: **`WEAK_BUT_PRESENT`**, not `FUNCTIONALLY_COLLAPSED`.
Alternative factors have real anomaly-side utility, so extra factor capacity
is not yet authorized.

## Decision and resume order

- Decision: **`EXIT_FOR_DISCUSSION`**
- Decision branch: **D2**
- Reason: **`MULTIPLE_TEACHER_CHANGES_REQUIRED`**

ACT scale and canonical Router discriminability are both broken. Changing both
at once would destroy attribution. Future discussion should preserve this
order:

1. Isolate the ACT teacher first, including the near-zero negative-support
   issue. The frozen-replay p90 is a candidate, not a canonical setting.
2. Only after ACT is understood and frozen, investigate the Router teacher.
3. Only after teacher problems are resolved, reassess whether factor capacity
   is actually necessary.

## Evidence and source paths

- `runs/p1_v84a_gpu/fresh_8b_seed0_final/smoke_summary.json`
- `runs/p1_v84a_gpu/fresh_300b_seed0_attempt1/final_summary.json`
- `runs/p1_v84a_gpu/fresh_300b_seed0_attempt1/smoke_summary.json`
- `runs/p1_v84a_gpu/post300_root_cause_audit.json`
- `tools/audit_p1_v84a_post300.py`
- `train.py` (diagnostic-only ACT/runtime instrumentation)

Large `adapter_1.pth` files are intentionally not tracked. The compact audit
contains checkpoint provenance and hashes needed to understand the evidence.

## Evidence status

### PROVEN

- True residual semantics removed the old anomaly all-harm pathology.
- Residual Oracle improves Base in normal and anomaly regions.
- The ACT positive gate 0.02 is out of scale.
- The canonical Router teacher has zero informative support.
- The ACT runtime gradient path is alive and its initial no-op is exact.

### LIKELY

- Teacher configuration currently blocks useful learning before factor
  capacity becomes the primary bottleneck.

### NOT YET ESTABLISHED

- The best new ACT positive/negative target rule.
- The best Router tau/entropy rule.
- Whether P1-v8.4-B capacity is ultimately necessary.

## Do not rerun or start without a new decision

- Do not rerun the completed fresh 8B, fresh 300B attempt 1, or post-300 replay.
- Do not run 1e, 3e, medical, or final20.
- Do not create or run P1-v8.4-B yet.
- Do not revert residual semantics.
- Do not bundle ACT and Router changes into one experiment.

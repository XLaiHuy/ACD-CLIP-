# Selected solution implementation audit

Solution: `SELECTIVE_PHASE2B_ANCHOR`

Implementation status: PASS for bounded smoke; not authorized for a full
scientific run.

## Changed surface

- `scripts/cir_rmt/train_full.py`: adds an optional train-only normalized anchor
  term and checkpoint/resume metadata. The default coefficient is zero.
- `tools/cir_rmt/parameter_anchor.py`: loads and freezes the parent E14 image
  adapter reference and computes the normalized parameter distance.
- `tools/cir_rmt/source_solution_gate.py`: evaluates only the fixed VisA source
  sample using the unchanged repository forward/evaluator/deployment helpers.
- `tests/cir_rmt/test_parameter_anchor.py`: focused anchor and identity tests.

No architecture, RMT runtime, evaluator, Gaussian operator, configuration file,
optimizer group definition, Adam default, StepLR definition/timing, or frozen
checkpoint was changed.

## Contract checks

- Adam remains three groups: image adapter, text adapter, soft prompt.
- Base LRs and the 0.9 StepLR trajectory remain unchanged; the smoke recorded
  E01 post-step image/text/prompt LRs `9e-4/4.5e-4/0` and E02
  `8.1e-4/4.05e-4/0`.
- The anchor adds no parameter or optimizer group. Its reference tensors are
  detached and have no gradients; only live image-adapter parameters receive
  anchor gradients.
- The existing loss terms remain `cls + seg + 0.001*kg + 0*k`; the selected
  term is an optional fifth train-only term with coefficient `1e-3`.
- FP32, AMP false, TF32 false, effective batch six, prompt freeze, DFG policy,
  checkpoint policy, source, seed, CLIP asset, and deployment are unchanged.
- Resume smoke restored the anchor reference/coefficient, advanced from E01 to
  E02, and restored scheduler `last_epoch=2`, `_step_count=3`.

## Real smoke evidence

The four-step VisA smoke passed with finite loss and parameters, nonzero RMT
delta, a valid `last.pth`, and peak allocation about 9.0 GiB. A one-step resume
smoke also passed. The source-gate checkpoint hash after resume is:

`7368f3a7c9a0ea259921ff62535d4641236bb74e1f3661232cb383aab41f5b7b`

The source gate intentionally does not treat the E02 smoke metrics as a
scientific improvement/regression because P/C0 are only available at the E14
comparison horizon in the frozen compact archive.

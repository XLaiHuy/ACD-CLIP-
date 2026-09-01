# Post-fix diff review

Compared with exact source commit `e03966997d4cecfd985943a4053a93e1e40197ec`,
the tracked source changes are limited to:

- `train.py`: explicit reproducibility metadata, resumable full-state saves,
  sorted class order, successful-step accounting, optional safe anchor/CIR
  terms, and bounded smoke support.
- `model/adapter.py`: optional train-only CIR call after native logits, with
  alpha-zero/inference path unchanged.
- `test.py`: explicit evaluator mode and benchmark-exact stride/rounding
  guard plus disk-backed exact pixel accumulation; legacy defaults remain
  replay-compatible.
- `h2_clean/exact_metrics.py`: bounded-RSS full-resolution AUROC/AP using a
  private disk spool and packed external sort.
- The smoke/factorial checkpoint validators use a conditional `sys.exit`
  call, fixing the success-path `raise None` launcher defect found in smoke.
- `utils.py`: optional raw metric display mode; default rounding is unchanged.

New files are limited to the H2 contract, factorial/evaluation launchers,
configuration, tests, and audit artifacts. No dataset, CLIP weight, historical
checkpoint, or unrelated base worktree file is modified.

The diff does not change deployment geometry, inference CIR/RMT, medical image
score blending, Gaussian-noise behavior, DFG equations, or the historical
model-only replay contract. `geometry_authorized=false` remains explicit.

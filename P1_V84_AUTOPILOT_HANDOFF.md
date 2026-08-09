# P1-v8.4-A Autopilot Handoff

## Terminal status

`EXIT_FOR_DISCUSSION` — user-directed stop before development gates could be
completed.  This is not `DEVELOPMENT_GATES_PASS` and no P1-v8.4-A training
candidate is approved.

## Frozen provenance

- Worktree: `/home/ai4/caohuy/ACD-CLIP-p1v83-autopilot`
- Branch at handoff: `autopilot/p1-v83-root-cause-d58b84bc`
- Starting commit: `459e5462c92ea34b3e8c2f29c4f1727e07a2581a`
- Authoritative specification read in full: `P1_V83_POST300_AUDIT.md`
  (SHA-256 `ab450ff795c044d9478a09dd9d7faed6dcd05bbbf581f2fc75e0a6c6d24d31e9`)
- Baseline checkpoint audited:
  `runs/p1_v83_dev/corrected_300b_primary_anchored_attempt1/adapter_1.pth`
  (SHA-256 `0a53a2abb00162e1565bad1d776fc7c6be134f9a0053d6976575f3e9e9f8629b`)

## Completed evidence

1. The 300-batch forward-only forensic replay completed with no optimizer,
   backward pass, or model mutation. `forensic_summary.json` reports identical
   before/after model hashes and all gradient fields clear.

2. The exact no-op is `expected_noop_pre_expert_bank`, not the base DFG logit:
   the no-op bank error was `5.96e-08`, no-op factor-logit error was `0`, and
   the local reference vs base-margin correlation was only `0.264`.

3. The corrected v8.3 absolute-factor pathology is real on anomaly patches:
   all four absolute candidates are harmful for 100% of replayed anomaly
   patches. Residual candidates relative to exact no-op reduce residual
   all-harm to `0.0` and have positive best residual gain (`+0.00473`).

4. Absolute factor functions are common-mode dominated (mean correlation
   `0.9973`, effective rank `1.23`), while no-op residual functions separate
   materially (mean correlation `-0.2832`, effective rank `2.72`). The
   evidence supports forensic `F-A` with `F-C`: true residual semantics plus
   ACT/no-ACT is authorized; extra factor capacity was not authorized.

5. The first material collapse appears after distinct concept/query slots,
   at the prototype/state-generation path, and is then amplified in the
   structured prompt/text path. This is recorded in `collapse_trace.json`;
   it was not used to justify a P1-v8.4-B redesign.

6. P1-v8.4-A source is implemented behind the explicit
   `P1-v8.4-A` progress version:

   - factor correction is `rho * (factor_logit - exact_noop_logit)`;
   - a minimum LayerNorm+Linear ACT head uses existing router patch features;
   - final correction is continuous `ACT * rho * sum(router * residual)`;
   - factor utility uses residual candidates; ACT targets use the required
     positive/negative/ambiguous zones; router supervision remains limited to
     ACT-positive informative patches;
   - checkpoint metadata declares v8.4-A semantics and version 9;
   - the P1-v8.3 path has no ACT parameters and retains absolute semantics.

7. A no-step calibration found and fixed a real tensor-broadcasting error
   before any training: the no-op reference lacked a singleton patch axis,
   producing `[3,3,P,4]` instead of `[3,1,P,4]`. The corrected code now uses
   the explicit patch axis and has a regression test.

8. Focused CPU validation passed after the fix:

   ```text
   51 passed in 1.65s
   ```

## Incomplete work

- The fresh-init no-step v8.4-A gradient calibration was interrupted by the
  user at `16/24` natural six-microbatch windows. It has no final summary and
  no selected ACT lambda. The partial artifact is intentionally retained at
  `runs/p1_v83_dev/v84a_gradient_calibration/progress.json`.
- The first calibration attempt completed all 24 windows but correctly
  stopped before reporting a lambda: zero initialization of the ACT output
  linear layer gives zero ACT-loss gradient into image features at step zero.
  The calibration was revised to use the actual common `act_head` group while
  explicitly reporting the initial zero feature-path gradient; the revised run
  is the one interrupted at window 16.
- Consequently, none of the following was run: v8.4-A 8-batch smoke, fresh
  v8.4-A 300B, 1e, 3e, final20, or medical evaluation.
- Corrected-300B attempts consumed by v8.4-A: `0 / 3`.

## Safe resume order

1. Re-run `tools/calibrate_p1_v84_gradients.py` to a completed 24-window
   summary; accept a lambda only if its no-step integrity and stability checks
   pass.
2. Re-run focused tests, `py_compile`, and `git diff --check`.
3. Run the specified fresh-init 8-batch P1-v8.4-A smoke using the calibrated
   ACT lambda.
4. Only if smoke passes, run one fresh 300B attempt and apply the A1--A5
   decision tree from the authoritative specification.

No final20 or medical evaluation was started. No P1-v8.4-B work was started.

# Parent and corrected-run engineering audit

| area | status | evidence | causal interpretation |
|---|---|---|---|
| optimizer family | CORRECT | PARENT_OPTIMIZATION_LEDGER.csv; Adam in both runs | matched, not shown suboptimal |
| betas/eps | CORRECT | betas (0.9,0.999), eps 1e-8 in checkpoint groups | matched |
| weight decay | CORRECT | zero in all three groups | matched; not an explanatory difference |
| parameter-group LRs | CORRECT | image 1e-3, text 5e-4, prompt 1e-4 base; ratios 1:0.5:0.1 | matched |
| StepLR | CORRECT | step size 1, gamma 0.9; state advances to candidate epoch | pre-fix CIR bug removed |
| StepLR timing | CORRECT | after epoch, before history and candidate save | matched |
| prompt policy | CORRECT | frozen E01–E03, trainable E04+; post group policy matches | matched |
| DFG schedule | CORRECT | warmup 0.10, target 0.10; ledger | matched |
| gradient clipping | STABLE_BUT_UNPROVEN_OPTIMAL | norm 1.0 per optimizer step in code | frequency was not logged |
| batch geometry | CORRECT | micro/effective batch 6, accumulation 1 | matched |
| seed/source/CLIP | CORRECT | seed 0, source identity SHA, CLIP SHA | matched |
| precision/AMP/TF32 | CORRECT | FP32, AMP false, TF32 false | matched |
| DataLoader | CORRECT | workers 4, pin/persistent/prefetch 2 | matched |
| checkpoint timing | CORRECT | E10/E12/E14/E16/E18/E20 after scheduler step | matched |
| resume semantics | CORRECT_TESTED | regression suite covers state restoration; no resume was needed | no observed defect |
| loss | CORRECT_BUT_K_INACTIVE | cls + seg + .001*kg + 0*k; bounded K gradient is zero | causal importance unknown |
| deployment operator | MATERIAL_RISK | training and deployed maps differ in bounded E14 audit | causal share unknown |
| evaluation resilience | CORRECT | exact spools, teardown, atomic cells, 108/108 complete | no OOM/killed incident |

NOT OPTIMAL is not treated as BUG. This audit identifies only the historical missing scheduler.step() as a confirmed protocol bug; the corrected pair matches the intended canonical schedule.

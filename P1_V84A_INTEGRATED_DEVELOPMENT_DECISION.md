# P1-v8.4-A integrated development decision

Source: `f3e5cd7755d7c7557612b9e218e3aa1c68af219f` on local branch
`autopilot/p1-v84a-integrated-dev`. The frozen configuration used true
residual P1-v8.4-A semantics; factor utility tau `0.05`, factor lambda
`0.03`; Router margin eligibility (`valid && best_gain > 0 && margin_rel >
0.10`) with `patch_zscore_softmax` target and lambda
`0.00044262806523447237`; routed ACT teacher `g_route`, threshold `0.0`,
and lambda `7.435420936678605e-05`; and fixed, non-trainable rho `0.05`.
No source or scientific setting changed between the fresh 1e and fresh
continuous 3e runs.

## Clean 1e gate

The fresh seed-0 VisA/train 1e run completed `2162/2162` batches and `361`
optimizer steps. Its structural gate passed (`hard_failure=false`, no soft
warnings); MAIN exact-change and surgery reconstruction maxima were both
`0.0`. The epoch-1 canonical VisA/test averages were pixel AUROC/AP
`94.9392/23.8500` and image AUROC/AP `88.6283/90.8167`.

## Fresh continuous 3e evidence

The fresh seed-0 3e run completed three full epochs, with all epoch gates
passing and no non-finite state or OOM. Recorded peak runtime telemetry was
allocated/reserved/peak `2.16/4.25/3.79 GiB`; rho remained `0.05`.

| Epoch | VisA pixel AUROC/AP | VisA image AUROC/AP | Router dense F1/F2/F3/F4 |
| --- | --- | --- | --- |
| 1 | 94.8842 / 27.0267 | 87.0333 / 89.7417 | .2858 / .2391 / .2409 / .2342 |
| 2 | 94.1608 / 29.9925 | 89.3883 / 91.3483 | .3158 / .2287 / .2285 / .2270 |
| 3 | 94.2658 / 33.4775 | 95.5600 / 96.2733 | .3438 / .2152 / .2168 / .2242 |

At epoch 3, however, the scheduled sparse Router path regressed: for all
three levels its shares were approximately `F1=.605`, `F2<=.000024`,
`F3<=.000258`, `F4=.394`; each level reported `sparse_dead=[2,2,2]` and
unique top-k pairs fell to `1.178`, `1.189`, and `1.056`. Thus useful target
factors F2 and F3 disappear in the active sparse path. This is a Router
collapse under the predeclared 3e gate, even though dense diagnostic usage
and the canonical evaluation remain finite.

Factor and ACT raw gradients remained finite (epoch-3 factor norms
`1.17e-05` to `1.99e-05`, ACT-head norm `1.25e-05`), but this does not
override the Router non-collapse requirement. No post-hoc threshold,
lambda, target, capacity, or routing change was made.

## Decision

`EXIT_FOR_DISCUSSION` — `P1_V84A_3E_ROUTER_REGRESSION`.

The development candidate is **not frozen** and the fresh 20e preflight and
20e launch are **not authorized**. Next discussion must diagnose the
epoch-3 scheduled sparse Router collapse from the existing compact 3e
evidence; it must not silently tune or rerun this scientific configuration.

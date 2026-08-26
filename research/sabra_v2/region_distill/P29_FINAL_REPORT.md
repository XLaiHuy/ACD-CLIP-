# P29 FINAL REPORT

## IDENTITY

- Parent P28R1 terminal SHA: `feb7d5dcf5a0e0b4933e1069288cb4ab30dc8101`
- P29 prereg SHA: `7d39df9c7201680ee9b9259481267e824ee0802b`
- P29 execution-base SHA: `24135127c246d024636ec752c656e9bb828f8cdf`
- P29 attempt UUID: `bcfd6874-7927-4be4-bb07-73c079a9ff76`
- Terminal SHA: pending terminal-evidence commit

## ENGINEERING

- Targeted tests: 38 passed (P29-focused plus reused P27/P28R1 cache, region-pool, and parity regressions).
- Cache parity: P27 Tier-A/Tier-B float32 provenance validated by the runner audit.
- Cached median sec/step: 0.010768339969217777 s (engineering profile).
- Peak GPU allocated: 115313152 bytes (engineering profile).
- Total scientific runtime: 17831.198 s.
- Engineering anomaly: required one-shot post-hoc mechanism audit failed before output because R0 utility action computation was called under `torch.no_grad()`; no rerun or patch is allowed.

## SCIENTIFIC RESULTS (FROZEN SCORES)

- Native macro pAP: 0.4525216034
- P29 macro pAP: 0.4559244442
- Delta pAP: +0.0034028408
- Native macro pAUROC: 0.9345650496
- P29 macro pAUROC: 0.9191979029
- Delta pAUROC: -0.0153671466
- Improving / non-regressing / regressing pAP classes: 5 / 5 / 7
- Median pAP delta: -0.0045215234
- Best pAP delta: fryum +0.0512168821
- Worst pAP delta: macaroni1 -0.0322847530
- Top-1 / top-2 positive gain concentration: 34.4991% / 63.4512%

The full per-class table is `P29_CLASS_TABLE.csv`.

## MECHANISM

- P27 sign agreement: 0.5228332297.
- P29 sign agreement: unavailable; the required one-shot mechanism audit failed before output.
- Normal-score mean/q99 shift: unavailable.
- Anomaly-score mean/median shift: unavailable.
- OR pAP/pAUROC recovery ratios: unavailable.

## AUDIT

- Held GT reads before prediction: 0
- Held mask reads before prediction: 0
- Held teacher reads before prediction: 0 (source-only cache/firewall contract).
- New CLIP forwards: 0
- New Phase2B forwards: 0
- Phase2B / CLIP optimization steps: 0 / 0
- MVTec / Medical reads: 0 / 0
- Runner post-audit: PASS; terminal audit: FAIL due solely to incomplete required mechanism evidence.

## OBSERVED

The independent 12-fold train/predict/score run completed exactly once with all predictions frozen before scoring and a passing runner audit. Its required lightweight mechanism audit then failed before generating mechanism evidence.

## INTERPRETATION

No P29 mechanism interpretation is made. The terminal status is engineering stop rather than a scientific conclusion because required preregistered mechanism evidence is incomplete.

## FINAL

- FINAL STATUS: `P29_ENGINEERING_STOP`
- NEXT ACTION: `PRESERVE_EVIDENCE_AND_STOP`

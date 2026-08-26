# P28 FINAL MECHANISM DIAGNOSTIC

## IDENTITY

1. P27 terminal SHA: `cdf06234bee861bbe81a7f07e382530f9a66c207`
2. P27 scientific execution-base: `de41b380449dcbc0b124f71f4f8fbb789e1a96f0`
3. P28 prereg SHA: `651ee5a784d2d037ff13d5891671a679fd8691e3`
4. P28 execution-base SHA: `0c6d4b27d06c33b9a691b72c354fff20e0df0e3a`
5. P28 attempt UUID: `47fa52ee-a5cd-45ce-aa11-47737f3462f7`

## AUDIT

6. Training steps: `0`
7. Parameter-update steps: `0`
8. New CLIP forwards: `0`
9. New Phase2B forwards: `0`
10. MVTec reads: `0`
11. Medical reads: `0`
12. Post-audit: `P28_ENGINEERING_STOP`

## FOUR-STATE RESULTS

13. N macro pAP / pAUROC: `N/A — diagnostic stopped`
14. OP macro pAP / pAUROC: `N/A — diagnostic stopped`
15. OR macro pAP / pAUROC: `N/A — diagnostic stopped`
16. S macro pAP / pAUROC: `N/A — diagnostic stopped`

## DECOMPOSITION

17. OP-N teacher effect: `N/A`
18. OR-OP regionization effect: `N/A`
19. S-OR student-transfer effect: `N/A`
20. S-N final effect: `N/A`

## BREADTH

21. Positive / non-regressing / negative classes: `N/A`
22. Median class effects: `N/A`
23. Best/worst classes: `N/A`
24. Gain concentration: `N/A`

## ENGINEERING FAILURE

The one and only diagnostic execution stopped on class `candle` at the
frozen-map parity gate. Native cache reconstruction matched its immutable
native map exactly (`max_abs=0.0`), but the adapter-derived student map
differed from the immutable P27 student map by
`0.00022339820861816406`, exceeding the fixed implementation audit tolerance
`0.00002`. The consumed attempt marker and this failure evidence are
preserved. No patch, bypass, or rerun was performed.

Held VisA masks were used only post-freeze as permitted: the stopped candle
class opened 100 held anomaly masks before the parity failure. P27’s own
pre-scoring held-GT and held-mask counts remain zero. No MVTec or Medical data
were read.

## RANKING MECHANISM

25–30. `N/A`: no complete ranking diagnostic exists.

## ALIGNMENT

31–35. `N/A`: no complete teacher/student alignment diagnostic exists.

## HYPOTHESES

36–40. `N/A`: no hypothesis classification is scientifically valid from an
incomplete decomposition.

## ROOT CAUSE

41. Primary mechanism: `INSUFFICIENT`
42. Secondary mechanism: `ENGINEERING_STOP`
43. OBSERVED: the single post-marker execution stopped at candle student-map
parity; no four-state scientific result was produced.
44. INTERPRETATION: the P28 mechanism question is unresolved. The parity
failure must not be converted into a teacher, regionization, or transfer
claim.

## NEXT-STEP DECISION

45. Recommendation for P29: `STOP SABRA-V2 REGION LINEAGE`

## FINAL STATUS

`P28_ENGINEERING_STOP`

P28 does not promote a deployable architecture. P26 remains the deployable
architecture. No P29 implementation was started.

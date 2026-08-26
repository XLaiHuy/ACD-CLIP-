# P28R1 FINAL MECHANISM DIAGNOSTIC

## IDENTITY

1. P27 terminal SHA: `cdf06234bee861bbe81a7f07e382530f9a66c207`
2. P27 scientific execution-base: `de41b380449dcbc0b124f71f4f8fbb789e1a96f0`
3. P28R1 prereg SHA: `1e1e4fb4ae96342d04ce4206bbdc4091103b28e7`
4. P28R1 execution-base SHA: `3ed8e9f5624fa858f2baf26f29970ec22d2e1eac`
5. P28R1 attempt UUID: `1c41fbc7-1d70-42a4-bbda-ca512c82f73f`

## AUDIT

6. Training steps: `0`
7. Optimizer steps: `0`
8. New CLIP forwards: `0`
9. New Phase2B forwards: `0`
10. MVTec reads: `0`
11. Medical reads: `0`
12. Held GT reads before scoring: `0`
13. Held mask reads before scoring: `0`
14. Post-hoc held mask reads: `1200`
15. Post-audit: `PASS`

## FOUR-STATE RESULTS

1
6
.

N

m
a
c
r
o

p
A
P

/

p
A
U
R
O
C
:

`
0
.
4
5
2
5
2
1
6
0
3
4
`

/

`
0
.
9
3
4
5
6
5
0
4
9
6
`
1
7
.

O
P

m
a
c
r
o

p
A
P

/

p
A
U
R
O
C
:

`
0
.
5
1
6
2
4
2
0
8
8
8
`

/

`
0
.
9
9
0
7
4
7
2
5
3
0
`
1
8
.

O
R

m
a
c
r
o

p
A
P

/

p
A
U
R
O
C
:

`
0
.
5
2
5
5
9
4
8
8
8
1
`

/

`
0
.
9
8
5
5
5
5
0
1
3
4
`
1
9
.

S

m
a
c
r
o

p
A
P

/

p
A
U
R
O
C
:

`
0
.
4
6
1
3
8
7
5
6
6
3
`

/

`
0
.
9
2
0
3
4
1
1
9
3
9
`

## DECOMPOSITION

2
0
.

O
P
-
N

t
e
a
c
h
e
r

e
f
f
e
c
t
:

p
A
P

`
+
0
.
0
6
3
7
2
0
4
8
5
4
`
,

p
A
U
R
O
C

`
+
0
.
0
5
6
1
8
2
2
0
3
5
`
2
1
.

O
R
-
O
P

r
e
g
i
o
n
i
z
a
t
i
o
n

e
f
f
e
c
t
:

p
A
P

`
+
0
.
0
0
9
3
5
2
7
9
9
3
`
,

p
A
U
R
O
C

`
-
0
.
0
0
5
1
9
2
2
3
9
7
`
2
2
.

S
-
O
R

s
t
u
d
e
n
t
-
t
r
a
n
s
f
e
r

e
f
f
e
c
t
:

p
A
P

`
-
0
.
0
6
4
2
0
7
3
2
1
8
`
,

p
A
U
R
O
C

`
-
0
.
0
6
5
2
1
3
8
1
9
5
`
2
3
.

S
-
N

f
i
n
a
l

P
2
7

e
f
f
e
c
t
:

p
A
P

`
+
0
.
0
0
8
8
6
5
9
6
2
9
`
,

p
A
U
R
O
C

`
-
0
.
0
1
4
2
2
3
8
5
5
7
`

## BREADTH

2
4
.

S
-
N

p
A
P
:

i
m
p
r
o
v
i
n
g

`
8
/
1
2
`
,

n
o
n
-
r
e
g
r
e
s
s
i
n
g

`
8
/
1
2
`
,

r
e
g
r
e
s
s
i
n
g

`
4
/
1
2
`
;

m
e
d
i
a
n

`
+
0
.
0
0
1
1
4
6
9
1
5
7
`
;

b
e
s
t

`
c
a
s
h
e
w

+
0
.
0
7
6
4
8
2
8
8
4
1
`
;

w
o
r
s
t

`
m
a
c
a
r
o
n
i
1

-
0
.
0
4
4
1
6
8
9
6
3
2
`
.
25. S-N pAUROC: improving `1/12`, non-regressing `1/12`, regressing `11/12`; median `-0.0073768832`; best `pipe_fryum +0.0213021902`; worst `macaroni2 -0.0575124601`.
26. S-N top-1/top-2 positive pAP gain concentration: `39.738536%` / `69.079093%`.

| class | native pAP | P27 pAP | delta pAP | native pAUROC | P27 pAUROC | delta pAUROC |
|---|---:|---:|---:|---:|---:|---:|
| candle | 0.5141403049 | 0.5142973531 | +0.0001570481 | 0.9806671435 | 0.9711054390 | -0.0095617045 |
| capsules | 0.6978763720 | 0.6892354886 | -0.0086408834 | 0.9849781994 | 0.9795885461 | -0.0053896533 |
| cashew | 0.7655127373 | 0.8419956214 | +0.0764828841 | 0.9930016112 | 0.9890906369 | -0.0039109743 |
| chewinggum | 0.8701427462 | 0.8716845862 | +0.0015418400 | 0.9919411142 | 0.9886921440 | -0.0032489703 |
| fryum | 0.3872210698 | 0.4436914546 | +0.0564703848 | 0.9550680179 | 0.9457039049 | -0.0093641130 |
| macaroni1 | 0.2114248870 | 0.1672559237 | -0.0441689632 | 0.9495230825 | 0.8955779194 | -0.0539451631 |
| macaroni2 | 0.0894527937 | 0.0703068176 | -0.0191459761 | 0.8960057743 | 0.8384933143 | -0.0575124601 |
| pcb1 | 0.2039319994 | 0.2199487377 | +0.0160167383 | 0.7888956749 | 0.7745854966 | -0.0143101783 |
| pcb2 | 0.2150817035 | 0.2201766882 | +0.0050949847 | 0.8373864178 | 0.8083002013 | -0.0290862164 |
| pcb3 | 0.4848358825 | 0.4855878738 | +0.0007519914 | 0.8999823087 | 0.8965311533 | -0.0034511554 |
| pcb4 | 0.4523718587 | 0.4382539584 | -0.0141179003 | 0.9820203967 | 0.9798125270 | -0.0022078697 |
| pipe_fryum | 0.5382668860 | 0.5742162927 | +0.0359494067 | 0.9553108536 | 0.9766130438 | +0.0213021902 |

## RANKING MECHANISM

27. Aggregate S pair-ordering change: gained `1361725068537`, lost `1984224948617`, net `-622499880080`.
28. Aggregate OP pair-ordering change: gained `6569144970633`, lost `10196991709`, net `6558947978924`.
29. Aggregate OR pair-ordering change: gained `6369708436440`, lost `23617269162`, net `6346091167278`.
30. S anomaly shift mean across classes `0.1850043045`; S normal shift mean `0.0008240923`; S normal q99 shift mean `0.0007015706`.
31. Fixed top-rank fractions and all per-class shift, top-rank, and pair-ordering evidence are in `P28R1_RANKING_DIAGNOSTIC.json`.

## ALIGNMENT

32. Macro teacher/student alignment: Pearson `0.7492951999`, Spearman `0.6949611922`, sign agreement `0.5228332297`, MAE `0.9880948008`, robust magnitude ratio `435012175391.8274536133`.
33. No calibration, fitting, threshold, or rescaling was performed.

## HYPOTHESES

34. H1 teacher objective conflict: `NOT_SUPPORTED`.
35. H2 regionization loss: `PLAUSIBLE`.
36. H3 student transfer failure: `SUPPORTED`.
37. H4 normal-score inflation: `SUPPORTED`.
38. H5 heterogeneous category actionability: `SUPPORTED`.

## ROOT CAUSE

39. Primary mechanism: `STUDENT_TRANSFER_FAILURE`.
40. Secondary mechanism: `HETEROGENEOUS_ACTIONABILITY`.

### OBSERVED

OP improves both pAP and pAUROC across all 12 classes, so the teacher semantics do not create the observed AP/AUROC conflict. OR retains large oracle headroom but introduces a smaller AUROC loss in regionization. S loses nearly all OR oracle benefit across all 12 classes; the final S-N result retains a small macro pAP gain but loses macro pAUROC. The S ranking counts show more lost than gained anomaly>normal orderings, and category effects are heterogeneous.

### INTERPRETATION

The decomposition supports student transfer failure as the primary mechanism, with heterogeneous actionability secondary and regionization ranking loss plausible but not dominant. Normal-score inflation is a supported contributor. These are post-hoc diagnostic/oracle findings only; P26 remains deployable.

## NEXT-STEP DECISION

41. Recommendation for P29: `ROBUST_STUDENT_TRANSFER`.

## ENGINEERING

42. Scientific wall time: `8890.084348` seconds (2.4695 hours).
43. Cache mode: `Tier-A memmap, one class at a time`; replay batch size `1`.
44. Parity qualification runtime: `126.76698591094464` seconds; all 12 native and all 12 student max absolute errors were `0.0`.
45. Terminal commit SHA: recorded by the final Git terminal commit containing this report.

## FINAL STATUS

`P28R1_DIAGNOSTIC_COMPLETE`

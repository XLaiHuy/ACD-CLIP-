# P14 Nested LOCO Contract

For outer held class H, all direction/harm predictions on each outer-train
class J exclude J exactly as R2-v2.  `tau20_J` and `tau40_J` use only Level-2
OOF risk rows from `outer_train \ {J}`.  Source V targets for J use those
excluded predictions and J labels only.  The value OOF prediction for J trains
on image targets/features from `outer_train \ {J}`.

For the held H, final direction/harm models use the 11 outer-train classes;
final thresholds use outer-train leakage-safe Level-2 OOF risk only; the final
value head trains on all outer-train image labels.  H labels never enter models,
thresholds, features, selection, or action decisions.

Value-score selection evaluates only q=`{.50,.60,.70,.80,.90}` from
`value_oof`, plus `NO_EXPANSION` (+infinity).  A q is eligible only if source
OOF accepted wrong-sign rate is <=5%, weighted-harm reduction vs direction is
>=50%, and source OOF macro pAP is strictly above SAFE20.  Choose the greatest
pAP; ties within 1e-12 choose higher q.  If none is eligible, use NO_EXPANSION.

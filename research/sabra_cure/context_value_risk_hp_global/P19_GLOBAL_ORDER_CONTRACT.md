# P19 Global Value Order Contract

Frozen P14 constructs `v` and `vh` with:

```python
np.concatenate([folds[name]['target']['v'] for name in r1.CLASSES])
np.concatenate([folds[name]['vhat'] for name in r1.CLASSES])
```

The exact held order is `candle, capsules, cashew, chewinggum, fryum,
macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum`. Within a held class
the order is the immutable `image_path` order recorded by its science worker.
`value_pairs.npz` stores exactly `image_index` (`int64`), `vhat` (`float64`),
and `V_j` (`float64`) in that order.

The global worker concatenates in the listed order, filters finite paired
values exactly as `p12.correlation`, computes Pearson in float64, then uses
stable `np.argsort(np.argsort(x, kind='stable'), kind='stable')` independently
for both inputs before float64 rank correlation. Sign accuracy uses exactly
`abs(V_j)>P14.EPS`. No parent loads these arrays.

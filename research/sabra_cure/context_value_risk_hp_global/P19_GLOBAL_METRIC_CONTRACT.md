# P19 Global Metric Contract

The global metric worker is the sole exception to compact scalar aggregation.
It may load only 12 `value_pairs.npz` artifacts, verify expected fields/order,
and write JSON containing finite count, global Pearson, global stable-rank
Spearman, global sign accuracy, eligible sign count, fold pair hashes, and an
order digest. It must not load score maps, masks, actions, source shards, fold
NPZ artifacts, or features.

The independent global audit repeats the same restricted load and comparison.
Both must pass before parent JSON-only aggregation applies the original P14
gates. The mandatory stable-sort counterexamples are `+0.8` and `-0.8`.

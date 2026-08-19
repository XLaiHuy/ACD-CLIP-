# Trust-v2 reserve-index forensic classification

Classification: REPORTING_ONLY_BUG

Forensic checkpoint:

- synchronized HEAD and remote: 7da64645405345f133cfed3698046bcf9a3875c0
- divergence: 0 0
- VisA Trust OOF stage completed before the exception
- no Trust-v2 result artifact or OOF checkpoint was written
- MVTec reads: 0
- medical reads: 0
- finalized GT-free cache and shard hashes were unchanged

## Data-flow finding

construct_b1_v2 creates reserves with shape (2, 1369) and writes
reserves[0, query] as the exact p9 identity and reserves[1, query] as the
exact p16 identity. Cache serialization preserves these arrays as
(image_count, 1369) fields. The finalized shards were checked directly:
all twelve reserve_p9_index and reserve_p16_index arrays are
image-by-query-patch arrays, with dtype int64; the corresponding peer arrays
are (image_count, 1369, 8).

Before reporting:

- compact geometry uses features[:, reserves], producing patch-specific
  query/reserve and reserve/peer geometry;
- cache p16 parity uses reserve[patch] for each tested query patch;
- PGM/PCRR reserve replacements are derived from that patch-specific geometry;
- reserve PGM/PCRR ranks have shape (reserve, replacement_slot, query_patch);
- S9, R9, S16, and R16 consume those rank tensors, not image-level IDs;
- the Trust OOF matrices consume only E, credibility features, and
  S9/R9/S16/R16; raw reserve IDs are not OOF inputs;
- the row inclusion mask uses valid_b1, baseline E, selected Trust score, and
  VisA GT. It does not use p9/p16 IDs.

A synthetic direct-index parity check gave exact zero error for both compact
query/reserve and reserve/peer geometry. The cache construction and parity
checks therefore confirm reserve_index[query_patch] semantics before
reporting.

## Exact failure and correction

The failure is at
tools/sabra/trust_v2/visa_audit.py:238-239:

    p9 = max(int(p9_rows[local_index]), 0)
    p16 = max(int(p16_rows[local_index]), 0)

p9_rows[local_index] and p16_rows[local_index] are length-1369 query-patch
arrays, so conversion to an image-level scalar raises
TypeError: only length-1 arrays can be converted to Python scalars.

The minimal correction is to perform the existing scalar conversion at the
already-selected row patch:

    p9 = max(int(p9_rows[local_index][patch]), 0)
    p16 = max(int(p16_rows[local_index][patch]), 0)

This changes only the p9_index and p16_index metadata values in
STABLE_BUT_WRONG_V2.csv and allows the existing row serializer to complete.
It does not change cache values, PGM/PCRR, geometry, S/R features, OOF inputs,
OOF predictions, class folds, model coefficients, model selection, thresholds,
gates, or the row inclusion criterion.

No OOF arrays were persisted on disk; the completed in-memory OOF outputs were
lost when the process terminated at stable-wrong row construction. They must
not be represented as checkpointed artifacts or silently reconstructed by this
repair.

## Follow-up Authority reporting defect

During the resumed VisA audit from c23a137, Trust OOF, PCRR, C1, and
stable-wrong reporting completed. Authority score arrays were then formed, but
tools/sabra/trust_v2/visa_audit.py:194 raised while aggregating the
occupancy Spearman summary called from line 324. Authority intentionally
passes a zero occupancy placeholder; every _spearman value is therefore
None, and np.nanmean on the raw None list raises TypeError.

The correction maps None to an undefined metric (JSON null) when all values are
None, while preserving the existing numeric nanmean path when values exist.
Authority scores, AUROC/AP/Brier, A2-vs-A1 effect, status, safety, and all
scientific inputs are unaffected. The partial artifact checkpoint before this
guard was:
TRUST_V2_MODEL_AUDIT.json cd2a0a1085a0428c3392b8b1c52e56f5fcc86b482ead6c84bc972e9b4b0b75b8
PCRR_DISAGREEMENT_AUDIT.json 36d4796f4d26fc87ede469823057e6f165c9e58a3285aa4cc068ed515b097743
NEED_C1_FROZEN_MODEL.json b83eeed2104926bbdae61ebefbeb0f21d049ab48bf53a8d304bc9fbd54210b53
AUTHORITY_V2_AUDIT.json d54f547b2c70cc15e444e3c581802f606b8bfc0db1a756adedabb134f2b08d6f

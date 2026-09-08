# MVTec-source manifest audit

Audit completed before Phase 2 training on branch `research/mvtec-source-anchor-ref-e8-v1`.

- Parent branch: `research/anchor_ref_e8_r1`
- Parent SHA: `b946a3effffaed4a7d25826e96494d5d91edab8d`
- New branch initial SHA: `b946a3effffaed4a7d25826e96494d5d91edab8d`
- Manifest: `dataset/hub/MVTec.jsonl`
- Manifest SHA256: `3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c`
- Working manifest matches `HEAD`: yes

The original manifest is used directly. No alias, rewrite, split, subsampling, or source-manifest diff is present. The source root is the existing `data/mvtec_ad` link to `/workspace/datasets/mvtec`.

## Pool and integrity checks

| Check | Result |
|---|---:|
| Total records | 1725 |
| Normal records (`label=0`) | 467 |
| Anomaly records (`label=1`) | 1258 |
| Categories | 15 |
| Missing images | 0 |
| Missing anomaly masks | 0 |
| Anomaly records without masks | 0 |
| Normal records with unexpected masks | 0 |
| Duplicate image paths | 0 |
| Duplicate anomaly mask paths | 0 |
| Train/good records | 0 |

All normal records resolve to `<category>/test/good/*`; all anomaly records resolve to `<category>/test/<defect_type>/*`; anomaly masks resolve to `<category>/ground_truth/<defect_type>/*_mask.png`. Label values are exactly `{0, 1}`.

## Category audit

| Category | Total | Normal | Anomaly | Defect types |
|---|---:|---:|---:|---|
| bottle | 83 | 20 | 63 | broken_large, broken_small, contamination |
| cable | 150 | 58 | 92 | bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation |
| capsule | 132 | 23 | 109 | crack, faulty_imprint, poke, scratch, squeeze |
| carpet | 117 | 28 | 89 | color, cut, hole, metal_contamination, thread |
| grid | 78 | 21 | 57 | bent, broken, glue, metal_contamination, thread |
| hazelnut | 110 | 40 | 70 | crack, cut, hole, print |
| leather | 124 | 32 | 92 | color, cut, fold, glue, poke |
| metal_nut | 115 | 22 | 93 | bent, color, flip, scratch |
| pill | 167 | 26 | 141 | color, combined, contamination, crack, faulty_imprint, pill_type, scratch |
| screw | 160 | 41 | 119 | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top |
| tile | 117 | 33 | 84 | crack, glue_strip, gray_stroke, oil, rough |
| toothbrush | 42 | 12 | 30 | defective |
| transistor | 100 | 60 | 40 | bent_lead, cut_lead, damaged_case, misplaced |
| wood | 79 | 19 | 60 | color, combined, hole, liquid, scratch |
| zipper | 151 | 32 | 119 | broken_teeth, combined, fabric_border, fabric_interior, rough, split_teeth, squeezed_teeth |

This audit passed. Phase 2 may use MVTec as the training source and must evaluate the resulting checkpoints on VisA; it must not use the standard MVTec train/good split or the custom 962-normal/1200-anomaly split.

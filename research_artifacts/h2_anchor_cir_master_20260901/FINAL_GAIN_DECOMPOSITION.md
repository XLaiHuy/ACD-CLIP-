# Final gain decomposition

All Medical values below are macro means over the six frozen targets. Pixel metrics are defined for all six targets. Image metrics are defined for three targets; the supported-target count is recorded in the CSV. Values are in `[0, 1]`.

The historical H2 replay is a reference, not a new arm. Its current exact-evaluator macro is `0.9092218791 / 0.4037306455` Pixel AUROC/AP. The new R E10 arm is the matched H2 training control and measures the observed recovery gap from that historical checkpoint, not an isolated Anchor or CIR effect.

The source-only same-E10 evidence is clear for the Anchor on Pixel AUROC/AP (`+0.03304292 / +0.08580855`) and mixed for CIR: RCA is lower in Pixel AUROC (`-0.04585601`) but slightly higher in Pixel AP (`+0.00252156`) than RA. Thus RCA does not provide a robust source-side matched dominance.

The Medical comparison uses the frozen source-selected candidates R E10, RA E16, and RCA E12. Because the selected epochs differ, `RA - R` and `RCA - RA` in the Medical rows are selected-checkpoint comparisons with an epoch-selection confound; they are not clean same-epoch causal estimates. On Medical, Anchor changes Pixel AUROC/AP by `-0.01539608 / +0.01719941`, while CIR changes them by `-0.01049996 / +0.00192286`. Supported image AUROC rises, but that does not overturn the primary pixel-metric mixture.

The final candidate therefore remains the preregistered source-selected **RA E16**, with native inference and no inference-time RMT. No claim is made that the Medical result proves or disproves the CIR mechanism independently of the selected-epoch limitation.

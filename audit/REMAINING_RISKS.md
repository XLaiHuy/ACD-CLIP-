# Remaining unresolved risks

1. The historical 96-image VisA gate overlaps the categories used to train the
   model and is not a true unseen-category generalization test. It is not
   authorized for target selection in the clean config.
2. Historical H2 E10 is a model-only checkpoint. Exact continuation from that
   checkpoint is impossible without its optimizer/RNG/DataLoader state; clean
   arms therefore share a new seeded full-state E1.
3. Exact benchmark metrics use stride 1 and raw values, while the published
   historical oracle uses stride 4 and per-class four-decimal rounding. The
   two numbers are both preserved and must not be compared as if they were one
   protocol.
4. The environment has torch 2.12.1+cu130/CUDA 13.0 while the old requirements
   file names older torch/CUDA pins. The actual runtime environment is recorded
   in clean checkpoints; cross-environment bitwise identity is not claimed.
5. Medical colon datasets contain only positive image labels, so image-level
   AUC/AP are structurally zero under the evaluator guard.
6. Full training and final medical evaluation are intentionally not launched
   by the guarded scripts in this audit. The smoke is wiring-only; no
   performance winner is decided.

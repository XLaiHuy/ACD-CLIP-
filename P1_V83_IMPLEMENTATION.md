# P1-v8.3 pre-real-data-smoke checkpoint

## Implemented

- Runtime portability and checkpoint/config parity contracts
- Structured, factor-specific `STATE` and deterministic `CLASS` text path
- Shared Text-LoRA dynamic prompt adaptation
- Four-factor utility specialization and dense utility-supervised routing
- Base/BestSingle/Oracle/Uniform/Routed diagnostics
- Offline Med-VISA setup tooling and no-data regression tests

## Verified

- No-data runtime/unit/structured-utility/setup suite: 20 passed, 1 warning in 1.42s
- Python compilation gate for the setup, model, dataset, training, and evaluation modules
- Git whitespace validation (`git diff --check`)

## Pending

- Completed Med-VISA recovery/setup and manifest integrity verification
- Model/data preflight
- One-batch forward and backward checks
- Eight-batch GPU smoke
- Optional 32--64 batch diagnostics
- Real `G_local`, `G_multi`, and router capture evidence

## Not run

- Final 20-epoch VisA training
- Full exact six-medical final evaluation

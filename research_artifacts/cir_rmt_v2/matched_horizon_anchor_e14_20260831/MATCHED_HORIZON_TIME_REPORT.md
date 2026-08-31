# Matched-horizon timing report

Status: PASS for the E14 source-only speed path.

The run trains only through E14, retains candidates E10/E12/E14, and performs no target evaluation.

| epoch | seconds | images/sec | peak allocated GiB | peak reserved GiB |
|---:|---:|---:|---:|---:|
| E10 | 950.987 | 2.2776 | 9.708 | 11.008 |
| E12 | 950.415 | 2.2790 | 9.708 | 11.008 |
| E14 | 952.647 | 2.2737 | 9.708 | 11.008 |

- Training wall time: 13278.694 seconds.
- Anchor-loss baseline median: 2.454597 seconds per measured batch.
- Anchor-loss median: 2.463086 seconds per measured batch.
- Anchor overhead: 0.008490 seconds (0.345875%).
- The anchor reference was loaded once and kept resident; no vectorization or denominator approximation was introduced.

| eval epoch | inference seconds | total evaluation seconds |
|---:|---:|---:|
| E10 | 16.392 | 35.691 |
| E12 | 15.878 | 34.856 |
| E14 | 15.701 | 34.939 |

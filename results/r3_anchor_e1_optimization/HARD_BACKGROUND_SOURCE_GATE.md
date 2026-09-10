# Hard-background patch ranking source gate

Status: `HARD_BACKGROUND_SOURCE_GATE_FAIL`

This was the single permitted novelty mechanism after the locked E15 target
audit remained below the published-context target. It changed only the
training loss: the highest-scoring background pixels were ranked below hard
foreground pixels, with direct suppression on normal images. The inference
model and evaluator were unchanged, so there is no inference-time overhead.

The source selector used only the preregistered VisA held-out evaluation slice
(`candle`, `macaroni1`, `pcb3`, `pipe_fryum`), locked TTA-F, and source Pixel
AP. The training split contains all VisA categories, so this remains a source
category diagnostic rather than a fully independent held-out-training
estimate; that limitation is retained from the earlier source gates.

| candidate | source Pixel AP | source Pixel AUROC | source Image AP | source Image AUROC | non-finite loss skips | non-finite gradient skips | finite |
|---|---:|---:|---:|---:|---:|---:|---:|
| locked baseline | 24.6335168 | 95.6550694 | 90.6629398 | 88.0986422 | N/A | N/A | PASS |
| lambda 0.02 | 23.8179621 | 94.7615324 | 91.7987630 | 89.9112880 | 0 | 1 | PASS |
| lambda 0.05 | 20.6178665 | 95.2882783 | 90.0257513 | 87.2014865 | 0 | 0 | PASS |

The best candidate was lambda `0.02`, but it was `0.8155547` percentage
points below the locked source Pixel AP baseline. Lambda `0.05` was
`4.0156502` points below baseline. Therefore the hard-background source gate
failed clearly.

Neither candidate was target-evaluated. The candidate checkpoints, exact raw
source JSON hashes, checkpoint configuration hashes, training stability, and
the source-only decision are recorded in
[`HARD_BACKGROUND_SOURCE_GATE.json`](./HARD_BACKGROUND_SOURCE_GATE.json).

Per the strict stop policy, this idea is recorded as negative and no further
loss, lambda, prompt, or ranking search is launched.

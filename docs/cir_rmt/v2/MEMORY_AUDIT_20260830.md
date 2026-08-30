# CIR_DFG_RMT_V2 evaluator memory audit

Date: 2026-08-30

Checkout: `research/cir-dfg-rmt-v2-signfix` at `1e13517ff36788c50d093bda61c29fc33bbdb2d5`

This audit was written before the bounded-memory implementation. It covers
`scripts/cir_rmt/eval_full.py`, `evaluation/evaluator.py`, and
`evaluation/metrics.py` as they existed at the checkout above.

## Scale and direct measurements

The planning arithmetic used 620 as the image count, but the canonical Brain
manifest contains 3,715 images. With the unchanged batch size of 6, this is
620 DataLoader batches, which is why the real progress bar ends at `620/620`.
The planning estimate and the observed canonical-manifest count are therefore:

`N_plan = 620 * 518 * 518 = 166,360,880` pixels.

`N_actual = 3,715 * 518 * 518 = 996,823,660` pixels.

The byte-accounting table below retains `N_plan` so the original failure
estimate is directly auditable. The completed real runs processed `N_actual`;
their score spool was 3,987,294,640 bytes, confirming the larger count.

A read-only synthetic probe of the current evaluator used 4,000,000 pixels
and reported the following current RSS values. The probe intentionally omitted
the model and DataLoader so that the metric-only amplification was isolated.

| phase | RSS MiB | max RSS MiB |
| --- | ---: | ---: |
| records retained after inference | 77.3 | 76.3 |
| records regrouped | 77.3 | 76.3 |
| two `np.concatenate` outputs | 107.8 | 106.8 |
| `_arrays` float64 conversion | 146.8 | 187.5 |
| AUROC order and sorted copies | 207.8 | 206.6 |
| AUROC ranks allocated | 238.3 | 237.1 |
| after AUROC temporaries released | 139.3 | 237.1 |
| AP order and sorted copies | 219.3 | 237.1 |
| AP boundary scan probe | 239.9 | 238.8 |
| after AP temporaries released | 143.2 | 238.8 |

The synthetic RSS values are not used as a Brain peak estimate; they are
evidence that the native NumPy allocations are visible in RSS and that the
working-set stages are additive.

An isolated CPU model probe reported 493.3 MiB at process start, 656.2 MiB
after checkpoint load, 3,860.6 MiB after the frozen model and CLIP asset were
constructed, and 2,173.7 MiB after model teardown. This is a host-side
reference, not a claim about the CUDA model footprint; it demonstrates that
the current evaluator also enters metric computation while model/checkpoint
objects remain live.

## Brain-scale byte accounting

All values below use `N = 166,360,880` and binary GiB.

| allocation | bytes | GiB |
| --- | ---: | ---: |
| raw pixel scores retained by records, float32 | 665,443,520 | 0.619743 |
| raw masks retained by records, current float32 | 665,443,520 | 0.619743 |
| compact masks in the new spool, uint8 | 166,360,880 | 0.154936 |
| `np.concatenate` score output, float32 | 665,443,520 | 0.619743 |
| `np.concatenate` mask output, float32 | 665,443,520 | 0.619743 |
| `_arrays` score conversion, float64 | 1,330,887,040 | 1.239485 |
| `_arrays` label conversion, int8 | 166,360,880 | 0.154936 |
| one int64 ordering | 1,330,887,040 | 1.239485 |
| sorted float64 score copy | 1,330,887,040 | 1.239485 |
| sorted int8 label copy | 166,360,880 | 0.154936 |
| AUROC float64 ranks | 1,330,887,040 | 1.239485 |
| positive-selector bool array | 166,360,880 | 0.154936 |
| AP unary-negated float64 scores | 1,330,887,040 | 1.239485 |

The 620-record Python list, dictionaries, array-view objects, image-level
lists, and class grouping lists are small relative to the pixel buffers
(well below 0.01 GiB), but they keep every raw score and mask allocation alive.
Four workers with the current batch size 6 and prefetch factor 2 can retain
roughly `4 * 2 * 6 * 4 * 518 * 518 * 4` bytes of image/mask tensors, about
0.48 GiB, before worker interpreter and dataset overhead. The checkpoint file
is 169,377,707 bytes (0.158 GiB); the live frozen model/CLIP host footprint is
larger, as shown by the isolated probe.

With records and concatenation retained, the old AUROC path can hold roughly

`0.620 + 0.620 + 0.620 + 0.620 + 1.239 + 0.155 + 1.239 + 1.239 + 0.155 + 1.239 + 0.155`

GiB of pixel-related storage before allocator/sort workspaces and
model/worker memory. This is about 7.90 GiB before the data-dependent
positive-rank selection copy and hidden NumPy sort workspace. The AP path
then independently adds a float64 conversion, unary-negated score array,
second order, sorted score copy, and sorted label copy; its corresponding
pixel-related live estimate is about 7.75 GiB. These are phase estimates, not
an assumption that AUROC and AP temporaries coexist indefinitely.

## Root-cause conclusion

The two Brain kills are an end-to-end host-memory amplification failure, not
a GPU-capacity failure and not a Python list of AP tie groups alone. During
inference, `eval_full.py` appends one full-resolution score view and one full
resolution float32 mask view to `records` for every image. `evaluate_records`
then retains those records while regrouping them and allocates fresh whole
class arrays with `np.concatenate`. `metrics._arrays` copies the score vector
to float64 even though the model produced float32 values. AUROC allocates a
full int64 order, sorted score/label copies, a full float64 ranks array, and
selector temporaries. AP calls the validation/conversion path again and
creates an independent descending ordering and sorted copies. The loader,
model, checkpoint, last forward tensors, and worker processes are still alive
through all of those metric peaks.

Removing only the Python list of AP group tuples therefore leaves the dominant
buffers and both independent orderings untouched. The required fix is a
bounded dataflow: write float32 scores and uint8 labels to a deterministic
temporary per-class spool, retain only image-level values in RAM, explicitly
tear down inference objects, then process one class at a time with one shared
exact ordering for AUROC and AP. The exact metric arithmetic remains float64,
and tie groups are accumulated across bounded chunks rather than represented
as per-pixel Python objects.

No real Brain evaluation was restarted during this audit.

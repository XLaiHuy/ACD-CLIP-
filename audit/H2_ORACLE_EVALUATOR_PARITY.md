# H2 evaluator parity

All values in the CSV are raw `[0,1]` metrics unless the row is explicitly
marked as historical display output. The historical H2 parser reproduces the
published E10 class rows using pixel stride 4 and per-class rounding to four
decimal places. Its pixel macro computed from those rounded class rows is
`0.9097500 / 0.4034833` (`90.9750 / 40.3483` percent); the historical parser
display row rounds this to `90.98 / 40.35`.

The current exact reference is the six-class full-resolution replay already
recorded in the H2 master artifact. Its pixel macro is
`0.9092218791 / 0.4037306455` (`90.9222 / 40.3731` percent). Relative to the
legacy displayed macro, the full-resolution/raw change is
`-0.0528 / +0.0247` percentage points. This difference is an evaluator
protocol difference, not a model-selection result.

The candidate checkout was additionally run in `benchmark_exact` mode on the
Brain class with pixel stride 1 and the new disk-backed spool. It completed
without the earlier RAM kill and produced `95.202911 / 38.280072` pixel and
`79.827940 / 93.548626` image percent. The independent exact reference is
`95.202928 / 38.280134` and `79.827439 / 93.548354`; the bounded candidate
replay agrees at the reporting precision needed for the audit. The six-class
reference table remains the authoritative full-resolution oracle; no current
candidate arm is selected from it.

The exact benchmark path is guarded against threshold binning and stride
other than 1. It writes float32 scores, uint8 labels, and packed sort keys to
a private disk spool, then removes the spool after the metric is computed.
The legacy evaluator remains the default and preserves its stride and
rounding behavior.

**Status:** evaluator implementation and bounded replay `PASS`; no claim is
made that a full six-class benchmark was rerun in this candidate checkout.

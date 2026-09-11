# CIR test status for the disabled fresh experiment

`tests/test_cir_v2_parity.py::test_cir_alpha05_reference_output_parity_synthetic`
fails under the current Python/Torch runtime by one strict FP32 assertion:
the maximum optimized-versus-reference difference is `3.814697e-6` while the
test tolerance is `3e-6`. The failure is in the CIR score-space parity test;
the original test and tolerance remain unchanged.

This is CIR-only. The model branch is entered only when
`not test_mode and cir_training and cir_alpha != 0`. The fresh v2 launchers
do not pass `--use_cir_training` or `--cir_alpha`, so training uses the normal
DFG/HPA path with `cir_training=False`. The evaluator calls the same fusion
function with `test_mode=True`, which also bypasses CIR. The failure does not
affect the fresh pipeline.

Action: keep CIR disabled, do not tune or enable it, record this isolated test
failure, and continue with the non-CIR preflight gates. The full pytest result
may therefore contain this one known CIR-only failure; the relevant tests are
run with `tests/test_cir_v2_parity.py` excluded.

# Resume parity audit

The deterministic toy continuation test in
`tests/test_resume.py::test_full_state_resume_matches_uninterrupted` compares
an uninterrupted six-step run with a three-step checkpoint followed by a
three-step resume. It asserts exact equality for model parameters, optimizer
state, scheduler state, scaler state, restored RNG streams, and the next
DataLoader batch. The test passes.

The clean checkpoint records `checkpoint_version=2`, model adapter and soft
prompt state, optimizer, scheduler, scaler, Python/NumPy/CPU/CUDA RNG state,
DataLoader generator state, epoch, global step, resolved config and hash,
repository/CLIP/dataset hashes, precision/AMP/TF32 metadata, and a
parameter-only `image_parameter_reference` plus optional anchor metadata.
The parameter-only field excludes BatchNorm buffers from anchor identity.

Resume rejects the historical model-only H2 payload (`checkpoint_version < 2`)
instead of pretending to continue its optimizer trajectory. Historical
adapter files remain valid for replay through the evaluator aliases.

**Unit-level status:** `RESUME_PARITY=PASS`.

The bounded smoke also resumed H/A/C/AC from the shared E1 full-state
checkpoint; all four arms reached E2 with complete checkpoints.

**Scope note:** no broad H2 full training was launched in this bounded audit;
the factorial launcher is guarded and requires `RUN_FULL_TRAIN=YES`.

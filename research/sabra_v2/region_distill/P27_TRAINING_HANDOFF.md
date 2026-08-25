# P27 Training Handoff

This is the frozen P27 V1 execution base. It requires the exact P26 assets in
`P27_PROTOCOL.json`; the training loader verifies their SHA-256 digests and
refuses substitutions. `REGION_TEACHER_HEADROOM=NOT_CHECKED` is intentional.

Use the prepared environment:

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/venvs/acdclip
export PYTHONPATH=/workspace/ACD-CLIP-sabra${PYTHONPATH:+:$PYTHONPATH}
```

First run the metadata-only preflight for the planned fold:

```bash
python -m tools.sabra_v2.audit_region_distill \
  --held-class candle \
  --output runs/sabra_v2/p27_region_distill_v1/preflight_candle
```

On an authorized scientific machine, execute each fold with the exact assets
and then score only the immutable GT-free held predictions:

```bash
P26_CHECKPOINT=runs/phase4v/v1_7/readiness_full/adapter_5.pth
CLIP_ASSET=model/ViT-L-14-336px.pt
VISA_ROOT=/workspace/data/source/VisA
for HELD in candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum; do
  OUT=runs/sabra_v2/p27_region_distill_v1/${HELD}
  python -m tools.sabra_v2.train_region_distill \
    --held-class "${HELD}" --visa-root "${VISA_ROOT}" \
    --p26-checkpoint "${P26_CHECKPOINT}" --clip-asset "${CLIP_ASSET}" \
    --output "${OUT}" --cache-dir "/workspace/p27_cache/${HELD}" \
    --epochs 20 --batch-size 1 --learning-rate 0.001 --seed 0
  python -m tools.sabra_v2.evaluate_region_distill \
    --held-class "${HELD}" --visa-root "${VISA_ROOT}" \
    --p26-checkpoint "${P26_CHECKPOINT}" --clip-asset "${CLIP_ASSET}" \
    --adapter-checkpoint "${OUT}/p27_region_adapter.pt" --output "${OUT}/predictions"
  python -m tools.sabra_v2.score_region_distill \
    --held-class "${HELD}" --visa-root "${VISA_ROOT}" \
    --predictions "${OUT}/predictions/p27_held_predictions.pt" --output "${OUT}/metrics"
done
```

The train entrypoint reads masks only from the 11 source classes. The teacher
is source-GT-only. It first writes a lossless fold-local FP32 memory-mapped
cache with per-file SHA-256 provenance and reuses the exact frozen tensors for
all 20 epochs. The prediction entrypoint uses `VisaEvidenceDataset`, has no
GT/mask input, and rejects engineering-only checkpoints. The scorer runs only
after saved held predictions and performs zero fitting or teacher steps.

Do not run MVTec or Medical paths. Do not use historical R0 cache artifacts,
change region size, introduce prediction/controller heads, or alter weights,
losses, or teacher semantics without an explicit P27 review.

#!/usr/bin/env bash
set -euo pipefail

# Organize the repo after current training/testing jobs have finished.
# Do not run while train.py/test.py or the train-then-test wrapper is active.

cd "$(dirname "$0")/../.."

if pgrep -af "ACD-CLIP-base-new-phase1/(train.py|test.py|run_phase1|test_phase1)" >/dev/null; then
  echo "Active ACD-CLIP-base-new-phase1 train/test process detected. Abort."
  echo "Run this again after the current train/test job finishes."
  exit 1
fi

mkdir -p \
  docs \
  scripts/baseline \
  scripts/phase1 \
  scripts/phase1_ablation \
  scripts/diagnostics \
  runs/baseline \
  runs/phase1 \
  runs/phase1_ablation \
  runs/phase1_dora_test \
  runs/smoke

move_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ] || [ -L "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
  fi
}

link_if_missing() {
  local target="$1"
  local link="$2"
  if [ ! -e "$link" ] && [ ! -L "$link" ]; then
    ln -s "$target" "$link"
  fi
}

# Docs/notes.
move_if_exists BASELINE_REPRO_DEBUG.md docs/BASELINE_REPRO_DEBUG.md
move_if_exists EXPERIMENT_LOG_PHASE1.md docs/EXPERIMENT_LOG_PHASE1.md
move_if_exists PHASE1A_RESULTS_DEBUG_PLAN.md docs/PHASE1A_RESULTS_DEBUG_PLAN.md
move_if_exists PHASE1_BACKUP_PUSH_NOTES.md docs/PHASE1_BACKUP_PUSH_NOTES.md
move_if_exists PHASE1_DFG_ATTENTION_SS2D_PLAN.md docs/PHASE1_DFG_ATTENTION_SS2D_PLAN.md

# Baseline scripts.
move_if_exists train_baseline_noamp_n3.sh scripts/baseline/train_baseline_noamp_n3.sh

# Phase1 scripts.
for f in \
  train_phase1_v1_attn_tau4_n3.sh \
  train_phase1_v2_attn_tau8_n3.sh \
  train_phase1_v3_attn_tau8_ss2d_g02_n3.sh \
  train_phase1_v3c_attn_tau8_ss2d_weightres_beta010_n3.sh \
  train_phase1_v3c_attn_tau8_ss2d_weightres_betawarm010_n3.sh \
  train_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3.sh \
  train_phase1a_attn_n3.sh \
  train_phase1a_attn_tau8_n3.sh \
  test_phase1_v2_attn_tau8_selected_epochs.sh \
  test_phase1_v3_attn_tau8_ss2d_g02_selected_epochs.sh \
  test_phase1_v3c_attn_tau8_ss2d_weightres_beta010_selected_epochs.sh \
  test_phase1_v3c_attn_tau8_ss2d_weightres_betawarm010_selected_epochs.sh \
  test_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_selected_epochs.sh \
  test_phase1a_attn_tau8_selected_epochs.sh; do
  move_if_exists "$f" "scripts/phase1/$f"
done

# Current ablation script.
move_if_exists run_phase1_v3c_textw_ablation_train_then_test.sh \
  scripts/phase1_ablation/run_phase1_v3c_textw_ablation_train_then_test.sh

# Shared test script stays close to root code in scripts/common, with root symlink.
mkdir -p scripts/common
move_if_exists test_6medical_selected_epochs.sh scripts/common/test_6medical_selected_epochs.sh
link_if_missing scripts/common/test_6medical_selected_epochs.sh test_6medical_selected_epochs.sh

# Diagnostics.
move_if_exists debug_retina_protocol.sh scripts/diagnostics/debug_retina_protocol.sh

# Runs/checkpoints.
move_if_exists test_train_main_base_smoke runs/baseline/test_train_main_base_smoke
move_if_exists tmp_baseline_current_smoke runs/baseline/tmp_baseline_current_smoke

move_if_exists phase1_v1_attn_tau4 runs/phase1/01_phase1_v1_attn_tau4
move_if_exists phase1_v2_attn_tau8 runs/phase1/02_phase1_v2_attn_tau8
move_if_exists phase1_v3_attn_tau8_ss2d_g02 runs/phase1/03_phase1_v3_attn_tau8_ss2d_g02
move_if_exists phase1_v3c_attn_tau8_ss2d_weightres_betawarm010 runs/phase1/04_phase1_v3c_weightres_betawarm010_old
move_if_exists phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3 runs/phase1/05_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_best
move_if_exists phase1_v3_smoke runs/smoke/phase1_v3_smoke

move_if_exists phase1_v3c_textw010_nokeyanchor runs/phase1_ablation/phase1_v3c_textw010_nokeyanchor

# Rename abandoned DoRA experiments as Phase1 DoRA tests, not Phase2 direction.
move_if_exists phase2_dora_text_v3c_weightres_fp32attn_tau8_g3 \
  runs/phase1_dora_test/phase1_dora_test_text_v3c_weightres_fp32attn_tau8_g3_wrong_impl
move_if_exists phase2_dora_textlr1e4_clean_v3c_weightres_fp32attn_tau8_g3 \
  runs/phase1_dora_test/phase1_dora_test_textlr1e4_clean_v3c_weightres_fp32attn_tau8_g3

# Compatibility symlinks for important runs/scripts.
link_if_missing runs/phase1/05_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_best \
  phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3
link_if_missing runs/phase1_ablation/phase1_v3c_textw010_nokeyanchor \
  phase1_v3c_textw010_nokeyanchor
link_if_missing scripts/phase1/test_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_selected_epochs.sh \
  test_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_selected_epochs.sh
link_if_missing scripts/phase1_ablation/run_phase1_v3c_textw_ablation_train_then_test.sh \
  run_phase1_v3c_textw_ablation_train_then_test.sh

cat > docs/REPO_LAYOUT.md <<'EOF'
# Repo Layout

Root keeps source code and shared entrypoints:
- `train.py`, `test.py`, `utils.py`
- `model/`, `dataset/`

Organized folders:
- `runs/baseline/`: baseline smoke/local baseline artifacts
- `runs/phase1/`: Phase1 numbered runs
- `runs/phase1_ablation/`: Phase1 ablations, e.g. `text_adapt_weight=0.10`
- `runs/phase1_dora_test/`: abandoned DoRA tests, kept as historical Phase1 tests
- `runs/smoke/`: smoke/debug artifacts
- `scripts/baseline/`: baseline train scripts
- `scripts/phase1/`: Phase1 train/test scripts
- `scripts/phase1_ablation/`: current ablation wrappers
- `scripts/common/`: shared test wrappers
- `scripts/diagnostics/`: diagnostic scripts
- `docs/`: notes/plans/debug docs

Compatibility symlinks are kept at root for the most-used scripts and current
best run, so old commands continue to work.
EOF

echo "Organization complete."

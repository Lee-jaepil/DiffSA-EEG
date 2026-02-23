#!/usr/bin/env bash
# =============================================================================
# DiffSA-EEG Training Script
# =============================================================================
# Usage:
#   bash scripts/run_train.sh
#
# Before running, set DATA_ROOT and MODEL_ROOT below, or export them
# as environment variables.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable paths  (override via environment variables)
# ---------------------------------------------------------------------------
DATA_ROOT="${DATA_ROOT:-./Preprocessed_Data}"
MODEL_ROOT="${MODEL_ROOT:-./model_result}"
DEVICE="${DEVICE:-cuda:0}"

# ---------------------------------------------------------------------------
# 1. Train DiffSA-EEG (proposed model) on TUAB (TRA montage)
# ---------------------------------------------------------------------------
echo "=== Training DiffSA-EEG on TUAB (TRA) ==="
python main.py \
    --device "${DEVICE}" \
    --dataset TRA \
    --model SSDA_attn_SF_CBAM \
    --model_save_dir "${MODEL_ROOT}" \
    --num_epochs 100 \
    --batch_size 4 \
    --alpha 0.1 \
    --num_runs 10 \
    --seed 42

# ---------------------------------------------------------------------------
# 2. Train DiffSA-EEG on TUEP
# ---------------------------------------------------------------------------
echo "=== Training DiffSA-EEG on TUEP ==="
python main.py \
    --device "${DEVICE}" \
    --dataset TUEP \
    --model SSDA_attn_SF_CBAM \
    --model_save_dir "${MODEL_ROOT}" \
    --num_epochs 100 \
    --batch_size 8 \
    --alpha 0.1 \
    --num_runs 10 \
    --seed 42

# ---------------------------------------------------------------------------
# 3. Train baseline models (EEGNet, ChronoNet, BDTCN, Deep4Net)
# ---------------------------------------------------------------------------
for MODEL in EEGNet ChronoNet BDTCN Deep4Net; do
    echo "=== Training baseline: ${MODEL} ==="
    python main.py \
        --device "${DEVICE}" \
        --dataset TRA \
        --model "${MODEL}" \
        --model_save_dir "${MODEL_ROOT}" \
        --num_epochs 100 \
        --batch_size 4 \
        --num_runs 10 \
        --seed 42
done

echo "=== All training complete ==="

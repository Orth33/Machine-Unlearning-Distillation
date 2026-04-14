#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

eval "$(micromamba shell hook -s bash)"
micromamba activate ffcv
set -u

SEQUENCE_LOG="$ROOT_DIR/ham10000_ga_repair_v2_run_sequence.log"
DENSE_DIR="$ROOT_DIR/ham10000_ga_repair_v2_class0_dense_real"
SPARSE_DIR="$ROOT_DIR/ham10000_ga_repair_v2_class0_sparse_p80_real"

run_dense() {
  mkdir -p "$DENSE_DIR"
  python -u main_forget.py \
    --data ./data/ham10000 \
    --dataset ham10000 \
    --arch efficientnet_b0 \
    --input_size 224 \
    --num_classes 7 \
    --batch_size 64 \
    --workers 8 \
    --print_freq 20 \
    --save_dir "$DENSE_DIR" \
    --mask ham10000_effb0_baseline_v2_p80/0model_SA_best.pth.tar \
    --unlearn GA_repair_v2 \
    --class_to_replace 0 \
    --unlearn_lr 0.003 \
    --repair_lr 0.006 \
    --retain_steps 4 \
    --forget_steps 1 \
    --entropy_beta 1.5 \
    --anchor_lambda_start 2e-5 \
    --anchor_lambda_end 2e-4 \
    --forget_ce_weight 0.20 \
    --kd_lambda 1.0 \
    --distill_temperature 2.0 \
    --forget_warmup_epochs 3 \
    --freeze_bn \
    --smoothing_eps 0.05 \
    --mixup_alpha 0.2 \
    --sam_rho 2.0 \
    --unlearn_epochs 12 \
    --bn_recalibrate_batches 128 \
    --pretrained \
    > "$DENSE_DIR/train.log" 2>&1
}

run_sparse() {
  mkdir -p "$SPARSE_DIR"
  python -u main_forget.py \
    --data ./data/ham10000 \
    --dataset ham10000 \
    --arch efficientnet_b0 \
    --input_size 224 \
    --num_classes 7 \
    --batch_size 64 \
    --workers 8 \
    --print_freq 20 \
    --save_dir "$SPARSE_DIR" \
    --mask ham10000_effb0_baseline_v2_p80/1model_SA_best.pth.tar \
    --unlearn GA_repair_v2 \
    --class_to_replace 0 \
    --unlearn_lr 0.004 \
    --repair_lr 0.005 \
    --retain_steps 2 \
    --forget_steps 3 \
    --entropy_beta 1.5 \
    --anchor_lambda_start 2e-5 \
    --anchor_lambda_end 2e-4 \
    --forget_ce_weight 0.25 \
    --kd_lambda 1.0 \
    --distill_temperature 2.0 \
    --forget_warmup_epochs 3 \
    --freeze_bn \
    --smoothing_eps 0.05 \
    --mixup_alpha 0.2 \
    --sam_rho 2.0 \
    --unlearn_epochs 12 \
    --bn_recalibrate_batches 128 \
    --pretrained \
    > "$SPARSE_DIR/train.log" 2>&1
}

{
  echo "[$(date '+%F %T')] Starting dense GA_repair_v2 run"
  run_dense
  echo "[$(date '+%F %T')] Dense GA_repair_v2 run finished"
  echo "[$(date '+%F %T')] Starting sparse GA_repair_v2 run"
  run_sparse
  echo "[$(date '+%F %T')] Sparse GA_repair_v2 run finished"
} >> "$SEQUENCE_LOG" 2>&1

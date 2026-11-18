#!/usr/bin/env bash
# Train CLIP on InfCLIP-selected subsets over several ratios and seeds, then evaluate on ImageNet.
# Global batch size = 256 per GPU x 4 GPUs = 1024 (paper setting).
set -euo pipefail

DATASET=${DATASET:-cc3m}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Arrow shard directory}
IMAGENET_VAL=${IMAGENET_VAL:?set IMAGENET_VAL to the ImageNet val directory}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

if [ "$DATASET" = "cc3m" ]; then
  EPOCHS=10; RATIOS=${RATIOS:-"0.05 0.1 0.2 0.3 0.4 0.5"}
else
  EPOCHS=5;  RATIOS=${RATIOS:-"0.05 0.1 0.2 0.3 0.5"}
fi
SEEDS=${SEEDS:-"1 2 3"}

for ratio in $RATIOS; do
  for seed in $SEEDS; do
    python subset_run.py \
      --dataset_name "$DATASET" \
      --data_root "$DATA_ROOT" \
      --score_name InfCLIP \
      --score_path "outputs/scores/${DATASET}_infclip_l10_B1024_classmean.npy" \
      --subset_ratio "$ratio" \
      --seed "$seed" \
      --num_train_epochs "$EPOCHS" \
      --per_gpu_train_batch_size 256 \
      --imagenet_val "$IMAGENET_VAL" \
      --output_dir outputs/saved_model \
      --result_path "outputs/results/${DATASET}_imagenet.txt"
  done
done

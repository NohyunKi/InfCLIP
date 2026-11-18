#!/usr/bin/env bash
# Extract teacher features and compute InfCLIP scores for CC3M (set DATASET=cc12m for CC12M).
set -euo pipefail

DATASET=${DATASET:-cc3m}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the Arrow shard directory}
IMAGENET_VAL=${IMAGENET_VAL:?set IMAGENET_VAL to the ImageNet val directory}
DEVICE=${DEVICE:-cuda:0}

mkdir -p outputs/features outputs/scores

python extract_feature.py \
  --dataset_name "$DATASET" \
  --data_root "$DATA_ROOT" \
  --output_path "outputs/features/${DATASET}_image_text_features.pt" \
  --batch_size 256 --device "$DEVICE"

if [ ! -f outputs/features/imagenet_classmean_features.pt ]; then
  python extract_target_feature.py \
    --data_root "$IMAGENET_VAL" \
    --wordnet_path zero_shot/WordNetId2ClassName.txt \
    --output_path outputs/features/imagenet_classmean_features.pt \
    --batch_size 256 --device "$DEVICE"
fi

python calculate.py \
  --train_feature_path "outputs/features/${DATASET}_image_text_features.pt" \
  --target_feature_path outputs/features/imagenet_classmean_features.pt \
  --output_path "outputs/scores/${DATASET}_infclip_l10_B1024_classmean.npy" \
  --batch_size 1024 --logit_scale 10 --device "$DEVICE"

#!/bin/bash

# DN-DETR Training Script with Custom Dataset
# Training for 200 epochs with patience=3 early stopping

echo "Starting DN-DETR training with custom dataset..."
echo "Dataset path: coco2017_augmented/"
echo "Epochs: 200"
echo "Patience: 3"
echo "Output directory: logs/dn_dab_detr/custom_training"
echo "GPU: NVIDIA GeForce GTX 1650"
echo ""

# Activate virtual environment
source dn_detr_env/bin/activate

# Start training with GPU monitoring
python3 train_custom.py \
  -m dn_dab_detr \
  --output_dir logs/dn_dab_detr/custom_training \
  --batch_size 2 \
  --epochs 200 \
  --lr_drop 150 \
  --coco_path coco2017_augmented/ \
  --use_dn \
  --patience 3 \
  --num_workers 2 \
  --save_checkpoint_interval 25

echo "Training completed!"
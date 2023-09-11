#!/bin/bash

# Set environment variables
model_version='v1_1_text_encoder_unet'
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./text_encoder_dataset"
export CLASS_DIR="path_to_class_images"
export OUTPUT_DIR="./models/$model_version"

# Create OUTPUT_DIR if it doesn't exist
mkdir -p "$OUTPUT_DIR"
# Run the Python code
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="$MODEL_NAME"  \
  --instance_data_dir="$INSTANCE_DIR" \
  --class_data_dir="$CLASS_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a microscopic image of human embryo at certain phase recorded at a certain focal plane" \
  --class_prompt="a microscopic image of a human embryo" \
  --resolution=512 \
  --train_batch_size=2 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --enable_xformers_memory_efficient_attention

# Optional: Print a message when the script finishes
echo "Training script completed."

cp ./wandb/latest-run/files/config.yaml "$OUTPUT_DIR"

python test_diffusion.py \
  --m=$OUTPUT_DIR \
  --n=3 \
  --v=$model_version

#!/bin/bash
# Set environment variables
model_version='v1_1'
export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="./dataset/AA83-7"
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
  --instance_prompt="Medical image with optical microscope of a human embryo at a certain development stage" \
  --class_prompt="Microscopic image of a human embryo" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
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












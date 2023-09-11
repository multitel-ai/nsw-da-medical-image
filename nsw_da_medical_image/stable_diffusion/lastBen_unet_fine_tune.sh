#!/bin/bash

# Set environment variables
model_version="$4"
instance_dir_name="$1"
instance_prompt="$2"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="$instance_dir_name"
export CLASS_DIR="path_to_class_images"
#export CAPTIONS_DIR="./text_captions"
export OUTPUT_DIR="$5/$model_version"
export WANDB_PROJECT_NAME="stable-diffusion-2-1-fine-tune-unet-lastBen"

# Create OUTPUT_DIR if it doesn't exist
mkdir -p "$OUTPUT_DIR"
# mkdir -p "$CLASS_DIR"
# rm ./path_to_class_images/*
# Run the Python code
accelerate launch train_dreambooth_lastBen.py \
  --wandb_project="$WANDB_PROJECT_NAME" \
  --wandb_run_name="$4" \
  --train_only_unet \
  --save_starting_step=500 \
  --class_data_dir="$CLASS_DIR" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="$instance_prompt" \
  --class_prompt="a grayscale microscopic image of a human embryo" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=7e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=$3\
  --num_validation_images=5 \
  --validation_steps=100 \
  --validation_prompt="$instance_prompt"

    
# Optional: Print a message when the script finishes
echo "Training script completed."

latest_wandb_folders=($(ls -td ./wandb/*/ | head -3))

# Check if there are at least two wandb folders
if [ ${#latest_wandb_folders[@]} -ge 3 ]; then
  # Copy the config.yaml file from the second-to-latest folder to OUTPUT_DIR
  cp "${latest_wandb_folders[2]}files/config.yaml" "$OUTPUT_DIR"
else
  echo "Not enough wandb folders found."
fi

python test_diffusion.py \
  --m="$OUTPUT_DIR" \
  --n=$7 \
  --v="$model_version" \
  --p="$instance_prompt" \
  --o="$6"
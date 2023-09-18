#!/bin/bash

# make sure the script exit at the first error to avoid continuing
set -euxo pipefail

# Set environment variables
model_version="$4"
instance_prompt="$2"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export CLASS_DIR="path_to_class_images"
#export CAPTIONS_DIR="./text_captions"
export OUTPUT_DIR="$5/$model_version"
export WANDB_PROJECT_NAME="stable-diffusion-2-1-fine-tune-unet-lastBen"

# Create OUTPUT_DIR if it doesn't exist
mkdir -p "$OUTPUT_DIR"
# mkdir -p "$CLASS_DIR"
# rm ./path_to_class_images/*
# Run the Python code
accelerate launch -m nsw_da_medical_image.stable_diffusion.train_dreambooth_lastBen \
  --wandb_project="$WANDB_PROJECT_NAME" \
  --Session_dir="$8" \
  --wandb_run_name="$4" \
  --train_only_unet \
  --checkpointing_steps=$9 \
  --save_starting_step=$9 \
  --save_n_steps=$9 \
  --class_data_dir="$CLASS_DIR" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$1" \
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
  --max_train_steps=$3 \
  --num_validation_images=${11} \
  --validation_steps=${10} \
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

python -m nsw_da_medical_image.stable_diffusion.test_diffusion \
  --m="$OUTPUT_DIR" \
  --n="$7" \
  --v="$model_version" \
  --p="$instance_prompt" \
  --o="$6"

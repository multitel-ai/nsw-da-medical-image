#!/bin/bash

# Set environment variables
model_version='v1_1'
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="./dataset/AA83-7"
export CLASS_DIR="path_to_class_images"
export OUTPUT_DIR="./models/$model_version"
export WANDB_PROJECT_NAME="stable-diffusion-2-1-fine-tune-lastBen_$1"

# Create OUTPUT_DIR if it doesn't exist
mkdir -p "$OUTPUT_DIR"
# Run the Python code
accelerate launch train_dreambooth_lastBen.py \
  --wandb_project="$WANDB_PROJECT_NAME"\
  --train_only_unet \
  --save_starting_step=500 \
  --class_data_dir="$CLASS_DIR" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="Medical image with optical microscope of a human embryo at a certain development stage" \
  --class_prompt="A grayscale microscopic image of a human embryo" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --num_class_images=200 \
  --num_validation_images=5 \
  --validation_steps=100\
  --validation_prompt="Medical image with optical microscope of a human embryo at a certain development stage"



    
# Optional: Print a message when the script finishes
echo "Training script completed."

cp ./wandb/latest-run/files/config.yaml "$OUTPUT_DIR"

python test_diffusion.py \
  --m=$OUTPUT_DIR \
  --n=3 \
  --v=$model_version

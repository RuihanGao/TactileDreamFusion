#!/bin/bash

# Define your list of class prefixes (you can also automatically generate this if needed)
class_prefixes=('redcloth1')  # 'strawberry' 'avocado' 'glasstumbler' 'football' 'corn' 'goldgoat'  'strawhat' 'cuttingboard' 'spongehard' 'orangeglove' 'strawberryleaves' 'greensweater' 'footballhandle' 'potato' 'orange' 'trunk' 'clothbag' 'cantaloupe' 'tabletennishandle' 'redcloth2' 'spongesoft' 'blackbase' 'tabletennisface' 'rock' 'corkmat' 'whitesponge' 'pinkcloth' 'purpleglove' 'metalframe'

# Iterate over each class_prefix
for class_prefix in "${class_prefixes[@]}"
do
    echo "Running training for class: $class_prefix"

    # Set the environment variables dynamically
    export MODEL_NAME="CompVis/stable-diffusion-v1-4" # "runwayml/stable-diffusion-v1-5" 
    export INSTANCE_DIR="./data/normal_textures_${class_prefix}_selected"
    export CLASS_DIR="./data/all_normal_textures" #  "./data/normal_textures_except_${class_prefix}"


    # Run the accelerate command
    CUDA_VISIBLE_DEVICES=3 accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path=$MODEL_NAME  \
        --instance_data_dir=$INSTANCE_DIR \
        --class_data_dir=$CLASS_DIR \
        --output_dir="output/lora_${class_prefix}_sks" \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --num_dataloader_workers=1 \
        --mixed_precision="no" \
        --instance_prompt="sks normal map" \
        --class_prompt="normal map" \
        --use_lora \
        --train_text_encoder \
        --gradient_checkpointing
        
    echo "Training completed for class: $class_prefix"
done

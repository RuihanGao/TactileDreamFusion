#!/bin/bash

# Default to testing mode unless -train is specified
is_train=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -train) is_train=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Common variables
mesh_obj="a_cactus_in_a_pot_3"
texture="Orange"
texture2_name="OrangeGlove"
postfix="_example"

if $is_train; then
    echo "Running in training mode..."
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config configs/text_tactile_TSDS_multipart.yaml \
        save_path=${mesh_obj}_${texture}_${texture2_name}${postfix} \
        mesh=logs/${mesh_obj}/${mesh_obj}_mesh.obj \
        tactile_texture_object=${texture} \
        texture2_name=${texture2_name}
else
    echo "Running in testing mode..."
    echo "Visualizing: $mesh_obj, $texture, $texture2_name, postfix: $postfix"

    # Render different modes
    vis_modes=("lambertian" "tangent" "albedo" "normal" "shading_normal" "tactile_normal" "uv" "label_map")

    for ((k=0; k<${#vis_modes[@]}; k++)); do
        vis_mode=${vis_modes[$k]}
        echo "Render: Texture: $texture, Mode: $vis_mode"

        # Render video
        python vis_render.py \
            logs/${mesh_obj}_${texture}_${texture2_name}${postfix}/${mesh_obj}_${texture}_${texture2_name}${postfix}.obj \
            --mode $vis_mode \
            --save_video logs/${mesh_obj}_${texture}_${texture2_name}${postfix}/${mesh_obj}_${texture}_${texture2_name}${postfix}_${vis_mode}.mp4 &

        # Render front and back views as images
        python vis_render.py \
            logs/${mesh_obj}_${texture}_${texture2_name}${postfix}/${mesh_obj}_${texture}_${texture2_name}${postfix}.obj \
            --mode $vis_mode \
            --elevation 0 \
            --num_azimuth 2 \
            --save logs/${mesh_obj}_${texture}_${texture2_name}${postfix}/ &
    done

    # Wait for all background processes to finish
    wait
fi
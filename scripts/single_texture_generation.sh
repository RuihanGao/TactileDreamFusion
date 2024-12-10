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
mesh_obj="an_avocado_2"
texture="avocado"
postfix="_example"

if $is_train; then
    echo "Running in training mode..."
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config configs/text_tactile_TSDS.yaml \
        save_path=${mesh_obj}_${texture}${postfix} \
        mesh=data/base_meshes/${mesh_obj}/${mesh_obj}_mesh.obj \
        tactile_texture_object=${texture}
else
    echo "Running in testing mode..."
    # Render different modes
    vis_modes=("lambertian" "albedo" "tactile_normal" "viewspace_normal" "shading_normal") # "tangent" "uv" "normal"

    for ((q=0; q<${#vis_modes[@]}; q++)); do
        vis_mode=${vis_modes[$q]}
        echo "Render: Texture: $texture, Mode: $vis_mode"

        # Render video
        python vis_render.py \
            logs/${mesh_obj}_${texture}${postfix}/${mesh_obj}_${texture}${postfix}.obj \
            --mode $vis_mode \
            --save_video logs/${mesh_obj}_${texture}${postfix}/${mesh_obj}_${texture}${postfix}_${vis_mode}.mp4 &

        # Render front and back views as images
        python vis_render.py \
            logs/${mesh_obj}_${texture}${postfix}/${mesh_obj}_${texture}${postfix}.obj \
            --mode $vis_mode \
            --elevation 0 \
            --num_azimuth 2 \
            --save logs/${mesh_obj}_${texture}${postfix}/ &
    done

    # Wait for all background processes to finish
    wait
fi
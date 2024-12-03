# TactileDreamFusion
official implementation of paper "Tactile DreamFusion: Exploiting Tactile Sensing for 3D Generation"

## Examples
```
mesh_obj="an_avocado"
texture="avocado"
postfix="_test"
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/text_tactile_TSDS.yaml save_path=${mesh_obj}_${texture}${postfix} mesh=logs/${mesh_obj}/${mesh_obj}_mesh.obj tactile_texture_object=${texture}
```

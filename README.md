# TactileDreamFusion
official implementation of paper "Tactile DreamFusion: Exploiting Tactile Sensing for 3D Generation"

## Examples
* single texture generation
```
mesh_obj="an_avocado"
texture="avocado"
postfix="_test"
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/text_tactile_TSDS.yaml save_path=${mesh_obj}_${texture}${postfix} mesh=logs/${mesh_obj}/${mesh_obj}_mesh.obj tactile_texture_object=${texture}
```

* multi-part texture generation
```
mesh_obj="a_cactus_in_a_pot_3"
texture="Orange"
texture2_name="OrangeGlove"
postfix="_test"
CUDA_VISIBLE_DEVICES=3 python main.py --config configs/text_tactile_TSDS_multipart.yaml save_path=${mesh_obj}_${texture}_${texture2_name}${postfix} mesh=logs/${mesh_obj}/${mesh_obj}_mesh.obj tactile_texture_object=${texture} texture2_name=${texture2_name}
```
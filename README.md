# TactileDreamFusion

<a href="https://arxiv.org/abs/2412.06785"><img src="https://img.shields.io/badge/arXiv-2412.06785-b31b1b.svg"></a>
<a href='https://ruihangao.github.io/TactileDreamFusion/'><img src='https://img.shields.io/badge/Project_Page-gray?logo=github'></a>
<a href='https://huggingface.co/datasets/Ruihan28/TactileDreamFusion'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-blue'></a>
<a href='https://github.com/dunbar12138/blender-render-toolkit/tree/main'><img src='https://img.shields.io/badge/Blender-render_toolkit-1?logo=blender'></a>
<a href="https://github.com/RuihanGao/TactileDreamFusion?tab=MIT-1-ov-file#"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>

[**Project**](https://ruihangao.github.io/TactileDreamFusion/) | [**Paper**](https://arxiv.org/abs/2412.06785)

<!-- https://github.com/user-attachments/assets/400db33f-2843-4201-ae1b-a1434700b4ff -->
<img src="assets/imgs/teaser.gif" alt="teaser gif" width="800">

> **3D content creation with touch**: TactileDreamFusion integrates high-resolution tactile sensing with diffusion-based image priors to enhance fine geometric details for text- or image-to-3D generation. The following results are rendered using Blender, with full-color rendering on the top and normal rendering at the bottom.

**Tactile DreamFusion: Exploiting Tactile Sensing for 3D Generation** <br>
[Ruihan Gao](https://ruihangao.github.io/),
[Kangle Deng](https://dunbar12138.github.io/),
[Gengshan Yang](https://gengshan-y.github.io/),
[Wenzhen Yuan](https://siebelschool.illinois.edu/about/people/all-faculty/yuanwz),
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/) <br>
Carnegie Mellon University <br>
**NeurIPS 2024**

## Results

The following results are rendered using [blender-render-toolkit](https://github.com/dunbar12138/blender-render-toolkit/tree/main).

### Same Object with Diverse Textures

We show diverse textures synthesized on the same object, which facilitates the custom design of 3D assets.

<!-- https://github.com/user-attachments/assets/218c1e7d-9b89-4353-9759-0c12eea81f67 -->
<img src="assets/imgs/application.gif" alt="application gif" width="800">

### Single Texture Generation

We show 3D generation with a single texture. Our method generates realistic and coherent visual textures and geometric details.

<!-- https://github.com/user-attachments/assets/befdb815-69a3-4c98-b82b-3c58e3eab63a -->
<img src="assets/imgs/singleTexture.gif" alt="singleTexture gif" width="800">

### Multi-Part Texture Generation

This grid demonstrates different render types for each object: predicted label map, albedo, normal map, zoomed-in normal patch, and full-color rendering.

<!-- https://github.com/user-attachments/assets/e9ce2bbb-1b60-4b0d-993a-2b55152e360a -->
<img src="assets/imgs/multiPart.gif" alt="multiPart gif" width="800">


## Getting Started

### Environment setup

Our environment has been tested on linux, python 3.10.13, pytorch 2.2.1, and CUDA 12.1.

```
git clone https://github.com/RuihanGao/TactileDreamFusion.git
cd TactileDreamFusion
conda create -n TDF python=3.10
conda activate TDF
pip install torch==2.2.1+cu121 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/dunbar12138/blender-render-toolkit.git
cd blender-render-toolkit
git checkout tactileDreamfusion

```

### Hardware requirements

All results in the paper were produced on a single **NVIDIA A6000 (48 GB)**, which is what the default single-texture config is tuned for (`batch_size: 4`, `patch_batch_size: 4` in `configs/text_tactile_TSDS.yaml`). The multi-part config is lighter and already defaults to `batch_size: 1`.

On smaller GPUs, edit `configs/text_tactile_TSDS.yaml` and lower `batch_size` to trade throughput for memory. A 24 GB card (e.g. RTX 3090 / 4090) fits `batch_size: 2` for single-texture training. If you still hit OOM, also reduce `patch_batch_size` (default 4).

### Download dataset and TextureDreamBooth weights

- Run the following script to download our tactile texture dataset and example base meshes to `data` folder.

```
bash scripts/download_hf_dataset.sh
```

- Run the following script to download our pretrained TextureDreamBooth weights and patch data to `TextureDreambooth` folder.

```
bash scripts/download_hf_model.sh
```

### Single texture generation

- Training

```
bash scripts/single_texture_generation.sh -train
```

It takes about 10 mins on a single A6000 gpu to run single texture generation for a mesh.

All artifacts are written to `logs/<save_path>/` (for the default script, `logs/an_avocado_2_avocado_example/`):

- **Mesh & textures** — `<save_path>.obj` + `.mtl`, `_albedo.png`, `_tactile_normal.png` are the final textured mesh; `_initialized.*` is the checkpoint written after the 150-iter init stage, before refinement begins.
- **Config & loss logs** — `_opt.json` (the resolved config used for the run), `_loss_dict_all.pkl` (per-iter value for every loss term), `_loss_plot.png` (loss curves).
- **Per-iter debug videos** (one frame per iter, 256×256) show how each signal evolves over training:
  - `_rendered_albedos_list.mp4` / `_rendered_target_albedos_list.mp4` — learned vs. target albedo
  - `_rendered_lambertians_list.mp4` — shaded rendering fed to the ControlNet SDS loss
  - `_rendered_perturb_normals_list.mp4` / `_rendered_target_perturb_normals_list.mp4` — learned vs. target tactile normal
  - `_rendered_guidance_perturb_normals_list.mp4` — TextureDreambooth-refined normal used as the tactile guidance target
  - `_controlnet_refined_images_list.mp4` / `_controlnet_control_images_list.mp4` — ControlNet refinement output and its normal-map control
  - `_rendered_*_patch_list.mp4` — the same signals rendered from close-up patch views (used for tactile-scale supervision)
- **Side-by-side summary** — `_SDS_concat_rendering.mp4` shows `(albedo | lambertian | ControlNet-refined)` per-iter to inspect the SDS loop.

- Visualization (nvdiffrast)

```
bash scripts/single_texture_generation.sh
```

For each rendering mode in `{lambertian, albedo, tactile_normal, viewspace_normal, shading_normal}`, the script writes into the same `logs/<save_path>/` folder:

- `<save_path>_<mode>.mp4` — 360° camera orbit at a fixed elevation (1° azimuth step, 30 fps).
- `<elevation>_<azimuth>_light_<le>_<la>_<amb>_<mode>.png` — front (azimuth 0) and back (azimuth 180) frames as PNGs.

Meaning of each mode:
- `albedo` — learned RGB texture only (no lighting)
- `lambertian` — diffuse shading using surface normal + tactile perturbation
- `tactile_normal` — the learned high-frequency tactile normal map in tangent space
- `shading_normal` — combined (surface + tactile) normal in world space
- `viewspace_normal` — shading normal in camera space (comparable to ControlNet's BAE normals)

Example 360° orbit video (`lambertian` mode for `an_avocado_2_avocado_example`, file `logs/an_avocado_2_avocado_example/an_avocado_2_avocado_example_lambertian.mp4`). The other four modes follow the same filename pattern.

<p align="center">
<video src="assets/imgs/example_nvdiffrast_lambertian.mp4" controls width="400"></video>
</p>

- Visualization (blender)
  Note: After training, visualize different output meshes in `logs` directory by changing `mesh_objs` list in each bash script.

```
cd blender-render-toolkit
bash scripts/batch_blender_albedo.sh
bash scripts/batch_blender_normal.sh
bash scripts/batch_blender.sh
```

Each script writes into `blender-render-toolkit/output/` — per-frame PNGs under `<obj>_<modality>_rotate/` (252 frames, 360° orbit) and a concatenated `<obj>_<modality>_rotate.mp4`. `batch_blender_normal.sh` also produces single-frame PNGs `<obj>_normal.png` (textured) and `<obj>_normal_geometry.png` (base geometry only).

Example full-color 360° orbit video (`blender-render-toolkit/output/an_avocado_2_avocado_example_full_color_rotate.mp4`). See the matching `_albedo_rotate.mp4` and `_normal_rotate.mp4` in the same directory for the other two modalities.

<p align="center">
<video src="assets/imgs/example_blender_full_color.mp4" controls width="400"></video>
</p>

### Multi-part texture generation

- Training

```
bash scripts/multi_part_texture_generation.sh -train
```

It takes about 15 mins on a single A6000 gpu to run multi-part texture generation for a mesh.

Outputs go to `logs/<save_path>/` with the same layout as single-texture training (see above), plus:

- `_label_map.png` — learned per-part segmentation (red = partA, green = partB), used at render time to mask the two tactile textures.
- `_rendered_labels_list.mp4`, `_rendered_labels_patch_list.mp4` — per-iter label field evolution for full and patch views.
- `_rendered_target_perturb_normal2s_list.mp4` / `_rendered_guidance_perturb_normal2s_list.mp4` — partB's target tactile normal and TextureDreambooth-refined guidance (partA's uses the `_perturb_normals` videos as in the single-texture case).
- `_concat_patch_masks.mp4` — side-by-side `(partA mask | partB mask | target albedo patch)` across training iters, to inspect the DiffSeg segmentation.

- Visualization (nvdiffrast)

```
bash scripts/multi_part_texture_generation.sh
```

Output layout is the same as single-texture visualization (orbit `.mp4` plus front/back PNGs per mode). The multi-part script renders additional modes: `tangent`, `normal`, `uv`, and `label_map` (the per-part segmentation, red = partA, green = partB).

## Citation

If you find this repository useful for your research, please cite the following work.

```
@inproceedings{gao2024exploiting,
      title     = {Tactile DreamFusion: Exploiting Tactile Sensing for 3D Generation},
      author    = {Gao, Ruihan and Deng, Kangle and Yang, Gengshan and Yuan, Wenzhen and Zhu, Jun-Yan},
      booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
      year      = {2024},
}
```

## Acknowlegements

We thank Sheng-Yu Wang, Nupur Kumari, Gaurav Parmar, Hung-Jui Huang, and Maxwell Jones for their helpful comments and discussion. We are also grateful to Arpit Agrawal and Sean Liu for proofreading the draft. Kangle Deng is supported by the Microsoft research Ph.D. fellowship. Ruihan Gao is supported by the A\*STAR National Science Scholarship (Ph.D.).

Part of this codebase borrows from [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) and [DiffSeg](https://github.com/google/diffseg).

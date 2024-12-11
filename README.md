# TactileDreamFusion

<a href="https://arxiv.org/abs/2412.06785"><img src="https://img.shields.io/badge/arXiv-2412.06785-b31b1b.svg"></a>
<a href='https://ruihangao.github.io/TactileDreamFusion/'><img src='https://img.shields.io/badge/Project_Page-gray?logo=github'></a>
<a href='https://huggingface.co/datasets/Ruihan28/TactileDreamFusion'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-blue'></a>
<a href='https://github.com/dunbar12138/blender-render-toolkit/tree/main'><img src='https://img.shields.io/badge/Blender-render_toolkit-1?logo=blender'></a>
<a href="https://github.com/RuihanGao/TactileDreamFusion?tab=MIT-1-ov-file#"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>

[**Project**](https://ruihangao.github.io/TactileDreamFusion/) | [**Paper**](https://arxiv.org/abs/2412.06785)

https://github.com/user-attachments/assets/400db33f-2843-4201-ae1b-a1434700b4ff

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

https://github.com/user-attachments/assets/218c1e7d-9b89-4353-9759-0c12eea81f67

### Single Texture Generation

We show 3D generation with a single texture. Our method generates realistic and coherent visual textures and geometric details.

https://github.com/user-attachments/assets/befdb815-69a3-4c98-b82b-3c58e3eab63a

### Multi-Part Texture Generation

This grid demonstrates different render types for each object: predicted label map, albedo, normal map, zoomed-in normal patch, and full-color rendering.

https://github.com/user-attachments/assets/e9ce2bbb-1b60-4b0d-993a-2b55152e360a

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

- Visualization (nvdiffrast)

```
bash scripts/single_texture_generation.sh
```

- Visualization (blender)
  Note: After training, visualize different output meshes in `logs` directory by changing `mesh_objs` list in each bash script.

```
cd blender-render-toolkit
bash scripts/batch_blender_albedo.sh
bash scripts/batch_blender_normal.sh
bash scripts/batch_blender.sh
```

### Multi-part texture generation

- Training

```
bash scripts/multi_part_texture_generation.sh -train
```

It takes about 15 mins on a single A6000 gpu to run multi-part texture generation for a mesh.

- Visualization (nvdiffrast)

```
bash scripts/multi_part_texture_generation.sh
```

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

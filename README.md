# TactileDreamFusion

[**Project**](https://ruihangao.github.io/TactileDreamFusion/) | [**Paper**]()


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

### Single texture generation

Training

```
bash scripts/single_texture_generation.sh -train
```

Visualization

```
bash scripts/single_texture_generation.sh
```

### Multi-part texture generation

Training

```
bash scripts/multi_part_texture_generation.sh -train
```

Visualization

```
bash scripts/multi_part_texture_generation.sh
```

## Acknowlegements

We thank Sheng-Yu Wang, Nupur Kumari, Gaurav Parmar, Hung-Jui Huang, and Maxwell Jones for their helpful comments and discussion. We are also grateful to Arpit Agrawal and Sean Liu for proofreading the draft. Kangle Deng is supported by the Microsoft research Ph.D. fellowship. Ruihan Gao is supported by the A\*STAR National Science Scholarship (Ph.D.).

Part of this codebase borrows from [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) and [DiffSeg](https://github.com/google/diffseg).

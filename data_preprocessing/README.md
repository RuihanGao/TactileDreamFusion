# Tactile Data Preprocessing

This directory contains the preprocessing pipeline that converts raw [GelSight Mini](https://www.gelsight.com/gelsightmini/) sensor captures into the **tactile normal maps** used by TactileDreamFusion.

## Overview

```
  Raw GelSight Data          Heightmap Patch          Seamless Texture          Normal Map
  (.npz: depth map)    -->   (256x256 PNG)     -->   (quilted PNG)      -->   (RGB PNG)
       Step 1                                           Step 2                  Step 3
  preprocess_tactile_data.py                      image_quilting.py      preprocess_tactile_data.py
```

**Step 1** extracts a clean tactile heightmap patch from the raw sensor data by removing low-frequency components (gel dome shape) and isolating the contact region.

**Step 2** uses [Image Quilting](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf) (Efros & Freeman, 2001) to synthesize a larger seamless tileable texture from the small patch.

**Step 3** converts the heightmap texture into a normal map (the format TactileDreamFusion expects).

## Requirements

- **Python**: 3.8+ (tested on 3.10.13, Linux)
- **Packages** (see `requirements.txt`):
  - `numpy >= 1.20`
  - `opencv-python >= 4.5`
  - `matplotlib >= 3.5`

### Installation

```bash
# Option 1: install into an existing environment
pip install -r requirements.txt

# Option 2: use the provided TactileDreamFusion conda environment
# (the packages above are already included)

# Option 3: create a minimal dedicated env
conda create -n tdf-preprocess python=3.10 -y
conda activate tdf-preprocess
pip install -r requirements.txt
```

No GPU is required. The scripts run on CPU in pure NumPy / OpenCV.

## Reproducibility

Image Quilting uses randomized patch selection. All scripts accept a `--iq_seed` (or `--seed`) argument to produce deterministic output. The default seed is `42`, so re-running the same command on the same input always yields byte-identical results.

## Quick Start

```bash
cd data_preprocessing

# Run the full pipeline (all 3 steps) on the included Strawberry example.
# The default quilting parameters (block_size=40, scale=3, seed=42) take ~15 min on CPU.
python preprocess_tactile_data.py \
    --step all \
    --data_dir ./example_data \
    --obj_name Strawberry \
    --iq_block_size 40 \
    --iq_scale 3 \
    --iq_seed 42

# Output will be in:
#   ./example_data/output/Strawberry/                 (Steps 1 & 2: patch + quilted texture)
#   ./example_data/output/tactile_textures/           (Step 3: normal map)

# For a quick smoke test (skip quilting, takes seconds instead of minutes):
python preprocess_tactile_data.py --step all --data_dir ./example_data --obj_name Strawberry --skip_quilting
```

The generated `Strawberry_tactile_texture_map_2_normal.png` can be copied to `data/tactile_textures/` and used directly with TactileDreamFusion.

### Expected Output (Strawberry example, default params)

After a successful run, `example_data/output/` contains:

```
Strawberry/
  Strawberry_sensor_image.png                # Raw GelSight RGB (240x320)
  Strawberry_step1_visualization.png         # Step 1 debug figure
  Strawberry_texture_sample.png              # 256x256 heightmap patch
  Strawberry_texture_sample_scale.npy        # Physical scale (scalar)
  Strawberry_texture_sample_quilted.png      # 780x780 seamless texture (Step 2)
tactile_textures/
  Strawberry_tactile_texture_map_2_normal.png       # 780x780 RGB normal map (final output)
  Strawberry_tactile_texture_map_2_displacement.png # 780x780 grayscale heightmap
  Strawberry_texture_sample_2_scale.npy             # Auto-scaled physical scale
  Strawberry_step3_visualization.png                # Step 3 debug figure
```

With the fixed seed (42), the pipeline is deterministic: the same command on the same input always produces byte-identical PNGs. Expected MD5 checksums for the Strawberry example run (`--iq_block_size 40 --iq_scale 3 --iq_seed 42`):

```
a1f0f6bc3c7c9784463d2b6e2d9eb155  Strawberry_texture_sample.png            (Step 1 patch)
dd6316a6b0faee60973267fb65d8251f  Strawberry_texture_sample_quilted.png    (Step 2 output)
87f8ad2681ee1d715882491a829e1d34  Strawberry_tactile_texture_map_2_normal.png  (Step 3 output)
```

Verify with: `md5sum example_data/output/Strawberry/*.png example_data/output/tactile_textures/*normal.png`

## Data Collection

### Hardware

We use the [GelSight Mini](https://www.gelsight.com/gelsightmini/) tactile sensor to capture surface texture. The sensor presses against an object's surface and captures the deformation of its elastomer gel pad, from which a depth map (heightmap) is reconstructed.

### Raw Data Format

Each GelSight capture is saved as a `.npz` file with the following fields:

| Key       | Shape       | Description                                    |
|-----------|-------------|------------------------------------------------|
| `dm`      | (240, 320)  | Reconstructed depth map / heightmap (mm)       |
| `image`   | (240, 320, 3) | RGB sensor image                            |
| `gx`      | (240, 320)  | x-gradient from GelSight reconstruction        |
| `gy`      | (240, 320)  | y-gradient from GelSight reconstruction        |
| `nz_mask` | (240, 320)  | Contact/no-contact mask                        |
| `dm_zero` | (240, 320)  | Reference (no-contact) depth map (if present)  |

### Directory Structure

Organize raw data as follows:

```
example_data/
  Strawberry/
    gelsight_images/
      00000_gxgy.npz      # First capture
      00001_gxgy.npz      # Second capture (optional)
      ...
  OrangeGlove/
    gelsight_images/
      00000_gxgy.npz
      ...
```

## Step-by-Step Pipeline

### Step 1: Extract Heightmap Patch

```bash
python preprocess_tactile_data.py --step 1 \
    --data_dir ./example_data \
    --obj_name Strawberry
```

This step:

1. **Loads the raw depth map** (`dm`) from the `.npz` file.

2. **High-pass FFT filtering** removes the low-frequency dome shape caused by the gel pad's curvature and uneven indentation. The `--freq_thresh` parameter (default: 5) controls the cutoff -- lower values remove more low-frequency content.

3. **Bilateral filtering** smooths sensor noise and discontinuities at contact boundaries while preserving the actual texture edges.

4. **Contact mask extraction** identifies where the sensor actually touched the surface:
   - Thresholds the depth map to create a binary contact mask
   - Dilates the mask and computes the convex hull
   - Finds the bounding box and center-crops to a square

5. **Normalization and saving**:
   - Min-max normalizes the cropped patch to [0, 1]
   - Saves as 256x256 PNG: `{ObjectName}_texture_sample.png`
   - Saves the physical scale factor: `{ObjectName}_texture_sample_scale.npy`

**Outputs** (in `output/{ObjectName}/`):
- `{ObjectName}_texture_sample.png` -- Normalized heightmap patch
- `{ObjectName}_texture_sample_scale.npy` -- Physical scale (for Step 3)
- `{ObjectName}_sensor_image.png` -- Raw sensor image (for reference)
- `{ObjectName}_step1_visualization.png` -- Pipeline visualization

**Key Parameters**:
- `--freq_thresh` (default: 5): FFT cutoff. Try 3 for very flat objects, 7+ for noisy data.
- `--depth_threshold` (default: 0.5): Contact detection threshold in mm. Lower for light touches.
- `--invert_depth`: Use this flag if the depth map z-axis is inverted (depends on GelSight calibration setup).

### Step 2: Image Quilting

This step creates a larger seamless tileable texture from the small (256x256) patch extracted in Step 1, using the [Image Quilting](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf) algorithm (Efros & Freeman, SIGGRAPH 2001).

```bash
# Via the main pipeline
python preprocess_tactile_data.py --step 2 \
    --data_dir ./example_data \
    --obj_name Strawberry

# Or use image_quilting.py directly for more control
python image_quilting.py \
    --input ./example_data/output/Strawberry/Strawberry_texture_sample.png \
    --output ./example_data/output/Strawberry/Strawberry_texture_sample_quilted.png \
    --block_size 80 --overlap 0.5 --scale 10
```

**Input**: `{ObjectName}_texture_sample.png` from Step 1
**Output**: `{ObjectName}_texture_sample_quilted.png` -- a larger seamless heightmap (e.g., ~2560x2560 for scale=10)

The algorithm works by:
1. Extracting overlapping blocks from the input texture patch
2. Finding blocks that best match at the overlap boundaries
3. Applying minimum-cut blending to create seamless transitions
4. Tiling the output canvas row by row

**Key Parameters** (via `--iq_*` flags or `image_quilting.py` directly):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iq_block_size` | 80 | Patch size in pixels. Larger = faster, but may miss fine detail. Use ~1/3 of input size. |
| `--iq_overlap` | 0.5 | Overlap fraction of block_size. 0.5 works well for most textures. |
| `--iq_scale` | 10 | Output size multiplier. 10 produces ~2560x2560 from 256x256 input. |
| `--iq_tolerance` | 0.1 | Error tolerance for patch matching. Higher = more variety. |

**Parameter Tuning Tips**:
- For **fine/regular textures** (e.g., fabric weave): use smaller `block_size` (40-60) to capture the repeating unit.
- For **coarse/irregular textures** (e.g., avocado, cork): use larger `block_size` (80-100).
- If you see **visible block boundaries**: increase `overlap` (e.g., 0.5).
- For **quick testing**: use `--iq_scale 3` (takes ~30s instead of minutes).

**Runtime**: With default parameters (block_size=80, scale=10), expect 2-10 minutes per texture on CPU depending on input size. Smaller scale values run proportionally faster.

If you want to **skip quilting** (use the raw 256x256 patch directly), pass `--skip_quilting` with `--step all`. This is useful for quick tests but produces a much smaller normal map.

### Step 3: Heightmap to Normal Map

```bash
python preprocess_tactile_data.py --step 3 \
    --data_dir ./example_data \
    --obj_name Strawberry
```

This step:

1. **Loads the heightmap** PNG (from Step 1 or Step 2) and the physical scale factor.

2. **Scale adjustment**: Different materials have vastly different tactile intensity. A glass tumbler has very subtle texture while an avocado has deep pits. The script automatically amplifies flat textures so that normal perturbations are visible:

   | Original Scale Range | Scale-up Factor |
   |---------------------|-----------------|
   | > 20                | 1.25x           |
   | 10 -- 20            | 1.5x            |
   | 5 -- 10             | 3.0x            |
   | <= 5                | 5.0x            |

   Disable with `--no_auto_scale`.

3. **Gradient computation** using central finite differences:
   ```
   dz/dx = (z[x+1,y] - z[x-1,y]) / 2
   dz/dy = (z[x,y-1] - z[x,y+1]) / 2
   ```

4. **Normal map construction**:
   ```
   normal = normalize([-dz/dx, -dz/dy, 1.0])
   ```
   Following the OpenGL convention: x (red) = right, y (green) = up, z (blue) = outward.

5. **Saving**: Normals in [-1, 1] are mapped to [0, 255] via `(n * 0.5 + 0.5) * 255` and saved as an RGB PNG.

**Output** (in `output/tactile_textures/`):
- `{ObjectName}_tactile_texture_map_2_normal.png` -- The normal map used by TactileDreamFusion

### Final Integration

Copy the generated normal map to TactileDreamFusion's data directory:

```bash
cp output/tactile_textures/Strawberry_tactile_texture_map_2_normal.png \
   ../data/tactile_textures/
```

Then reference it in your generation script:

```bash
bash scripts/single_texture_generation.sh \
    --tactile_texture_object Strawberry \
    --mesh_path ./data/base_meshes/a_strawberry.obj
```

## Batch Processing

Process all objects at once:

```bash
# Step 1: Extract patches for all objects in data directory
python preprocess_tactile_data.py --step 1 --data_dir /path/to/gelsight_data --batch

# Step 3: Convert all patches to normal maps
python preprocess_tactile_data.py --step 3 --data_dir /path/to/gelsight_data --batch
```

## Example Data

The `example_data/` directory contains one GelSight capture for Strawberry:

```
example_data/
  Strawberry/
    gelsight_images/
      00000_gxgy.npz    # Raw GelSight Mini capture
```

Run the full pipeline on it:

```bash
python preprocess_tactile_data.py --step all --data_dir ./example_data --obj_name Strawberry
```

## Object List

The following objects are included in the full TactileDreamFusion dataset (available on [HuggingFace](https://huggingface.co/datasets/Ruihan28/TactileDreamFusion)):

| Object Name       | Description                                    |
|--------------------|-----------------------------------------------|
| Strawberry         | Strawberry surface with seeds                 |
| avocado            | Avocado skin                                  |
| cantaloupe         | Cantaloupe melon rind                         |
| GreenSweater       | Knitted sweater fabric                        |
| OrangeGlove        | Rubber glove with grip pattern                |
| PurpleGlove        | Latex glove                                   |
| CorkMat            | Cork material                                 |
| Corn               | Corn cob kernels                              |
| CuttingBoard       | Wood cutting board grain                      |
| Football           | American football leather                     |
| FootballHandle     | Football grip texture                         |
| GlassTumbler       | Etched glass                                  |
| GoldGoat           | Metal sculpture surface                       |
| MetalFrame         | Metal mesh/frame                              |
| Orange             | Orange peel                                   |
| PinkCloth          | Pink fabric weave                             |
| Potato             | Potato skin                                   |
| RedCloth1          | Red fabric variant 1                          |
| RedCloth2          | Red fabric variant 2                          |
| SpongeHard         | Hard sponge                                   |
| SpongeSoft         | Soft sponge                                   |
| StrawHat           | Woven straw                                   |
| TableTennisHandle  | Table tennis paddle handle (rubber grip)      |
| TableTennisFace    | Table tennis paddle face (rubber pips)        |
| ClothBag           | Woven cloth bag                               |
| BlackBase          | Black plastic base                            |

## Troubleshooting

**"No contact region found"**: Lower `--depth_threshold` (e.g., `0.2`) or check that the sensor was properly pressed against the surface.

**Normal map looks flat/uniform**: The physical scale may be too small. Try `--no_auto_scale` and manually check the scale value, or collect with firmer contact pressure.

**Normal map has visible seams/blocks**: This occurs when Image Quilting block size doesn't match the texture's natural period. Adjust the block size parameter in Image Quilting.

**Inverted normals**: If the depth appears inverted (bumps instead of dips), use `--invert_depth`. This depends on the GelSight calibration convention.

"""
Tactile Data Preprocessing Pipeline for TactileDreamFusion.

This script converts raw GelSight Mini sensor data into normal maps
that TactileDreamFusion uses as tactile texture input.

Pipeline overview:
    Step 1: Extract tactile heightmap patch from raw GelSight .npz data
    Step 2: Image Quilting to synthesize a seamless tileable texture
    Step 3: Convert quilted heightmap to normal map for TactileDreamFusion

Usage:
    # Run individual steps
    python preprocess_tactile_data.py --step 1 --data_dir ./example_data --obj_name Strawberry
    python preprocess_tactile_data.py --step 2 --data_dir ./example_data --obj_name Strawberry
    python preprocess_tactile_data.py --step 3 --data_dir ./example_data --obj_name Strawberry

    # Or run all steps at once
    python preprocess_tactile_data.py --step all --data_dir ./example_data --obj_name Strawberry

See README.md for detailed documentation.
"""

import argparse
import os
import os.path as osp
import glob

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def generate_gradients(height_map: np.ndarray) -> np.ndarray:
    """Compute spatial gradients of a height map using central finite differences.

    Follows the OpenGL convention: x (red) -> right, y (green) -> up, z (blue) -> outward.

    Args:
        height_map: 2D array of shape (H, W).

    Returns:
        gxy: 3D array of shape (H, W, 2) containing [dz/dx, dz/dy].
    """
    h, w = height_map.shape
    center = height_map[1 : h - 1, 1 : w - 1]
    top = height_map[0 : h - 2, 1 : w - 1]
    bot = height_map[2:h, 1 : w - 1]
    left = height_map[1 : h - 1, 0 : w - 2]
    right = height_map[1 : h - 1, 2:w]

    dzdx = (right - left) / 2.0
    dzdy = (top - bot) / 2.0

    def pad_edge(x):
        return np.pad(x, ((1, 1), (1, 1)), "edge")

    gx = pad_edge(dzdx)
    gy = pad_edge(dzdy)
    return np.stack([gx, gy], axis=-1)


def heightmap_to_normal(heightmap_arr: np.ndarray) -> np.ndarray:
    """Convert a scaled heightmap to a normal map.

    Args:
        heightmap_arr: 2D float array, physical-scale heightmap.

    Returns:
        normal_map: (H, W, 3) float array in range [-1, 1].
    """
    # Negate because surface normal convention: F(x,y,z) = f(x,y) - z = 0
    gxy = generate_gradients(-heightmap_arr)
    ones = np.ones_like(gxy[:, :, :1], dtype=np.float32)
    f = np.dstack([gxy, ones])
    normal_map = f / np.linalg.norm(f, axis=-1, keepdims=True)
    return normal_map


def normal_to_image(normal_map: np.ndarray) -> np.ndarray:
    """Convert a [-1, 1] normal map to a [0, 255] uint8 RGB image."""
    img = ((normal_map * 0.5 + 0.5) * 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Step 1: Extract tactile heightmap patch from raw GelSight data
# ---------------------------------------------------------------------------


def extract_heightmap_patch(
    npz_path: str,
    output_dir: str,
    obj_name: str,
    freq_thresh: int = 5,
    depth_threshold: float = 0.5,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 125,
    bilateral_sigma_space: float = 125,
    invert_depth: bool = False,
    visualize: bool = True,
) -> dict:
    """Extract a clean tactile heightmap patch from raw GelSight sensor data.

    The raw depth map from GelSight contains low-frequency components (dome shape
    from the gel pad, uneven indentation depth) that must be removed to isolate
    the high-frequency tactile texture.

    Processing steps:
        1. Load depth map from .npz
        2. FFT high-pass filter to remove low-frequency dome shape
        3. Bilateral filter to smooth sensor noise while preserving texture edges
        4. Create contact mask, find bounding box, center-crop to square
        5. Min-max normalize and save as PNG with scale factor

    Args:
        npz_path: Path to the raw GelSight .npz file.
        output_dir: Directory to save outputs.
        obj_name: Object name for file naming.
        freq_thresh: FFT frequency threshold for high-pass filtering.
            Lower = more aggressive filtering (removes more low-freq).
        depth_threshold: Threshold for contact mask (in mm).
        bilateral_d: Bilateral filter diameter.
        bilateral_sigma_color: Bilateral filter sigma in color space.
        bilateral_sigma_space: Bilateral filter sigma in coordinate space.
        invert_depth: Set True if the depth map z-axis is inverted.
        visualize: Whether to save visualization plots.

    Returns:
        dict with keys: 'heightmap_path', 'scale_path', 'scale_value'
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load raw data ---
    data = np.load(npz_path)
    gt_depth = data["dm"]
    if invert_depth:
        gt_depth = -gt_depth

    sensor_image = data["image"] if "image" in data.files else None
    print(f"[Step 1] Loaded {npz_path}")
    print(f"  depth map shape: {gt_depth.shape}, range: [{gt_depth.min():.3f}, {gt_depth.max():.3f}]")

    # --- Save sensor image for reference ---
    if sensor_image is not None:
        cv2.imwrite(osp.join(output_dir, f"{obj_name}_sensor_image.png"), sensor_image)

    # --- FFT high-pass filtering ---
    rows, cols = gt_depth.shape
    f = np.fft.fft2(gt_depth)
    fshift = np.fft.fftshift(f)

    crow, ccol = rows // 2, cols // 2
    low_pass = np.zeros((rows, cols), np.uint8)
    low_pass[crow - freq_thresh : crow + freq_thresh, ccol - freq_thresh : ccol + freq_thresh] = 1
    high_pass = 1 - low_pass

    high_freq = np.fft.ifftshift(fshift * high_pass)
    high_freq = np.fft.ifft2(high_freq)
    high_freq_img = np.real(high_freq) - np.min(np.real(high_freq))

    print(f"  after high-pass filter (thresh={freq_thresh}): range [{high_freq_img.min():.3f}, {high_freq_img.max():.3f}]")

    # --- Bilateral filtering ---
    high_freq_img = cv2.bilateralFilter(
        high_freq_img.astype(np.float32),
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    # --- Contact mask: find the region where sensor touches the object ---
    contact_mask = (gt_depth > depth_threshold).astype(np.uint8)
    dilate_kernel = np.ones((5, 5), np.uint8)
    contact_mask = cv2.dilate(contact_mask, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(contact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        raise ValueError("No contact region found. Check depth_threshold or data quality.")

    hull = cv2.convexHull(contours[0])
    hull_mask = np.zeros_like(contact_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(hull_mask)
    # Center-crop bounding box to square
    if w > h:
        x = x + (w - h) // 2
        w = h
    else:
        y = y + (h - w) // 2
        h = w

    # --- Crop heightmap to contact region ---
    gt_texture = high_freq_img[y : y + h, x : x + w]
    print(f"  cropped to contact region: {gt_texture.shape}")

    # Center-crop to square (in case of non-square contact region)
    th, tw = gt_texture.shape
    crop_size = min(th, tw)
    gt_texture = gt_texture[
        th // 2 - crop_size // 2 : th // 2 + crop_size // 2,
        tw // 2 - crop_size // 2 : tw // 2 + crop_size // 2,
    ]

    # --- Min-max normalize and save ---
    scale = gt_texture.max() - gt_texture.min()
    if scale < 1e-6:
        raise ValueError("Heightmap has near-zero range. Check data quality.")
    gt_texture_norm = (gt_texture - gt_texture.min()) / scale

    # Resize to 256x256 for consistency
    target_size = 256
    gt_texture_resized = cv2.resize(gt_texture_norm, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    # Save as 3-channel grayscale PNG
    gt_texture_rgb = np.stack([gt_texture_resized] * 3, axis=-1)
    heightmap_path = osp.join(output_dir, f"{obj_name}_texture_sample.png")
    cv2.imwrite(heightmap_path, (gt_texture_rgb * 255).astype(np.uint8))

    # Save physical scale factor
    scale_path = osp.join(output_dir, f"{obj_name}_texture_sample_scale.npy")
    np.save(scale_path, scale)

    print(f"  saved heightmap patch: {heightmap_path}")
    print(f"  saved scale factor: {scale:.4f}")

    # --- Visualization ---
    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(gt_depth, cmap="gray")
        axes[0].set_title("Raw Depth Map")
        axes[1].imshow(contact_mask, cmap="gray")
        axes[1].set_title("Contact Mask")
        axes[2].imshow(high_freq_img, cmap="gray")
        axes[2].set_title(f"High-pass Filtered (t={freq_thresh})")
        axes[3].imshow(gt_texture_resized, cmap="gray")
        axes[3].set_title("Cropped Patch")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(osp.join(output_dir, f"{obj_name}_step1_visualization.png"), dpi=150)
        plt.close()
        print(f"  saved visualization: {obj_name}_step1_visualization.png")

    return {"heightmap_path": heightmap_path, "scale_path": scale_path, "scale_value": scale}


# ---------------------------------------------------------------------------
# Step 3: Convert (quilted) heightmap to normal map
# ---------------------------------------------------------------------------

# Scale-up factors to normalize tactile intensity across different materials.
# Objects with subtle texture need amplification to produce visible normal perturbations.
SCALE_UP_THRESHOLDS = [
    (20, 1.25),   # scale > 20: gentle boost
    (10, 1.5),    # 10 < scale <= 20
    (5, 3.0),     # 5 < scale <= 10
    (0, 5.0),     # scale <= 5: strong boost
]


def get_scale_up_factor(original_scale: float) -> float:
    """Determine scale-up factor based on original heightmap scale.

    Different materials have vastly different tactile intensity. Flat/subtle
    textures (e.g., glass) need stronger amplification to produce visible
    normal perturbations, while rough textures (e.g., avocado) need less.
    """
    for threshold, factor in SCALE_UP_THRESHOLDS:
        if original_scale > threshold:
            return factor
    return 5.0


def heightmap_to_normal_map(
    heightmap_path: str,
    scale_path: str,
    output_dir: str,
    obj_name: str,
    texture_index: int = 2,
    auto_scale: bool = True,
    max_crop_size: int = 2560,
    visualize: bool = True,
) -> dict:
    """Convert a heightmap texture to a normal map for TactileDreamFusion.

    This is the final step that produces the normal map PNG file loaded by
    TactileDreamFusion's mesh_tactile.py.

    Args:
        heightmap_path: Path to the heightmap PNG (grayscale).
        scale_path: Path to the .npy file with the physical scale factor.
        output_dir: Directory to save outputs.
        obj_name: Object name for file naming.
        texture_index: Texture variant index (used in filename).
        auto_scale: Whether to apply automatic scale-up for flat textures.
        max_crop_size: Maximum crop size in pixels.
        visualize: Whether to save visualization.

    Returns:
        dict with key 'normal_map_path' pointing to the output file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load heightmap ---
    heightmap = cv2.imread(heightmap_path, cv2.IMREAD_GRAYSCALE)
    if heightmap is None:
        raise FileNotFoundError(f"Cannot read heightmap: {heightmap_path}")

    heightmap_arr = heightmap.astype(np.float32) / 255.0
    print(f"[Step 3] Loaded heightmap: {heightmap_path}")
    print(f"  shape: {heightmap_arr.shape}, pixel range: [{heightmap.min()}, {heightmap.max()}]")

    # --- Center crop to square ---
    h, w = heightmap_arr.shape
    crop_size = min(h, w, max_crop_size)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    heightmap_cropped = heightmap_arr[start_y : start_y + crop_size, start_x : start_x + crop_size]

    # --- Apply physical scale ---
    original_scale = float(np.load(scale_path))
    if auto_scale:
        scale_factor = get_scale_up_factor(original_scale)
        effective_scale = original_scale * scale_factor
        print(f"  original scale: {original_scale:.4f}, scale-up factor: {scale_factor}x, effective: {effective_scale:.4f}")
    else:
        effective_scale = original_scale
        print(f"  scale: {effective_scale:.4f} (auto_scale disabled)")

    heightmap_scaled = heightmap_cropped * effective_scale

    # --- Compute normal map ---
    normal_map = heightmap_to_normal(heightmap_scaled)
    print(f"  normal map shape: {normal_map.shape}, range: [{normal_map.min():.4f}, {normal_map.max():.4f}]")

    # --- Save normal map as PNG (RGB, [0, 255]) ---
    normal_img = normal_to_image(normal_map)
    normal_map_name = f"{obj_name}_tactile_texture_map_{texture_index}_normal.png"
    normal_map_path = osp.join(output_dir, normal_map_name)
    # OpenCV uses BGR; our normal map is RGB
    cv2.imwrite(normal_map_path, cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
    print(f"  saved normal map: {normal_map_path}")

    # --- Also save the displacement map (cropped heightmap) ---
    displacement_name = f"{obj_name}_tactile_texture_map_{texture_index}_displacement.png"
    displacement_path = osp.join(output_dir, displacement_name)
    cv2.imwrite(displacement_path, (heightmap_cropped * 255).astype(np.uint8))

    # --- Save adjusted scale ---
    adjusted_scale_path = osp.join(output_dir, f"{obj_name}_texture_sample_{texture_index}_scale.npy")
    np.save(adjusted_scale_path, effective_scale)

    # --- Visualization ---
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(heightmap_cropped, cmap="gray")
        axes[0].set_title("Heightmap (cropped)")
        axes[1].imshow(heightmap_scaled, cmap="gray")
        axes[1].set_title(f"Scaled (x{effective_scale:.1f})")
        axes[2].imshow(normal_img)
        axes[2].set_title("Normal Map")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(osp.join(output_dir, f"{obj_name}_step3_visualization.png"), dpi=150)
        plt.close()
        print(f"  saved visualization: {obj_name}_step3_visualization.png")

    return {"normal_map_path": normal_map_path}


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def batch_step1(data_dir: str, output_dir: str, **kwargs):
    """Run Step 1 for all objects found under data_dir/{ObjName}/gelsight_images/."""
    results = {}
    for obj_dir in sorted(glob.glob(osp.join(data_dir, "*/gelsight_images"))):
        obj_name = osp.basename(osp.dirname(obj_dir))
        npz_files = sorted(glob.glob(osp.join(obj_dir, "*_gxgy.npz")))
        if not npz_files:
            continue
        # Use the first capture by default
        npz_path = npz_files[0]
        print(f"\n{'='*60}")
        print(f"Processing: {obj_name} ({osp.basename(npz_path)})")
        print(f"{'='*60}")
        try:
            result = extract_heightmap_patch(
                npz_path=npz_path,
                output_dir=osp.join(output_dir, obj_name),
                obj_name=obj_name,
                **kwargs,
            )
            results[obj_name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
    return results


def batch_step3(input_dir: str, output_dir: str, **kwargs):
    """Run Step 3 for all heightmap patches found under input_dir."""
    results = {}
    for scale_file in sorted(glob.glob(osp.join(input_dir, "**/*_texture_sample_scale.npy"), recursive=True)):
        obj_name = osp.basename(scale_file).replace("_texture_sample_scale.npy", "")
        patch_dir = osp.dirname(scale_file)
        heightmap_path = osp.join(patch_dir, f"{obj_name}_texture_sample.png")
        if not osp.exists(heightmap_path):
            # Also check for quilted heightmap
            heightmap_path = osp.join(patch_dir, f"{obj_name}_tactile_texture_map.png")
        if not osp.exists(heightmap_path):
            print(f"  Skipping {obj_name}: no heightmap found")
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {obj_name}")
        print(f"{'='*60}")
        try:
            result = heightmap_to_normal_map(
                heightmap_path=heightmap_path,
                scale_path=scale_file,
                output_dir=output_dir,
                obj_name=obj_name,
                **kwargs,
            )
            results[obj_name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tactile Data Preprocessing Pipeline for TactileDreamFusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Extract heightmap patch from raw GelSight data
  python preprocess_tactile_data.py --step 1 --data_dir ./example_data --obj_name Strawberry

  # Step 2: Image Quilting (synthesize seamless texture from patch)
  python preprocess_tactile_data.py --step 2 --data_dir ./example_data --obj_name Strawberry

  # Step 3: Convert heightmap to normal map
  python preprocess_tactile_data.py --step 3 --data_dir ./example_data --obj_name Strawberry

  # Full pipeline (all 3 steps)
  python preprocess_tactile_data.py --step all --data_dir ./example_data --obj_name Strawberry

  # Quick test (skip quilting, use patch directly)
  python preprocess_tactile_data.py --step all --data_dir ./example_data --obj_name Strawberry --skip_quilting

  # Batch process all objects in a directory
  python preprocess_tactile_data.py --step 1 --data_dir /path/to/gelsight_data --batch
        """,
    )

    parser.add_argument(
        "--step",
        required=True,
        choices=["1", "2", "3", "all"],
        help="Pipeline step: 1 (extract patch), 2 (image quilting), 3 (heightmap->normal), all (full pipeline)",
    )
    parser.add_argument("--data_dir", required=True, help="Root directory containing raw GelSight data")
    parser.add_argument("--obj_name", type=str, default=None, help="Object name to process (omit for --batch)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: data_dir/output)")
    parser.add_argument("--batch", action="store_true", help="Process all objects found in data_dir")
    parser.add_argument("--sample_index", type=int, default=0, help="GelSight capture index to use (default: 0)")
    parser.add_argument("--texture_index", type=int, default=2, help="Texture variant index for output naming (default: 2)")
    parser.add_argument("--no_auto_scale", action="store_true", help="Disable automatic scale-up for flat textures")
    parser.add_argument("--no_visualize", action="store_true", help="Disable saving visualization plots")

    # Step 1 parameters
    parser.add_argument("--freq_thresh", type=int, default=5, help="FFT frequency threshold for high-pass filter (default: 5)")
    parser.add_argument("--depth_threshold", type=float, default=0.5, help="Depth threshold for contact mask in mm (default: 0.5)")
    parser.add_argument("--invert_depth", action="store_true", help="Invert depth map z-axis (for some GelSight setups)")

    # Step 2 parameters (Image Quilting)
    parser.add_argument("--iq_block_size", type=int, default=80, help="Image Quilting block size in pixels (default: 80)")
    parser.add_argument("--iq_overlap", type=float, default=0.5, help="Image Quilting overlap fraction (default: 0.5)")
    parser.add_argument("--iq_scale", type=float, default=10, help="Image Quilting output scale factor (default: 10)")
    parser.add_argument("--iq_tolerance", type=float, default=0.1, help="Image Quilting error tolerance (default: 0.1)")
    parser.add_argument("--iq_seed", type=int, default=42, help="Image Quilting random seed for reproducibility (default: 42)")
    parser.add_argument("--skip_quilting", action="store_true", help="Skip Step 2 in --step all (use patch directly)")

    args = parser.parse_args()

    output_dir = args.output_dir or osp.join(args.data_dir, "output")
    visualize = not args.no_visualize

    if args.step in ("1", "all"):
        print("\n" + "=" * 60)
        print("STEP 1: Extract Heightmap Patch from GelSight Data")
        print("=" * 60)

        if args.batch:
            batch_step1(
                data_dir=args.data_dir,
                output_dir=output_dir,
                freq_thresh=args.freq_thresh,
                depth_threshold=args.depth_threshold,
                invert_depth=args.invert_depth,
                visualize=visualize,
            )
        else:
            if args.obj_name is None:
                parser.error("--obj_name is required when not using --batch")

            # Find the .npz file
            gelsight_dir = osp.join(args.data_dir, args.obj_name, "gelsight_images")
            if not osp.isdir(gelsight_dir):
                gelsight_dir = osp.join(args.data_dir, args.obj_name)
            npz_path = osp.join(gelsight_dir, f"{args.sample_index:05d}_gxgy.npz")
            if not osp.exists(npz_path):
                raise FileNotFoundError(f"GelSight data not found: {npz_path}")

            step1_out = osp.join(output_dir, args.obj_name)
            extract_heightmap_patch(
                npz_path=npz_path,
                output_dir=step1_out,
                obj_name=args.obj_name,
                freq_thresh=args.freq_thresh,
                depth_threshold=args.depth_threshold,
                invert_depth=args.invert_depth,
                visualize=visualize,
            )

    if args.step in ("2", "all"):
        print("\n" + "=" * 60)
        print("STEP 2: Image Quilting (Seamless Texture Synthesis)")
        print("=" * 60)

        if args.step == "all" and args.skip_quilting:
            print("  SKIPPED (--skip_quilting). Using patch directly for Step 3.")
        else:
            from image_quilting import quilting

            if args.batch:
                import glob as _glob
                patches = sorted(_glob.glob(osp.join(output_dir, "*/*_texture_sample.png")))
                for patch_path in patches:
                    _obj = osp.basename(osp.dirname(patch_path))
                    quilted_path = osp.join(osp.dirname(patch_path), f"{_obj}_texture_sample_quilted.png")
                    print(f"\n{'='*60}")
                    print(f"Quilting: {_obj}")
                    print(f"{'='*60}")
                    try:
                        quilting(
                            patch_path, quilted_path,
                            block_size=args.iq_block_size,
                            overlap=args.iq_overlap,
                            scale=args.iq_scale,
                            tolerance=args.iq_tolerance,
                            seed=args.iq_seed,
                        )
                    except Exception as e:
                        print(f"  ERROR: {e}")
            else:
                if args.obj_name is None:
                    parser.error("--obj_name is required when not using --batch")
                step1_out = osp.join(output_dir, args.obj_name)
                patch_path = osp.join(step1_out, f"{args.obj_name}_texture_sample.png")
                if not osp.exists(patch_path):
                    raise FileNotFoundError(f"Patch not found: {patch_path}\nRun Step 1 first.")
                quilted_path = osp.join(step1_out, f"{args.obj_name}_texture_sample_quilted.png")
                quilting(
                    patch_path, quilted_path,
                    block_size=args.iq_block_size,
                    overlap=args.iq_overlap,
                    scale=args.iq_scale,
                    tolerance=args.iq_tolerance,
                    seed=args.iq_seed,
                )

    if args.step in ("3", "all"):
        print("\n" + "=" * 60)
        print("STEP 3: Convert Heightmap to Normal Map")
        print("=" * 60)

        if args.batch:
            # In batch mode for step 3, scan output_dir for patches
            search_dir = output_dir if args.step == "all" else args.data_dir
            batch_step3(
                input_dir=search_dir,
                output_dir=osp.join(output_dir, "tactile_textures"),
                texture_index=args.texture_index,
                auto_scale=not args.no_auto_scale,
                visualize=visualize,
            )
        else:
            if args.obj_name is None:
                parser.error("--obj_name is required when not using --batch")

            step1_out = osp.join(output_dir, args.obj_name)
            scale_path = osp.join(step1_out, f"{args.obj_name}_texture_sample_scale.npy")

            # Prefer quilted heightmap if available, fall back to raw patch
            quilted_path = osp.join(step1_out, f"{args.obj_name}_texture_sample_quilted.png")
            raw_path = osp.join(step1_out, f"{args.obj_name}_texture_sample.png")
            if osp.exists(quilted_path) and not (args.step == "all" and args.skip_quilting):
                heightmap_path = quilted_path
                print(f"  Using quilted heightmap: {quilted_path}")
            elif osp.exists(raw_path):
                heightmap_path = raw_path
                print(f"  Using raw patch (no quilted version found): {raw_path}")
            else:
                raise FileNotFoundError(
                    f"Heightmap not found in: {step1_out}\nRun Step 1 first."
                )

            tactile_textures_dir = osp.join(output_dir, "tactile_textures")
            heightmap_to_normal_map(
                heightmap_path=heightmap_path,
                scale_path=scale_path,
                output_dir=tactile_textures_dir,
                obj_name=args.obj_name,
                texture_index=args.texture_index,
                auto_scale=not args.no_auto_scale,
                visualize=visualize,
            )

    print("\n" + "=" * 60)
    print("Done!")
    if args.step in ("3", "all"):
        tactile_textures_dir = osp.join(output_dir, "tactile_textures")
        print(f"\nTo use with TactileDreamFusion, copy the normal map(s) from:")
        print(f"  {tactile_textures_dir}/")
        print(f"to:")
        print(f"  TactileDreamFusion/data/tactile_textures/")
    print("=" * 60)


if __name__ == "__main__":
    main()

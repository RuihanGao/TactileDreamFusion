"""
Image Quilting for Texture Synthesis (Step 2 of the preprocessing pipeline).

Synthesizes a larger seamless tileable texture from a small tactile heightmap
patch using the Image Quilting algorithm (Efros & Freeman, SIGGRAPH 2001).

This is a self-contained implementation adapted from:
    https://github.com/rohitrango/Image-Quilting

Usage:
    # Basic usage
    python image_quilting.py --input patch.png --output quilted.png

    # With custom parameters
    python image_quilting.py --input patch.png --output quilted.png \
        --block_size 80 --overlap 0.5 --scale 10

    # Integrated with the preprocessing pipeline (Step 1 output -> Step 2)
    python image_quilting.py \
        --input example_data/output/Strawberry/Strawberry_texture_sample.png \
        --output example_data/output/Strawberry/Strawberry_quilted.png \
        --block_size 80 --overlap 0.5 --scale 10

See README.md for parameter tuning guidance.
"""

import argparse
import os
import time
from itertools import product
from math import ceil

import cv2
import numpy as np

inf = float("inf")


# ---------------------------------------------------------------------------
# Core Image Quilting Algorithm
# ---------------------------------------------------------------------------


def find_patch_horizontal(ref_block, texture, blocksize, overlap, tolerance):
    """Find a patch from texture that best matches ref_block's right edge horizontally."""
    H, W = texture.shape[:2]
    err_mat = np.full((H - blocksize, W - blocksize), inf)
    for i, j in product(range(H - blocksize), range(W - blocksize)):
        rms = ((texture[i : i + blocksize, j : j + overlap] - ref_block[:, -overlap:]) ** 2).mean()
        if rms > 0:
            err_mat[i, j] = rms

    min_val = np.min(err_mat)
    y, x = np.where(err_mat < (1.0 + tolerance) * min_val)
    c = np.random.randint(len(y))
    return texture[y[c] : y[c] + blocksize, x[c] : x[c] + blocksize]


def find_patch_vertical(ref_block, texture, blocksize, overlap, tolerance):
    """Find a patch from texture that best matches ref_block's bottom edge vertically."""
    H, W = texture.shape[:2]
    err_mat = np.full((H - blocksize, W - blocksize), inf)
    for i, j in product(range(H - blocksize), range(W - blocksize)):
        rms = ((texture[i : i + overlap, j : j + blocksize] - ref_block[-overlap:, :]) ** 2).mean()
        if rms > 0:
            err_mat[i, j] = rms

    min_val = np.min(err_mat)
    y, x = np.where(err_mat < (1.0 + tolerance) * min_val)
    c = np.random.randint(len(y))
    return texture[y[c] : y[c] + blocksize, x[c] : x[c] + blocksize]


def find_patch_both(ref_block_left, ref_block_top, texture, blocksize, overlap, tolerance):
    """Find a patch that matches both left and top neighbors."""
    H, W = texture.shape[:2]
    err_mat = np.full((H - blocksize, W - blocksize), inf)
    for i, j in product(range(H - blocksize), range(W - blocksize)):
        rms_top = ((texture[i : i + overlap, j : j + blocksize] - ref_block_top[-overlap:, :]) ** 2).mean()
        rms_left = ((texture[i : i + blocksize, j : j + overlap] - ref_block_left[:, -overlap:]) ** 2).mean()
        rms = rms_top + rms_left
        if rms > 0:
            err_mat[i, j] = rms

    min_val = np.min(err_mat)
    y, x = np.where(err_mat < (1.0 + tolerance) * min_val)
    c = np.random.randint(len(y))
    return texture[y[c] : y[c] + blocksize, x[c] : x[c] + blocksize]


def _min_cut_path(err):
    """Compute minimum-cost vertical path through an error surface using dynamic programming.

    Args:
        err: 2D array of shape (H, W) with per-pixel overlap error.

    Returns:
        path: list of column indices, one per row.
    """
    min_index = []
    E = [list(err[0])]
    for i in range(1, err.shape[0]):
        e = [inf] + E[-1] + [inf]
        e = np.array([e[:-2], e[1:-1], e[2:]])
        min_arr = e.min(0)
        min_arg = e.argmin(0) - 1
        min_index.append(min_arg)
        E.append(list(err[i] + min_arr))

    # Backtrack
    path = []
    col = np.argmin(E[-1])
    path.append(col)
    for idx in min_index[::-1]:
        col = col + idx[col]
        path.append(col)
    return path[::-1]


def min_cut_horizontal(block1, block2, blocksize, overlap):
    """Blend block1 (left context) and block2 (new patch) with a minimum-cut seam."""
    err = ((block1[:, -overlap:] - block2[:, :overlap]) ** 2).mean(2)
    path = _min_cut_path(err)

    mask = np.zeros((blocksize, blocksize, block1.shape[2]))
    for i, p in enumerate(path):
        mask[i, : p + 1] = 1

    res = np.zeros(block1.shape)
    res[:, :overlap] = block1[:, -overlap:]
    res = res * mask + block2 * (1 - mask)
    return res


def min_cut_vertical(block1, block2, blocksize, overlap):
    """Blend block1 (top context) and block2 (new patch) with a minimum-cut seam."""
    res = min_cut_horizontal(np.rot90(block1), np.rot90(block2), blocksize, overlap)
    return np.rot90(res, 3)


def min_cut_both(ref_block_left, ref_block_top, patch_block, blocksize, overlap):
    """Blend with minimum-cut seams for both left and top boundaries."""
    # Horizontal (left) seam
    err = ((ref_block_left[:, -overlap:] - patch_block[:, :overlap]) ** 2).mean(2)
    path = _min_cut_path(err)
    mask1 = np.zeros((blocksize, blocksize, patch_block.shape[2]))
    for i, p in enumerate(path):
        mask1[i, : p + 1] = 1

    # Vertical (top) seam
    err = ((np.rot90(ref_block_top)[:, -overlap:] - np.rot90(patch_block)[:, :overlap]) ** 2).mean(2)
    path = _min_cut_path(err)
    mask2 = np.zeros((blocksize, blocksize, patch_block.shape[2]))
    for i, p in enumerate(path):
        mask2[i, : p + 1] = 1
    mask2 = np.rot90(mask2, 3)

    # Corner priority: left mask takes precedence
    mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)

    res = np.zeros(patch_block.shape)
    res[:, :overlap] = mask1[:, :overlap] * ref_block_left[:, -overlap:]
    res[:overlap, :] = res[:overlap, :] + mask2[:overlap, :] * ref_block_top[-overlap:, :]
    res = res + (1 - np.maximum(mask1, mask2)) * patch_block
    return res


def generate_texture_map(image, blocksize, overlap, out_h, out_w, tolerance=0.1):
    """Synthesize a texture map using image quilting.

    Args:
        image: Source texture, shape (H, W, C), normalized to [0, 1].
        blocksize: Patch size in pixels.
        overlap: Overlap in pixels.
        out_h: Desired output height.
        out_w: Desired output width.
        tolerance: Error tolerance fraction for patch selection.

    Returns:
        texture_map: Synthesized texture, shape (H', W', C), normalized to [0, 1].
    """
    n_h = int(ceil((out_h - blocksize) / (blocksize - overlap)))
    n_w = int(ceil((out_w - blocksize) / (blocksize - overlap)))

    tex_h = blocksize + n_h * (blocksize - overlap)
    tex_w = blocksize + n_w * (blocksize - overlap)
    texture_map = np.zeros((tex_h, tex_w, image.shape[2]))

    H, W = image.shape[:2]

    # Place a random starting block
    ri, rj = np.random.randint(H - blocksize), np.random.randint(W - blocksize)
    texture_map[:blocksize, :blocksize, :] = image[ri : ri + blocksize, rj : rj + blocksize]

    # Fill first row (horizontal constraints only)
    for blk_j in range(blocksize - overlap, tex_w - overlap, blocksize - overlap):
        ref = texture_map[:blocksize, blk_j - blocksize + overlap : blk_j + overlap]
        patch = find_patch_horizontal(ref, image, blocksize, overlap, tolerance)
        texture_map[:blocksize, blk_j : blk_j + blocksize] = min_cut_horizontal(ref, patch, blocksize, overlap)

    # Fill first column (vertical constraints only)
    for blk_i in range(blocksize - overlap, tex_h - overlap, blocksize - overlap):
        ref = texture_map[blk_i - blocksize + overlap : blk_i + overlap, :blocksize]
        patch = find_patch_vertical(ref, image, blocksize, overlap, tolerance)
        texture_map[blk_i : blk_i + blocksize, :blocksize] = min_cut_vertical(ref, patch, blocksize, overlap)

    # Fill remaining blocks (both constraints)
    for i in range(1, n_h + 1):
        for j in range(1, n_w + 1):
            bi = i * (blocksize - overlap)
            bj = j * (blocksize - overlap)
            ref_left = texture_map[bi : bi + blocksize, bj - blocksize + overlap : bj + overlap]
            ref_top = texture_map[bi - blocksize + overlap : bi + overlap, bj : bj + blocksize]
            patch = find_patch_both(ref_left, ref_top, image, blocksize, overlap, tolerance)
            texture_map[bi : bi + blocksize, bj : bj + blocksize] = min_cut_both(
                ref_left, ref_top, patch, blocksize, overlap
            )
        print(f"  Row {i + 1}/{n_h + 1} complete")

    return texture_map


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def quilting(
    input_path,
    output_path,
    block_size=80,
    overlap=0.5,
    scale=10,
    tolerance=0.1,
    seed=None,
):
    """Run image quilting on a single input image.

    Args:
        input_path: Path to input texture patch (PNG).
        output_path: Path to save the quilted output (PNG).
        block_size: Patch size in pixels.
        overlap: Overlap as fraction of block_size (0.0 - 1.0).
        scale: Output size multiplier relative to input.
        tolerance: Error tolerance for patch selection.
        seed: Random seed for reproducibility.

    Returns:
        output_path: Path to the saved output.
    """
    if seed is not None:
        np.random.seed(seed)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    H, W = image.shape[:2]
    out_h, out_w = int(scale * H), int(scale * W)
    overlap_px = int(overlap * block_size)

    print(f"[Image Quilting] Input: {input_path} ({H}x{W})")
    print(f"  block_size={block_size}, overlap={overlap} ({overlap_px}px), scale={scale}")
    print(f"  Output target size: {out_h}x{out_w}")

    start = time.time()
    texture_map = generate_texture_map(image, block_size, overlap_px, out_h, out_w, tolerance)
    elapsed = time.time() - start
    print(f"  Synthesis completed in {elapsed:.1f}s, output shape: {texture_map.shape[:2]}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_img = (255 * np.clip(texture_map, 0, 1)).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out_img)
    print(f"  Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Image Quilting for Texture Synthesis (Step 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default parameters (block_size=80, overlap=0.5, scale=10)
  python image_quilting.py --input patch.png --output quilted.png

  # Smaller output for quick testing
  python image_quilting.py --input patch.png --output quilted.png --scale 4

  # Fine-grained texture (use smaller blocks)
  python image_quilting.py --input patch.png --output quilted.png --block_size 40

  # Batch: process all patches in a directory
  python image_quilting.py --input_dir ./patches/ --output_dir ./quilted/

Parameter tuning guide:
  block_size  Larger = faster but may miss fine detail. Start with 80.
  overlap     0.5 (50%%) works well for most textures. Lower = faster but more seams.
  scale       10 produces ~2560x2560 from 256x256 input. Use 4 for quick tests.
  tolerance   0.1 (10%%) adds variety. Lower = more uniform but slower.
        """,
    )
    parser.add_argument("-i", "--input", type=str, help="Path to input texture patch")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save output (default: {input}_quilted.png)")
    parser.add_argument("--input_dir", type=str, help="Directory of input patches (batch mode)")
    parser.add_argument("--output_dir", type=str, help="Directory for output (batch mode)")
    parser.add_argument("-b", "--block_size", type=int, default=80, help="Block size in pixels (default: 80)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap as fraction of block_size (default: 0.5)")
    parser.add_argument("-s", "--scale", type=float, default=10, help="Output scale factor (default: 10)")
    parser.add_argument("-t", "--tolerance", type=float, default=0.1, help="Error tolerance fraction (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if args.input_dir:
        # Batch mode
        output_dir = args.output_dir or os.path.join(args.input_dir, "quilted")
        os.makedirs(output_dir, exist_ok=True)
        import glob

        patches = sorted(glob.glob(os.path.join(args.input_dir, "*_texture_sample.png")))
        if not patches:
            patches = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
        print(f"Found {len(patches)} patches to process")

        for patch_path in patches:
            name = os.path.splitext(os.path.basename(patch_path))[0]
            output_path = os.path.join(output_dir, f"{name}_quilted.png")
            try:
                quilting(
                    patch_path,
                    output_path,
                    block_size=args.block_size,
                    overlap=args.overlap,
                    scale=args.scale,
                    tolerance=args.tolerance,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"  ERROR processing {name}: {e}")
    else:
        # Single file mode
        if not args.input:
            parser.error("Either --input or --input_dir is required")
        output = args.output
        if output is None:
            base = os.path.splitext(args.input)[0]
            output = f"{base}_quilted.png"
        quilting(
            args.input,
            output,
            block_size=args.block_size,
            overlap=args.overlap,
            scale=args.scale,
            tolerance=args.tolerance,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

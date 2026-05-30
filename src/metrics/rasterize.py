"""Gaze, bbox, and attention rasterization at the model's native grid
and at a common cross-model grid.

Implements the choices frozen in docs/extraction-spec.md §Q3:
  - rasterize_bbox_to_grid:    list of bboxes → (G, G) binary mask /
                               prob map. Used by BBoxProbe v1.
  - rasterize_gaze_to_grid:    fixations → (G, G) prob map via Gaussian
                               KDE. Kept for the GazeProbe v2 lineage
                               (git tag gaze-v0.1) and for any future
                               REFLACX work if PhysioNet access lands.
  - reshape_attention_to_grid: 1D attention over image tokens → (G, G)
  - upsample_to_common_grid:   any (Gx, Gy) → (56, 56) for cross-model
                               comparison via bilinear interpolation
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- #
# Bbox rasterization (BBoxProbe v1 primary path)
# --------------------------------------------------------------------- #

def rasterize_bbox_to_grid(
    bboxes: list[tuple[float, float, float, float]] | np.ndarray | pd.DataFrame,
    image_hw: tuple[int, int],
    grid_edge: int,
    *,
    as_probability: bool = True,
) -> np.ndarray:
    """Rasterize one or more bboxes (in original-image pixel coords)
    onto a (grid_edge, grid_edge) mask.

    Each grid cell whose CENTER lies inside the union of bboxes is set
    to 1, else 0. Multi-rater bboxes are handled by passing the union
    of all radiologists' annotations (see docs/extraction-spec.md §Q3
    "multi-rater bbox handling").

    Args:
        bboxes: Iterable of (x_min, y_min, x_max, y_max) tuples in
            pixel coordinates of the original image, OR a numpy array
            of shape (N, 4), OR a DataFrame with columns named
            x_min/y_min/x_max/y_max (case-insensitive).
        image_hw: (H, W) of the original image, in pixels.
        grid_edge: Output grid edge length (e.g. 16 for MedGemma native).
        as_probability: If True (default), normalize the mask sum to 1
            (probability distribution suitable for the KL metric). If
            False, return a {0,1} binary mask (suitable for IoU /
            Pointing-Game).

    Returns:
        np.ndarray of shape (grid_edge, grid_edge), dtype float32.
        If as_probability=True and all bboxes are empty / outside the
        image, the output is uniform (no-information signal) rather
        than all-zeros, so KL is still well-defined.

    Raises:
        ValueError: if any bbox is malformed (x_max <= x_min, etc.)
            when at least one bbox is provided.
    """
    rects = _normalize_bbox_input(bboxes)

    H, W = image_hw
    if grid_edge <= 0:
        raise ValueError(f"grid_edge must be positive, got {grid_edge}")
    if H <= 0 or W <= 0:
        raise ValueError(f"image_hw must be positive, got {image_hw}")

    # Grid-cell centers in pixel coordinates
    cell_h = H / grid_edge
    cell_w = W / grid_edge
    cy = (np.arange(grid_edge) + 0.5) * cell_h            # (G,)
    cx = (np.arange(grid_edge) + 0.5) * cell_w            # (G,)
    mesh_y, mesh_x = np.meshgrid(cy, cx, indexing="ij")    # (G, G)

    mask = np.zeros((grid_edge, grid_edge), dtype=bool)
    for x_min, y_min, x_max, y_max in rects:
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"Malformed bbox ({x_min}, {y_min}, {x_max}, {y_max}): "
                "x_max must exceed x_min and y_max must exceed y_min."
            )
        inside = (
            (mesh_x >= x_min) & (mesh_x < x_max)
            & (mesh_y >= y_min) & (mesh_y < y_max)
        )
        mask |= inside

    if not as_probability:
        return mask.astype(np.float32)

    # Probability distribution. If empty, return uniform.
    out = mask.astype(np.float32)
    if out.sum() == 0:
        return np.full((grid_edge, grid_edge), 1.0 / (grid_edge * grid_edge), dtype=np.float32)
    return out / out.sum()


def _normalize_bbox_input(bboxes) -> list[tuple[float, float, float, float]]:
    """Coerce bbox input to a list of 4-tuples regardless of input format."""
    if isinstance(bboxes, pd.DataFrame):
        if len(bboxes) == 0:
            return []
        lc = {c.lower(): c for c in bboxes.columns}
        try:
            cols = (lc["x_min"], lc["y_min"], lc["x_max"], lc["y_max"])
        except KeyError as e:
            raise ValueError(
                f"DataFrame missing bbox column {e}; have {list(bboxes.columns)}"
            ) from None
        return [tuple(row) for row in bboxes[list(cols)].itertuples(index=False, name=None)]

    if isinstance(bboxes, np.ndarray):
        if bboxes.size == 0:
            return []
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError(f"numpy bbox input must be (N, 4), got shape {bboxes.shape}")
        return [tuple(row) for row in bboxes.tolist()]

    return [tuple(b) for b in bboxes]


# --------------------------------------------------------------------- #
# Gaze rasterization (GazeProbe v2 path, kept for future)
# --------------------------------------------------------------------- #

def rasterize_gaze_to_grid(
    fixations: pd.DataFrame,
    image_hw: tuple[int, int],
    grid_edge: int,
    sigma_frac: float = 0.03,
    *,
    x_col: str | None = None,
    y_col: str | None = None,
    duration_col: str | None = None,
) -> np.ndarray:
    """Gaussian KDE rasterization of fixation points to a (G, G) prob map.

    Procedure:
      1. Lay each fixation as a delta at (y, x) on a high-resolution
         image-sized buffer, weighted by duration if available.
      2. Convolve with a Gaussian of σ = `sigma_frac` × image diagonal.
      3. Mean-pool to (G, G).
      4. Normalize to a probability distribution.

    Args:
        fixations: DataFrame with at minimum x and y position columns.
            Column names are auto-detected if not provided (any column
            starting with 'x' or 'y' is taken as a position; any column
            with 'duration' or 'dwell' in the name is taken as weight).
        image_hw: (H, W) of the source image, in pixels.
        grid_edge: Output grid edge length (e.g. 16 for MedGemma native).
        sigma_frac: KDE bandwidth as a fraction of the image diagonal.
            0.03 corresponds to roughly 1° of visual angle at typical
            reading distance — refine once the REFLACX reference
            protocol's recommended σ is confirmed.

    Returns:
        np.ndarray of shape (G, G), dtype float32, sum == 1.

    Raises:
        ValueError: if no fixations, or if column detection fails.
    """
    if len(fixations) == 0:
        raise ValueError("rasterize_gaze_to_grid: empty fixations DataFrame")

    from scipy.ndimage import gaussian_filter

    H, W = image_hw

    def _detect(prefix: str) -> str:
        return next(c for c in fixations.columns if c.lower().startswith(prefix))

    xc = x_col or _detect("x")
    yc = y_col or _detect("y")

    # Duration column (optional)
    dur_col = duration_col
    if dur_col is None:
        for c in fixations.columns:
            cl = c.lower()
            if "duration" in cl or "dwell" in cl:
                dur_col = c
                break

    # Lay deltas
    hi = np.zeros((H, W), dtype=np.float32)
    xs = fixations[xc].clip(0, W - 1).astype(int).to_numpy()
    ys = fixations[yc].clip(0, H - 1).astype(int).to_numpy()
    if dur_col is not None:
        w = fixations[dur_col].to_numpy().astype(np.float32)
    else:
        w = np.ones_like(xs, dtype=np.float32)
    np.add.at(hi, (ys, xs), w)

    # Convolve
    sigma_px = sigma_frac * float(np.hypot(H, W))
    smoothed = gaussian_filter(hi, sigma=sigma_px)

    # Mean-pool to grid
    bin_h = H // grid_edge
    bin_w = W // grid_edge
    smoothed = smoothed[: bin_h * grid_edge, : bin_w * grid_edge]
    grid = smoothed.reshape(grid_edge, bin_h, grid_edge, bin_w).mean(axis=(1, 3))

    grid = grid + 1e-12
    grid = grid / grid.sum()
    return grid.astype(np.float32)


# --------------------------------------------------------------------- #
# Attention reshape
# --------------------------------------------------------------------- #

def reshape_attention_to_grid(
    attn_1d: np.ndarray,
    native_hw: tuple[int, int],
) -> np.ndarray:
    """Reshape a 1D attention vector over image tokens into (H, W).

    Normalizes to a probability distribution. Assumes tokens are
    laid out in row-major order over the patch grid, which is the
    convention for ViT-family encoders.
    """
    H, W = native_hw
    if attn_1d.size != H * W:
        raise ValueError(
            f"reshape_attention_to_grid: got {attn_1d.size} tokens, "
            f"expected {H}*{W}={H*W}"
        )
    grid = attn_1d.reshape(H, W).astype(np.float32)
    grid = grid + 1e-12
    grid = grid / grid.sum()
    return grid


# --------------------------------------------------------------------- #
# Common-grid upsampling
# --------------------------------------------------------------------- #

def upsample_to_common_grid(
    grid_2d: np.ndarray,
    target_hw: tuple[int, int] = (56, 56),
) -> np.ndarray:
    """Bilinear upsample/downsample to a common (H, W) grid for
    cross-model comparison. Re-normalizes to a probability distribution.
    """
    # Use scipy.ndimage.zoom for bilinear-like behavior, dependency-free
    # vs. pulling in PIL/cv2 for two-line resize.
    from scipy.ndimage import zoom

    H_in, W_in = grid_2d.shape
    H_out, W_out = target_hw
    zoom_h = H_out / H_in
    zoom_w = W_out / W_in
    out = zoom(grid_2d, (zoom_h, zoom_w), order=1, mode="nearest")
    # zoom can produce slightly off-by-one shapes; trim/pad to exact size
    out = out[:H_out, :W_out]
    if out.shape != (H_out, W_out):
        # Right-pad with edge values if zoom undershot
        pad_h = H_out - out.shape[0]
        pad_w = W_out - out.shape[1]
        out = np.pad(out, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")
        out = out[:H_out, :W_out]
    out = out.astype(np.float32) + 1e-12
    return out / out.sum()

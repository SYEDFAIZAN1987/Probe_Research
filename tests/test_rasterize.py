"""Unit tests for src/metrics/rasterize.py.

Verifies that the three rasterization helpers produce normalized
probability distributions of the right shape, peak at the right place
for synthetic inputs, and raise on bad inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.rasterize import (
    rasterize_gaze_to_grid,
    reshape_attention_to_grid,
    upsample_to_common_grid,
)


# --------------------------------------------------------------------- #
# rasterize_gaze_to_grid
# --------------------------------------------------------------------- #

class TestRasterizeGaze:
    def test_output_is_probability_distribution(self):
        fix = pd.DataFrame({"x_position": [100, 200], "y_position": [100, 150]})
        grid = rasterize_gaze_to_grid(fix, image_hw=(512, 512), grid_edge=16)
        assert grid.shape == (16, 16)
        assert grid.dtype == np.float32
        assert grid.sum() == pytest.approx(1.0, rel=1e-4)
        assert (grid >= 0).all()

    def test_single_central_fixation_peaks_near_center(self):
        # Single fixation at image center → KDE-smoothed peak at center
        # of the 16x16 grid. With 512x512 image and a fixation at
        # (256, 256), the peak should be at grid cell (8, 8) or nearby.
        fix = pd.DataFrame({"x_position": [256], "y_position": [256]})
        grid = rasterize_gaze_to_grid(fix, image_hw=(512, 512), grid_edge=16)
        peak = np.unravel_index(grid.argmax(), grid.shape)
        # Center of a 16-cell grid is index 7 or 8.
        assert peak[0] in (7, 8)
        assert peak[1] in (7, 8)

    def test_two_corner_fixations_peak_in_corners(self):
        # Two fixations at opposite corners; KDE smoothing then mean-pool
        # to 8x8. Each corner cell should have higher mass than the center.
        fix = pd.DataFrame({
            "x_position": [10, 502],
            "y_position": [10, 502],
        })
        grid = rasterize_gaze_to_grid(
            fix, image_hw=(512, 512), grid_edge=8, sigma_frac=0.02,
        )
        # Top-left and bottom-right corners should exceed center.
        assert grid[0, 0] > grid[4, 4]
        assert grid[7, 7] > grid[4, 4]

    def test_duration_column_weights_fixations(self):
        # One short fixation at TL, one long fixation at BR. Result
        # should have heavier mass at BR than at TL.
        fix = pd.DataFrame({
            "x_position": [50, 450],
            "y_position": [50, 450],
            "duration": [0.05, 1.0],
        })
        grid = rasterize_gaze_to_grid(
            fix, image_hw=(512, 512), grid_edge=8, sigma_frac=0.02,
        )
        assert grid[7, 7] > grid[0, 0]

    def test_empty_fixations_raises(self):
        with pytest.raises(ValueError):
            rasterize_gaze_to_grid(
                pd.DataFrame({"x_position": [], "y_position": []}),
                image_hw=(512, 512), grid_edge=16,
            )

    def test_alternative_column_names_via_autodetect(self):
        # Column names "x", "y" — should be picked up by the prefix
        # autodetection.
        fix = pd.DataFrame({"x": [100], "y": [100]})
        grid = rasterize_gaze_to_grid(fix, image_hw=(512, 512), grid_edge=8)
        assert grid.sum() == pytest.approx(1.0, rel=1e-4)


# --------------------------------------------------------------------- #
# reshape_attention_to_grid
# --------------------------------------------------------------------- #

class TestReshapeAttention:
    def test_output_shape_and_normalization(self):
        attn = np.ones(256, dtype=np.float32)
        grid = reshape_attention_to_grid(attn, native_hw=(16, 16))
        assert grid.shape == (16, 16)
        assert grid.sum() == pytest.approx(1.0, rel=1e-5)
        # Uniform input → uniform output
        assert (grid == grid[0, 0]).all()

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError):
            reshape_attention_to_grid(
                np.ones(100, dtype=np.float32),
                native_hw=(16, 16),
            )

    def test_row_major_layout(self):
        # If we put a 1.0 at index 0 (top-left in row-major), the peak
        # should be at (0, 0) post-reshape.
        attn = np.zeros(256, dtype=np.float32)
        attn[0] = 1.0
        grid = reshape_attention_to_grid(attn, native_hw=(16, 16))
        peak = np.unravel_index(grid.argmax(), grid.shape)
        assert peak == (0, 0)


# --------------------------------------------------------------------- #
# upsample_to_common_grid
# --------------------------------------------------------------------- #

class TestUpsample:
    def test_output_is_normalized(self):
        rng = np.random.default_rng(2)
        x = rng.random((16, 16)).astype(np.float32)
        x /= x.sum()
        out = upsample_to_common_grid(x, target_hw=(56, 56))
        assert out.shape == (56, 56)
        assert out.sum() == pytest.approx(1.0, rel=1e-4)

    def test_identity_size_preserves_shape(self):
        x = np.ones((56, 56), dtype=np.float32) / (56 * 56)
        out = upsample_to_common_grid(x, target_hw=(56, 56))
        assert out.shape == (56, 56)
        # Uniform stays uniform — use np.allclose, since pytest.approx
        # on a scalar doesn't broadcast over an array.
        assert np.allclose(out, out[0, 0], atol=1e-6)

    def test_downsample_works(self):
        # Downsample 56x56 → 8x8: just shape and normalization
        x = np.ones((56, 56), dtype=np.float32) / (56 * 56)
        out = upsample_to_common_grid(x, target_hw=(8, 8))
        assert out.shape == (8, 8)
        assert out.sum() == pytest.approx(1.0, rel=1e-4)

    def test_peak_location_roughly_preserved(self):
        x = np.zeros((16, 16), dtype=np.float32)
        x[4, 4] = 1.0
        # Bilinear upsample 16x16 → 32x32: peak should land near (8, 8)
        out = upsample_to_common_grid(x, target_hw=(32, 32))
        peak = np.unravel_index(out.argmax(), out.shape)
        # Allow ±2 cell tolerance due to bilinear smoothing
        assert abs(peak[0] - 8) <= 2
        assert abs(peak[1] - 8) <= 2

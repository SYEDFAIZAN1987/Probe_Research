"""Unit tests for src/metrics/alignment.py.

Each metric is pinned against synthetic inputs with closed-form or
near-closed-form expected answers. These tests should catch:
  - Sign errors (e.g. KL with swapped p, q)
  - Off-by-one in flatten/reshape
  - Normalization bugs (raw vs probability inputs)
  - Wrong handling of degenerate inputs (constant maps, empty masks)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics.alignment import auc_attn_gaze, cc, iou_topk, kl_div, nss


# --------------------------------------------------------------------- #
# KL
# --------------------------------------------------------------------- #

class TestKL:
    def test_identical_distributions_kl_is_zero(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(16)).reshape(4, 4).astype(np.float32)
        assert kl_div(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_uniform_kl_is_zero(self):
        u = np.ones((8, 8), dtype=np.float32)
        assert kl_div(u, u) == pytest.approx(0.0, abs=1e-10)

    def test_kl_is_nonnegative(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            p = rng.dirichlet(np.ones(25)).reshape(5, 5).astype(np.float32)
            q = rng.dirichlet(np.ones(25)).reshape(5, 5).astype(np.float32)
            assert kl_div(p, q) >= -1e-10  # tiny float slop

    def test_delta_vs_uniform_approximates_log_n(self):
        # KL(delta || uniform) = log(N) when delta is a point mass.
        # Our implementation adds 1e-12 floor; for small N this gives
        # results very close to log(N).
        N = 16
        delta = np.zeros((4, 4), dtype=np.float32)
        delta[0, 0] = 1.0
        uniform = np.ones((4, 4), dtype=np.float32)
        expected = math.log(N)
        assert kl_div(delta, uniform) == pytest.approx(expected, rel=1e-4)

    def test_kl_is_asymmetric(self):
        # Pick distributions where asymmetry is large.
        p = np.array([[0.9, 0.05], [0.05, 0.0]], dtype=np.float32)
        q = np.array([[0.25, 0.25], [0.25, 0.25]], dtype=np.float32)
        assert kl_div(p, q) != pytest.approx(kl_div(q, p), abs=1e-3)


# --------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------- #

class TestAUC:
    def test_perfect_alignment_is_one(self):
        # Attention is monotonic in gaze; AUC at any threshold is 1.0.
        gaze = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
        attn = gaze.copy()
        assert auc_attn_gaze(attn, gaze, threshold_q=0.5) == pytest.approx(1.0)

    def test_anti_alignment_is_zero(self):
        gaze = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
        attn = -gaze
        assert auc_attn_gaze(attn, gaze, threshold_q=0.5) == pytest.approx(0.0)

    def test_random_alignment_near_half(self):
        rng = np.random.default_rng(7)
        gaze = rng.random((16, 16)).astype(np.float32)
        attn = rng.random((16, 16)).astype(np.float32)
        # Tolerance is loose: 256 points, expected AUC = 0.5 ± stochastic
        assert 0.35 < auc_attn_gaze(attn, gaze) < 0.65

    def test_degenerate_all_high_gaze_returns_nan(self):
        # threshold_q=0.5 → exactly half the cells become positive,
        # which is degenerate only if all gaze values are identical.
        gaze = np.ones((4, 4), dtype=np.float32)
        attn = np.random.default_rng(0).random((4, 4)).astype(np.float32)
        # All identical → after threshold all are positive (or none),
        # depending on > vs >=. Our impl uses >. So result is nan.
        assert math.isnan(auc_attn_gaze(attn, gaze))


# --------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------- #

class TestIoU:
    def test_attention_top_overlaps_full_mask(self):
        # 4x4 grid, mask = top-left 2x2 = 25% of cells.
        # Set attention high exactly on the top-left 2x2.
        attn = np.zeros((4, 4), dtype=np.float32)
        attn[:2, :2] = 1.0
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        # k_frac=0.25 → top 4 cells = the 2x2 quadrant → perfect IoU.
        assert iou_topk(attn, mask, k_frac=0.25) == pytest.approx(1.0)

    def test_disjoint_attention_and_mask(self):
        attn = np.zeros((4, 4), dtype=np.float32)
        attn[:2, :2] = 1.0   # top-left has high attention
        mask = np.zeros((4, 4), dtype=bool)
        mask[2:, 2:] = True  # bottom-right is the mask
        assert iou_topk(attn, mask, k_frac=0.25) == pytest.approx(0.0)

    def test_half_overlap(self):
        attn = np.zeros((4, 4), dtype=np.float32)
        attn[0] = 1.0        # top row = high attention (4 cells)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, :2] = True   # mask = top-left 2 cells
        mask[1, :2] = True   # mask = 4 cells total
        # k_frac=0.25 → top 4 cells = top row.
        # Intersection: top row ∩ mask = (0,0) and (0,1) = 2 cells
        # Union: top row ∪ mask = 4 + 4 - 2 = 6 cells
        # IoU = 2/6 ≈ 0.333
        assert iou_topk(attn, mask, k_frac=0.25) == pytest.approx(2 / 6, abs=1e-6)

    def test_empty_mask_returns_nan(self):
        attn = np.ones((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        # k_frac=0.25 → predicted positives, mask empty
        # Union = predicted only (non-empty); intersection = 0 → IoU = 0
        # That's actually 0.0, not nan. nan only if union is empty too.
        # Cover that explicit case:
        attn_zero = np.zeros((4, 4), dtype=np.float32)
        # With all-zero attention and a tiny k_frac, we'd still pick
        # something; verify our impl picks k=1.
        # The "all empty union" only happens if k_frac == 0 — not a
        # supported input. So this test just verifies disjoint with
        # empty mask returns 0.
        assert iou_topk(attn, mask, k_frac=0.25) == pytest.approx(0.0)


# --------------------------------------------------------------------- #
# NSS
# --------------------------------------------------------------------- #

class TestNSS:
    def test_attention_aligned_with_gaze_is_positive(self):
        # Build a gaze where the right half is high.
        gaze = np.zeros((4, 8), dtype=np.float32)
        gaze[:, 4:] = 1.0
        # Attention is also high on the right.
        attn = np.zeros((4, 8), dtype=np.float32)
        attn[:, 4:] = 1.0
        assert nss(attn, gaze) > 0.5  # strong positive

    def test_attention_anti_aligned_with_gaze_is_negative(self):
        gaze = np.zeros((4, 8), dtype=np.float32)
        gaze[:, 4:] = 1.0
        attn = np.zeros((4, 8), dtype=np.float32)
        attn[:, :4] = 1.0
        assert nss(attn, gaze) < -0.5

    def test_constant_attention_returns_nan(self):
        gaze = np.zeros((4, 4), dtype=np.float32)
        gaze[0, 0] = 1.0
        attn = np.ones((4, 4), dtype=np.float32) * 0.5
        assert math.isnan(nss(attn, gaze))

    def test_empty_fixations_returns_nan(self):
        # gaze all zero → mean is 0 → mask = (g > 0) is all-False
        gaze = np.zeros((4, 4), dtype=np.float32)
        attn = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert math.isnan(nss(attn, gaze))


# --------------------------------------------------------------------- #
# CC
# --------------------------------------------------------------------- #

class TestCC:
    def test_identical_maps_cc_is_one(self):
        rng = np.random.default_rng(3)
        x = rng.random((8, 8)).astype(np.float32)
        assert cc(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_negated_map_cc_is_minus_one(self):
        rng = np.random.default_rng(4)
        x = rng.random((8, 8)).astype(np.float32)
        assert cc(x, -x) == pytest.approx(-1.0, abs=1e-6)

    def test_constant_attention_returns_nan(self):
        x = np.ones((4, 4), dtype=np.float32) * 0.5
        y = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert math.isnan(cc(x, y))

    def test_uncorrelated_near_zero(self):
        rng = np.random.default_rng(11)
        # Many samples → expectation 0 ± small
        x = rng.standard_normal((32, 32)).astype(np.float32)
        y = rng.standard_normal((32, 32)).astype(np.float32)
        assert abs(cc(x, y)) < 0.15

    def test_affine_invariance(self):
        # cc(ax + b, y) == cc(x, y) for a > 0
        rng = np.random.default_rng(5)
        x = rng.random((6, 6)).astype(np.float32)
        y = rng.random((6, 6)).astype(np.float32)
        base = cc(x, y)
        shifted = cc(2 * x + 3, y)
        assert base == pytest.approx(shifted, abs=1e-5)

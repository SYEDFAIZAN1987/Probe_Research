"""Attention↔gaze alignment metrics from docs/extraction-spec.md §3.3.

All five metrics accept attention and gaze as 2D arrays on the same grid.
Probability-distribution metrics (KL) re-normalize defensively. Saliency
metrics (NSS, CC) follow the eye-tracking literature's definitions
(Bylinskii et al., What do different evaluation metrics tell us about
saliency models?, TPAMI 2018).

Convention: `attn` is the model's signal (probability or score),
`gaze` is the ground-truth radiologist signal (probability or density).
Both passed as numpy float arrays of identical shape.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- #
# KL divergence
# --------------------------------------------------------------------- #

def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) — both arrays renormalized to probability simplex.

    Lower is better (more aligned). Asymmetric: kl_div(attn, gaze)
    penalizes attention mass placed where there's no gaze; reverse
    penalizes the opposite. Spec uses kl_div(attn, gaze).
    """
    p = p.astype(np.float64) + 1e-12
    q = q.astype(np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (np.log(p) - np.log(q))).sum())


# --------------------------------------------------------------------- #
# AUC: attention score over thresholded-gaze binary map
# --------------------------------------------------------------------- #

def auc_attn_gaze(
    attn: np.ndarray,
    gaze: np.ndarray,
    threshold_q: float = 0.5,
) -> float:
    """AUC treating gaze>quantile(threshold_q) as the positive class and
    attention as the score. Higher = better separation of looked-at vs
    not-looked-at cells.

    threshold_q = 0.5 means cells in the top 50% of gaze density are
    positives. Choice of 0.5 is conservative; spec leaves this tunable.
    """
    from sklearn.metrics import roc_auc_score

    g = gaze.flatten()
    a = attn.flatten()
    thresh = float(np.quantile(g, threshold_q))
    y = (g > thresh).astype(int)
    if y.sum() == 0 or y.sum() == y.size:
        # Degenerate: all positive or all negative — AUC undefined.
        return float("nan")
    return float(roc_auc_score(y, a))


# --------------------------------------------------------------------- #
# IoU: top-k attention vs binary mask (e.g. bbox)
# --------------------------------------------------------------------- #

def iou_topk(
    attn: np.ndarray,
    binary_mask: np.ndarray,
    k_frac: float = 0.2,
) -> float:
    """IoU between the top-k% of attention cells and a binary mask
    (typically a rasterized bounding box, or thresholded gaze).

    k_frac = 0.2 means "the top 20% of attention cells form the
    predicted positive set." Higher = better overlap.
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask > 0
    a = attn.flatten()
    if a.size == 0:
        return float("nan")
    k = max(1, int(round(k_frac * a.size)))
    top_thresh = float(np.partition(a, -k)[-k])
    pred = (attn >= top_thresh)
    inter = (pred & binary_mask).sum()
    union = (pred | binary_mask).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


# --------------------------------------------------------------------- #
# NSS: Normalized Scanpath Saliency
# --------------------------------------------------------------------- #

def nss(attn: np.ndarray, gaze: np.ndarray) -> float:
    """Normalized Scanpath Saliency.

    NSS = mean over fixation locations of the z-scored attention map.
    Bylinskii et al. (2018): "NSS measures the correspondence between
    saliency maps and ground truth fixation locations, with one value
    per saliency map."

    Here we use gaze cells with above-mean density as proxies for
    "fixation locations" since we don't have raw fixation coordinates
    at this stage (they were rasterized to the grid already).

    Higher is better. NSS=0 means attention at fixations is no
    different from random; NSS<0 means attention is anti-correlated
    with fixations.
    """
    a = attn.astype(np.float64)
    g = gaze.astype(np.float64)
    a_mean = a.mean()
    a_std = a.std()
    if a_std < 1e-12:
        return float("nan")
    z = (a - a_mean) / a_std
    fixation_mask = g > g.mean()
    if fixation_mask.sum() == 0:
        return float("nan")
    return float(z[fixation_mask].mean())


# --------------------------------------------------------------------- #
# CC: Pearson correlation between the two saliency maps
# --------------------------------------------------------------------- #

def cc(attn: np.ndarray, gaze: np.ndarray) -> float:
    """Pearson correlation between flattened attention and gaze maps.

    Range [-1, +1]. Higher (closer to 1) is better; near 0 = no
    relationship; negative = anti-correlated.
    """
    a = attn.flatten().astype(np.float64)
    g = gaze.flatten().astype(np.float64)
    a_std = a.std()
    g_std = g.std()
    if a_std < 1e-12 or g_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, g)[0, 1])

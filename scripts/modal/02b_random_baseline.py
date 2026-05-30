"""02b — Random-attention baseline for the MedGemma layer pilot.

CPU-only sanity check. For the same 50 VinDr-CXR cases the pilot
used, generates 100 uniform-random 16×16 attention maps per case
and computes the same five alignment metrics (KL, IoU, AUC, CC, NSS)
against the radiologist bbox mask. This gives us the empirical
random-baseline distribution we need to honestly contextualize the
pilot's per-layer numbers.

Why this matters: pilot v3 picked L0-L4 on all 5 metrics, but IoU =
0.045 is below an analytical random baseline of ~0.10 for top-20%
attention. We need to know whether the AUC = 0.71 at L0-L4 is
genuinely above-random (it should be — random AUC ≈ 0.50) or
whether something about our extraction floor is producing
sub-random numbers across the board.

Outputs (on /data/pilot/medgemma/):
  random_baseline.parquet   per-(case, sample, metric) rows
  random_summary.md         L0-L4 pilot vs. random distribution,
                            with percentile + z-score per metric

Run:
    modal run scripts/modal/02b_random_baseline.py
    modal run scripts/modal/02b_random_baseline.py --n-random-samples 200

Expected runtime: <2 min on a 2-CPU container.
Expected cost: ~$0.01 in Modal credits.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# ----------------------------------------------------------------------- #
# Modal app + image (CPU-only, much smaller than the GPU image)
# ----------------------------------------------------------------------- #

app = modal.App("gazeprobe-medgemma-random-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "pillow",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "tabulate",
    )
    .add_local_python_source("src")
)

data_vol = modal.Volume.from_name("gazeprobe-data", create_if_missing=True)


# ----------------------------------------------------------------------- #
# Modal function
# ----------------------------------------------------------------------- #

@app.function(
    image=image,
    cpu=2.0,
    timeout=20 * 60,
    volumes={"/data": data_vol},
)
def compute_random_baseline(
    n_cases: int = 50,
    n_random_samples: int = 100,
    seed: int = 0,
) -> dict:
    """Compute random-attention baseline distributions for the same
    cases the layer pilot used (same seed = same cases)."""
    import numpy as np
    import pandas as pd
    from PIL import Image as PILImage

    from src.metrics.alignment import auc_attn_gaze, cc, iou_topk, kl_div, nss
    from src.metrics.rasterize import rasterize_bbox_to_grid

    # Reproduce the pilot's case sampling exactly (same seed → same RNG state
    # → same selection out of the candidate_ids array).
    rng = np.random.default_rng(seed)
    train = pd.read_csv("/data/vindr/train.csv")
    with_findings = train[train["class_id"] != 14].dropna(
        subset=["x_min", "y_min", "x_max", "y_max"]
    )
    candidate_ids = with_findings["image_id"].unique()
    sampled_ids = rng.choice(
        candidate_ids, size=min(n_cases, len(candidate_ids)), replace=False,
    )
    print(f"[*] mirroring pilot's case selection: {len(sampled_ids)} cases")

    # Separate RNG for random attention maps so changing n_random_samples
    # doesn't change which cases get picked.
    attn_rng = np.random.default_rng(seed + 1)

    G = 16  # MedGemma native grid
    rows = []
    skipped = 0
    failed = 0

    for ci, image_id in enumerate(sampled_ids):
        try:
            img_path = Path(f"/data/vindr/train_png/{image_id}.png")
            if not img_path.exists():
                skipped += 1
                continue
            with PILImage.open(img_path) as img:
                image_hw = (img.height, img.width)

            case_rows = with_findings[with_findings["image_id"] == image_id].copy()
            case_rows["area"] = (
                (case_rows["x_max"] - case_rows["x_min"])
                * (case_rows["y_max"] - case_rows["y_min"])
            )
            primary = case_rows.loc[case_rows["area"].idxmax()]
            class_name = str(primary["class_name"])
            same_class = case_rows[case_rows["class_name"] == class_name]
            bboxes = same_class[
                ["x_min", "y_min", "x_max", "y_max"]
            ].values.tolist()

            bbox_prob = rasterize_bbox_to_grid(
                bboxes, image_hw=image_hw, grid_edge=G, as_probability=True,
            )
            bbox_binary = rasterize_bbox_to_grid(
                bboxes, image_hw=image_hw, grid_edge=G, as_probability=False,
            )

            for sample_idx in range(n_random_samples):
                # Uniform-on-simplex random attention via Dirichlet(α=1).
                # This is the "model has no idea where the finding is"
                # null hypothesis.
                rand_attn = attn_rng.dirichlet(np.ones(G * G)).reshape(G, G).astype(np.float32)

                rows.append({
                    "case_idx": ci,
                    "image_id": str(image_id),
                    "class_name": class_name,
                    "sample_idx": sample_idx,
                    "kl":  float(kl_div(rand_attn, bbox_prob)),
                    "iou": float(iou_topk(rand_attn, bbox_binary > 0, k_frac=0.2)),
                    "auc": float(auc_attn_gaze(rand_attn, bbox_prob, threshold_q=0.5)),
                    "cc":  float(cc(rand_attn, bbox_prob)),
                    "nss": float(nss(rand_attn, bbox_prob)),
                })
            if (ci + 1) % 10 == 0:
                print(f"  [{ci+1}/{len(sampled_ids)}] rows so far: {len(rows)}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] case {ci} ({image_id}): {e}")

    if not rows:
        raise RuntimeError(
            "No rows collected. Check /data/vindr/train_png/ and "
            "/data/vindr/train.csv."
        )

    df = pd.DataFrame(rows)
    out_dir = Path("/data/pilot/medgemma")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "random_baseline.parquet", index=False)
    print(f"\n[OK] wrote {out_dir / 'random_baseline.parquet'} "
          f"({len(df)} rows)")

    # ------------------------------------------------------------------ #
    # Comparison: pilot's L0-L4 mean vs random baseline distribution
    # ------------------------------------------------------------------ #
    pilot_path = out_dir / "layer_kl.parquet"
    if not pilot_path.exists():
        print(f"[WARN] {pilot_path} not found; skipping side-by-side comparison")
        comparison = None
    else:
        pilot_df = pd.read_parquet(pilot_path)
        chosen_window = [0, 1, 2, 3, 4]
        pilot_window = pilot_df[pilot_df["layer"].isin(chosen_window)]

        comparison = {}
        metric_cols = ["kl", "iou", "auc", "cc", "nss"]
        print("\n" + "=" * 72)
        print(f"L0-L4 (pilot window) vs random-attention baseline")
        print("=" * 72)
        print(f"{'metric':>6} | {'L0-L4 mean':>11} | {'random mean':>12} | "
              f"{'random p05':>11} | {'random p95':>11} | {'percentile':>10}")
        print("-" * 72)
        for m in metric_cols:
            pilot_mean = float(pilot_window[m].mean())
            random_mean = float(df[m].mean())
            random_p05 = float(df[m].quantile(0.05))
            random_p95 = float(df[m].quantile(0.95))
            # What fraction of random samples score worse than the pilot
            # mean. For "higher is better" metrics (iou, auc, cc, nss),
            # this is the percentile (0-100). For kl (lower is better),
            # we report the inverse.
            higher_is_better = (m != "kl")
            if higher_is_better:
                pct = float((df[m] < pilot_mean).mean()) * 100
            else:
                pct = float((df[m] > pilot_mean).mean()) * 100
            comparison[m] = {
                "pilot_L0_L4_mean": pilot_mean,
                "random_mean": random_mean,
                "random_p05": random_p05,
                "random_p95": random_p95,
                "percentile_against_random": pct,
                "higher_is_better": higher_is_better,
            }
            print(f"{m:>6} | {pilot_mean:>11.4f} | {random_mean:>12.4f} | "
                  f"{random_p05:>11.4f} | {random_p95:>11.4f} | {pct:>9.1f}%")

        # Save a markdown summary for the paper
        md_path = out_dir / "random_summary.md"
        md_lines = [
            "# Random baseline vs. MedGemma L0-L4 pilot",
            "",
            f"50 VinDr-CXR cases, {n_random_samples} uniform-random "
            "attention maps per case (Dirichlet α=1 on 16×16 grid).",
            "",
            "Higher is better for IoU/AUC/CC/NSS. Lower is better for KL.",
            "Percentile = fraction of random samples worse than L0-L4 mean.",
            "",
            "| metric | L0-L4 mean | random mean | random p05 | random p95 | percentile |",
            "|--------|-----------|-------------|------------|------------|------------|",
        ]
        for m, c in comparison.items():
            md_lines.append(
                f"| {m} | {c['pilot_L0_L4_mean']:.4f} | "
                f"{c['random_mean']:.4f} | {c['random_p05']:.4f} | "
                f"{c['random_p95']:.4f} | {c['percentile_against_random']:.1f}% |"
            )
        md_path.write_text("\n".join(md_lines))
        print(f"\n[OK] wrote {md_path}")

    data_vol.commit()

    return {
        "n_cases": int(df["case_idx"].nunique()),
        "n_random_samples_per_case": n_random_samples,
        "n_rows": len(df),
        "skipped": skipped,
        "failed": failed,
        "comparison": comparison,
    }


# ----------------------------------------------------------------------- #
# Local entrypoint
# ----------------------------------------------------------------------- #

@app.local_entrypoint()
def main(n_cases: int = 50, n_random_samples: int = 100, seed: int = 0):
    print(f"[local] launching random-baseline computation: "
          f"n_cases={n_cases}, n_random_samples={n_random_samples}")
    result = compute_random_baseline.remote(
        n_cases=n_cases,
        n_random_samples=n_random_samples,
        seed=seed,
    )
    print("\n[local] result:")
    print(json.dumps(result, indent=2))
    print("\n[local] interpretation guide:")
    print("  - percentile ≥ 90%  : pilot window is clearly above random")
    print("  - percentile  50-90% : pilot window is above random but modestly")
    print("  - percentile  ~50%   : pilot window is indistinguishable from random")
    print("  - percentile  < 10%  : pilot window is BELOW random — bug or genuine")
    print("                         anti-alignment; investigate before locking")

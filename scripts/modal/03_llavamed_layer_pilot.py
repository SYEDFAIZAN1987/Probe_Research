"""03 — LLaVA-Med layer-selection pilot (Modal port, BBoxProbe v1).

Second of three model-specific pilots. Parallel structure to
02_medgemma_layer_pilot.py — same 50 VinDr-CXR cases (same seed),
same teacher-forced canonical-template approach, same 5 alignment
metrics. Differences:

  - Uses LLaVAMedExtractor (untested skeleton in src/attn/
    extract_llavamed.py). First successful run validates the
    extractor's HF-class choice (LlavaForConditionalGeneration),
    image-token expansion mode (single vs 576-pre-expanded), and
    chat-template path.
  - Native grid is 24×24 = 576 image tokens (CLIP-ViT-L/14 @ 336²),
    so bbox masks are rasterized at higher resolution than MedGemma's
    16×16. Random-baseline IoU will be different: top-20% of 576 cells
    = 115 cells, baseline IoU ≈ 0.10 (similar to MedGemma's).
  - LLaVA-Med uses the mistral_instruct conversation template. If
    apply_chat_template fails inside the extractor, the documented
    fallback is manual construction of `<s>[INST] <image>\\n{user}
    [/INST] {report}</s>`.

Run:
    modal run scripts/modal/03_llavamed_layer_pilot.py
    modal run scripts/modal/03_llavamed_layer_pilot.py --n-cases 100

After this completes, run 02b_random_baseline.py with --grid-edge 24
(or the equivalent override) to get a directly-comparable random
floor for LLaVA-Med.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# ----------------------------------------------------------------------- #
# Modal app + image (same recipe as the MedGemma pilot)
# ----------------------------------------------------------------------- #

app = modal.App("gazeprobe-llavamed-layer-pilot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.45",
        "accelerate>=0.34",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "pillow",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "matplotlib",
        "nltk",
        "tqdm",
        "tabulate",
    )
    .add_local_python_source("src")
)

hf_cache = modal.Volume.from_name("gazeprobe-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("gazeprobe-data", create_if_missing=True)


# ----------------------------------------------------------------------- #
# Modal function
# ----------------------------------------------------------------------- #

@app.function(
    image=image,
    gpu="L40S",
    timeout=2 * 60 * 60,
    cpu=4.0,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data": data_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def run_pilot(n_cases: int = 50, seed: int = 0) -> dict:
    import os
    import sys
    import traceback

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    from src.attn.extract_llavamed import LLaVAMedExtractor
    from src.metrics.alignment import auc_attn_gaze, cc, iou_topk, kl_div, nss
    from src.metrics.rasterize import rasterize_bbox_to_grid

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Phase 1 — Load + filter VinDr train.csv  (identical to MedGemma pilot)
    # ------------------------------------------------------------------ #
    train_csv_path = Path("/data/vindr/train.csv")
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"{train_csv_path} not on volume. Run "
            "scripts/modal/00_download_vindr.py first."
        )
    train = pd.read_csv(train_csv_path)
    print(f"[*] train.csv: {len(train)} rows, "
          f"{train['image_id'].nunique()} unique image_ids")

    with_findings = train[train["class_id"] != 14].dropna(
        subset=["x_min", "y_min", "x_max", "y_max"]
    )
    candidate_ids = with_findings["image_id"].unique()
    print(f"[*] {len(candidate_ids)} image_ids have at least one bbox")

    n_take = min(n_cases, len(candidate_ids))
    sampled_ids = rng.choice(candidate_ids, size=n_take, replace=False)
    print(f"[*] sampled {n_take} cases (seed={seed} — same cases as MedGemma pilot)")

    # ------------------------------------------------------------------ #
    # Phase 2 — Load LLaVA-Med
    # ------------------------------------------------------------------ #
    print("[*] loading LLaVA-Med-1.5 in bf16 (attn_implementation=eager)...")
    print("    [Note: this extractor was UNTESTED before this run. "
          "First-load may surface architecture surprises.]")
    extractor = LLaVAMedExtractor()
    extractor.load()
    n_layers = extractor.model.config.text_config.num_hidden_layers \
        if hasattr(extractor.model.config, "text_config") \
        else extractor.model.config.num_hidden_layers
    print(f"[*] model loaded; reported n_decoder_layers = {n_layers}")

    # Smoke forward — this also tests the image-token-expansion path.
    # The LLaVAMedExtractor raises a clear NotImplementedError if the
    # single-token-image mode is in effect; we'd see that here.
    smoke_img = Image.new("RGB", (336, 336), color=(127, 127, 127))
    smoke_template = (
        "There is consolidation visible in this chest radiograph."
    )
    try:
        smoke_result = extractor.extract_teacher_forced(smoke_img, smoke_template)
    except NotImplementedError as e:
        print("[FATAL] LLaVA-Med uses single-token image expansion mode.")
        print(f"        {e}")
        print("        The extractor needs to be updated to read attentions "
              "over the POST-expansion sequence rather than input_ids. "
              "Inspect model.get_input_embeddings() and the forward path "
              "before retrying.")
        raise

    if smoke_result.attentions is None or len(smoke_result.attentions) == 0:
        raise RuntimeError(
            "Smoke forward returned no attentions. Confirm "
            "attn_implementation='eager' is honored by transformers."
        )

    img_pos_count = smoke_result.image_token_positions.numel()
    expected_img = extractor.native_grid[0] * extractor.native_grid[1]
    smoke_shapes = {tuple(a.shape[-2:]) for a in smoke_result.attentions}
    print(f"[*] smoke forward OK: {len(smoke_result.attentions)} layers, "
          f"img_pos count = {img_pos_count} (expected {expected_img})")
    print(f"[*] {len(smoke_result.attentions)} attention tensors, "
          f"{len(smoke_shapes)} distinct (q,k) shape(s): {smoke_shapes}")

    if img_pos_count != expected_img:
        raise RuntimeError(
            f"image_token_positions count ({img_pos_count}) does not match "
            f"the native grid ({expected_img}). Inspect "
            f"LLaVAMedExtractor._find_image_positions before running the "
            f"full loop."
        )

    # ------------------------------------------------------------------ #
    # Phase 3 — Per-case extraction
    # ------------------------------------------------------------------ #
    rows = []
    skipped = 0
    failed = 0
    G = extractor.native_grid[0]
    for ci, image_id in enumerate(sampled_ids):
        try:
            img_path = Path(f"/data/vindr/train_png/{image_id}.png")
            if not img_path.exists():
                skipped += 1
                continue
            image = Image.open(img_path).convert("RGB")
            image_hw = (image.size[1], image.size[0])

            case_rows = with_findings[with_findings["image_id"] == image_id].copy()
            case_rows["area"] = (
                (case_rows["x_max"] - case_rows["x_min"])
                * (case_rows["y_max"] - case_rows["y_min"])
            )
            primary = case_rows.loc[case_rows["area"].idxmax()]
            class_name = str(primary["class_name"])
            same_class = case_rows[case_rows["class_name"] == class_name]
            bboxes = same_class[["x_min", "y_min", "x_max", "y_max"]].values.tolist()

            bbox_prob = rasterize_bbox_to_grid(
                bboxes, image_hw=image_hw, grid_edge=G, as_probability=True,
            )
            bbox_binary = rasterize_bbox_to_grid(
                bboxes, image_hw=image_hw, grid_edge=G, as_probability=False,
            )

            template = (
                f"There is {class_name} visible in this chest radiograph."
            )
            result = extractor.extract_teacher_forced(image, template)

            a_start, a_end = result.assistant_range
            img_pos = result.image_token_positions
            content_mask = extractor.content_token_mask(result)
            if content_mask.sum() == 0:
                print(f"  [WARN] case {ci}: no content tokens")
                skipped += 1
                continue

            for layer_idx, attn in enumerate(result.attentions):
                a = attn[0].mean(dim=0)                          # (q, k)
                a = a[a_start:a_end][content_mask][:, img_pos]   # (n_content, n_img)
                if a.numel() == 0:
                    continue
                a = a.float().mean(dim=0).cpu().numpy()           # (n_img,)
                a_grid = a.reshape(G, G)
                a_grid_norm = a_grid / (a_grid.sum() + 1e-12)

                rows.append({
                    "case_idx": ci,
                    "image_id": str(image_id),
                    "class_name": class_name,
                    "layer": layer_idx,
                    "kl":   float(kl_div(a_grid_norm, bbox_prob)),
                    "iou":  float(iou_topk(a_grid_norm, bbox_binary > 0, k_frac=0.2)),
                    "auc":  float(auc_attn_gaze(a_grid_norm, bbox_prob, threshold_q=0.5)),
                    "cc":   float(cc(a_grid_norm, bbox_prob)),
                    "nss":  float(nss(a_grid_norm, bbox_prob)),
                })

            if (ci + 1) % 5 == 0:
                print(f"  [{ci+1}/{n_take}] done; rows so far: {len(rows)}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] case {ci} ({image_id}): {e}", file=sys.stderr)
            traceback.print_exc()

    if not rows:
        raise RuntimeError(
            "Pilot collected zero rows. Most likely the extractor or one of "
            "its unverified assumptions failed silently — re-read the "
            "Phase-3 traceback output."
        )

    print(f"\n[*] phase 3 done. rows: {len(rows)}, "
          f"skipped: {skipped}, failed: {failed}")

    # ------------------------------------------------------------------ #
    # Phase 4 — Aggregate + pick best layer window  (IoU primary)
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(rows)
    out_dir = Path("/data/pilot/llavamed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "layer_kl.parquet", index=False)
    print(f"[OK] wrote {out_dir / 'layer_kl.parquet'}")

    metric_cols = ["kl", "iou", "auc", "cc", "nss"]
    per_layer = df.groupby("layer")[metric_cols].mean().sort_index()

    iou_rolling = per_layer["iou"].rolling(window=5, center=False).mean()
    best_end = int(iou_rolling.idxmax())
    best_start = best_end - 4
    best_window = list(range(best_start, best_end + 1))
    best_iou = float(iou_rolling.max())

    kl_rolling  = per_layer["kl"].rolling(5).mean()
    auc_rolling = per_layer["auc"].rolling(5).mean()
    cc_rolling  = per_layer["cc"].rolling(5).mean()

    print(f"\n[REC] best contiguous-5 layer window by IoU (primary): "
          f"L{best_start}-L{best_end} (mean IoU = {best_iou:.4f})")
    print(f"  (KL  would pick window ending at L{int(kl_rolling.idxmin()) if kl_rolling.notna().any() else '?'})")
    print(f"  (AUC would pick window ending at L{int(auc_rolling.idxmax()) if auc_rolling.notna().any() else '?'})")
    print(f"  (CC  would pick window ending at L{int(cc_rolling.idxmax()) if cc_rolling.notna().any() else '?'})")

    print("\nPer-layer metric means:")
    print(per_layer.to_string(float_format="%.4f"))

    window_path = out_dir / "layer_window.json"
    window_path.write_text(json.dumps({
        "model": extractor.model_id,
        "layers": best_window,
        "selection_metric": "iou",
        "mean_iou_in_window": best_iou,
        "n_cases": int(df["case_idx"].nunique()),
        "per_layer": {
            str(int(layer)): {m: float(per_layer.loc[layer, m]) for m in metric_cols}
            for layer in per_layer.index
        },
    }, indent=2))
    print(f"[OK] wrote {window_path}")

    rec_path = out_dir / "recommendation.md"
    try:
        table = per_layer.to_markdown(floatfmt=".4f")
    except Exception:
        table = per_layer.to_string(float_format="%.4f")
    rec_path.write_text(
        f"# LLaVA-Med-1.5 layer-pilot recommendation (BBoxProbe v1)\n\n"
        f"Primary metric: **IoU** (top-20% attention vs. bbox mask), "
        f"per docs/extraction-spec.md §3.3.\n\n"
        f"Best contiguous-5 layer window by mean IoU on "
        f"{df['case_idx'].nunique()} VinDr-CXR cases (same seed as the "
        f"MedGemma pilot, so the *same* 50 image_ids):\n\n"
        f"**L{best_start}-L{best_end}** (mean IoU = {best_iou:.4f})\n\n"
        f"## All metrics per layer\n\n{table}\n\n"
        f"## Lock-in step\n\n"
        f"Replace the placeholder line in `docs/extraction-spec.md` §Q1 "
        f"(LLaVA-Med row) with:\n\n"
        f"> **L{best_start}-L{best_end}** (frozen 2026-MM-DD by 50-case pilot, "
        f"mean IoU = {best_iou:.4f}). See `data/pilot/llavamed/layer_window.json`.\n"
    )
    print(f"[OK] wrote {rec_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for ax, m in zip(axes.flatten(), metric_cols):
            ax.plot(per_layer.index, per_layer[m], marker="o")
            ax.axvspan(best_start - 0.5, best_end + 0.5, color="tab:green",
                       alpha=0.2, label=f"chosen L{best_start}-L{best_end}")
            ax.set_xlabel("decoder layer")
            ax.set_ylabel(f"mean {m}")
            ax.set_title(m.upper() + (" (PRIMARY)" if m == "iou" else ""))
            ax.legend(fontsize=8)
        axes.flatten()[-1].axis("off")
        fig.suptitle("LLaVA-Med-1.5 layer scan vs VinDr-CXR bbox alignment "
                     "(50 cases)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "layer_scan.png", dpi=120)
        plt.close(fig)
        print(f"[OK] wrote {out_dir / 'layer_scan.png'}")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    data_vol.commit()

    return {
        "n_cases_used": int(df["case_idx"].nunique()),
        "skipped": int(skipped),
        "failed": int(failed),
        "n_layers_scanned": int(per_layer.shape[0]),
        "native_grid": list(extractor.native_grid),
        "selection_metric": "iou",
        "best_window": best_window,
        "mean_iou_in_window": best_iou,
        "per_layer": {
            str(int(layer)): {m: round(float(per_layer.loc[layer, m]), 4)
                              for m in metric_cols}
            for layer in per_layer.index
        },
    }


# ----------------------------------------------------------------------- #
# Local entrypoint
# ----------------------------------------------------------------------- #

@app.local_entrypoint()
def main(n_cases: int = 50, seed: int = 0):
    print(f"[local] launching LLaVA-Med layer pilot: n_cases={n_cases}, seed={seed}")
    result = run_pilot.remote(n_cases=n_cases, seed=seed)
    print("\n[local] result:")
    print(json.dumps(result, indent=2))
    print("\n[local] next: compare per-layer pattern vs MedGemma's L0-L4. "
          "Same early-layer dominance, or LLaVA's expected L14-18 mid-layers?")

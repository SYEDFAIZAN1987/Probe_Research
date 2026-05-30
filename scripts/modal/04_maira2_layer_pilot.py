"""04 — MAIRA-2 layer-selection pilot (Modal port, BBoxProbe v1).

Third of three model-specific pilots. Same structure as 02/03 — same
50 VinDr-CXR cases (seed=0), same alignment metrics, IoU as primary
selection criterion.

Differences from 02/03 (any of which can break the first run):
  - Uses MAIRA2Extractor (untested skeleton in src/attn/
    extract_maira2.py). First-load surprises this script tries to
    surface explicitly:
      * The custom processor.format_and_preprocess_reporting_input
        function with empty clinical-context fields (VinDr has no
        indication/technique/etc.; we pass "").
      * Provisional native grid (37, 37) — could turn out to be 24×24
        or 16×16. Smoke check prints the actual image-token count and
        bails clearly if it doesn't match the provisional value.
      * Image-token id detection has a 3-tier fallback in the
        extractor; smoke prints which path succeeded.
      * No chat template — teacher-forcing works by appending the
        gold report's tokens to the processor's prompt input_ids and
        running forward.

  - MAIRA-2 generates explicit bbox tokens in production use; this
    pilot does NOT exercise that path. The bbox-emission audit is a
    separate experiment for the full extraction phase. This pilot is
    just for layer selection.

Failure modes braced for (any of these means we patch the extractor
and re-run):
  - AutoProcessor / AutoModelForCausalLM loading with trust_remote_code
    may fail if MAIRA-2's modeling code isn't on HF
  - format_and_preprocess_reporting_input may not exist or have a
    different signature
  - Empty-string clinical fields may produce a malformed prompt
  - Native grid mismatch — print the actual count, fail fast
  - The same single-token-image expansion problem as LLaVA-Med
    may apply here too

Run:
    modal run scripts/modal/04_maira2_layer_pilot.py

Expected runtime on first attempt: highly variable. Subsequent runs
once the extractor settles: ~15 min on L40S, ~$0.45.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# ----------------------------------------------------------------------- #
# Modal app + image
# ----------------------------------------------------------------------- #

app = modal.App("gazeprobe-maira2-layer-pilot")

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
        # MAIRA-2 may require this for its custom processor (image-tower
        # heuristically calls into torchvision for resize ops in some
        # MSR releases).
        "torchvision",
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

    from src.attn.extract_maira2 import MAIRA2Extractor
    from src.metrics.alignment import auc_attn_gaze, cc, iou_topk, kl_div, nss
    from src.metrics.rasterize import rasterize_bbox_to_grid

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Phase 1 — Same case selection as 02/03
    # ------------------------------------------------------------------ #
    train = pd.read_csv("/data/vindr/train.csv")
    print(f"[*] train.csv: {len(train)} rows, "
          f"{train['image_id'].nunique()} unique image_ids")
    with_findings = train[train["class_id"] != 14].dropna(
        subset=["x_min", "y_min", "x_max", "y_max"]
    )
    candidate_ids = with_findings["image_id"].unique()
    n_take = min(n_cases, len(candidate_ids))
    sampled_ids = rng.choice(candidate_ids, size=n_take, replace=False)
    print(f"[*] sampled {n_take} cases (seed={seed} — same as 02/03)")

    # ------------------------------------------------------------------ #
    # Phase 2 — Load MAIRA-2
    # ------------------------------------------------------------------ #
    print("[*] loading microsoft/maira-2 (trust_remote_code=True)...")
    print("    [Extractor is UNTESTED; expect first-load surprises.]")
    extractor = MAIRA2Extractor()
    try:
        extractor.load()
    except Exception as e:
        print(f"[FATAL] MAIRA-2 load failed: {e}")
        print(f"        Likely causes:")
        print(f"          (1) trust_remote_code=True did not pull the custom "
              f"modeling code from HF (auth/network issue)")
        print(f"          (2) MAIRA-2 requires AutoModelForVision2Seq or "
              f"another HF class than AutoModelForCausalLM")
        print(f"          (3) torchvision missing — added to image but "
              f"could be the wrong version")
        raise

    n_layers_attr = getattr(extractor.model.config, "num_hidden_layers", None)
    if hasattr(extractor.model.config, "text_config"):
        n_layers_attr = getattr(
            extractor.model.config.text_config, "num_hidden_layers", n_layers_attr,
        )
    print(f"[*] model loaded; reported n_decoder_layers = {n_layers_attr}")

    # Diagnostics: which processor methods exist?
    print(f"[*] processor type: {type(extractor.processor).__name__}")
    print(f"[*] processor methods of interest:")
    for attr in ("format_and_preprocess_reporting_input",
                 "apply_chat_template", "__call__"):
        has_it = hasattr(extractor.processor, attr)
        print(f"    {attr}: {'present' if has_it else 'MISSING'}")

    # Smoke forward — uses a synthetic CXR.
    smoke_img = Image.new("RGB", (518, 518), color=(127, 127, 127))
    smoke_template = (
        "There is consolidation visible in this chest radiograph."
    )
    try:
        smoke_result = extractor.extract_teacher_forced(smoke_img, smoke_template)
    except NotImplementedError as e:
        print(f"[FATAL] Extractor hit a NotImplementedError: {e}")
        raise
    except Exception as e:
        print(f"[FATAL] Smoke forward failed: {e}")
        print(f"        First place to look: extractor.prepare_inputs's "
              f"call to format_and_preprocess_reporting_input. "
              f"Inspect the kwargs it accepts.")
        traceback.print_exc()
        raise

    if smoke_result.attentions is None or len(smoke_result.attentions) == 0:
        raise RuntimeError(
            "Smoke forward returned no attentions. Confirm "
            "attn_implementation='eager' is honored."
        )

    img_pos_count = smoke_result.image_token_positions.numel()
    expected_img = extractor.native_grid[0] * extractor.native_grid[1]
    smoke_shapes = {tuple(a.shape[-2:]) for a in smoke_result.attentions}
    a_start_smoke, a_end_smoke = smoke_result.assistant_range
    print(f"[*] smoke forward OK:")
    print(f"      n_layers = {len(smoke_result.attentions)}")
    print(f"      attention (q,k) shapes = {smoke_shapes}")
    print(f"      img_pos count = {img_pos_count} "
          f"(provisional native grid {extractor.native_grid} → "
          f"expected {expected_img})")
    print(f"      assistant_range = ({a_start_smoke}, {a_end_smoke}) "
          f"= {a_end_smoke - a_start_smoke} tokens")

    if img_pos_count != expected_img:
        # This is the most likely first-load surprise. Suggest the
        # actual grid based on what we observed.
        from math import isqrt
        guessed_edge = isqrt(img_pos_count)
        if guessed_edge * guessed_edge == img_pos_count:
            suggestion = (f"Update src/attn/extract_maira2.py "
                          f"MAIRA2Extractor.__init__: change "
                          f"native_grid=(37, 37) to "
                          f"native_grid=({guessed_edge}, {guessed_edge}).")
        else:
            suggestion = (f"image_token_count = {img_pos_count} is not "
                          f"a perfect square; native grid may not be "
                          f"square or the count is wrong.")
        raise RuntimeError(
            f"MAIRA-2 native grid mismatch. Provisional was "
            f"{extractor.native_grid} ({expected_img} tokens), actual "
            f"is {img_pos_count}. {suggestion}"
        )

    # ------------------------------------------------------------------ #
    # Phase 3 — Per-case extraction (parallel to 02/03)
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
                skipped += 1
                continue

            for layer_idx, attn in enumerate(result.attentions):
                a = attn[0].mean(dim=0)
                a = a[a_start:a_end][content_mask][:, img_pos]
                if a.numel() == 0:
                    continue
                a = a.float().mean(dim=0).cpu().numpy()
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
        raise RuntimeError("Pilot collected zero rows — debug the failures above.")

    print(f"\n[*] phase 3 done. rows: {len(rows)}, "
          f"skipped: {skipped}, failed: {failed}")

    # ------------------------------------------------------------------ #
    # Phase 4 — Aggregate + pick best layer window
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(rows)
    out_dir = Path("/data/pilot/maira2")
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

    print(f"\n[REC] best contiguous-5 window by IoU (primary): "
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
        "native_grid": list(extractor.native_grid),
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
        f"# MAIRA-2 layer-pilot recommendation (BBoxProbe v1)\n\n"
        f"Primary metric: **IoU** (top-20% attention vs. bbox mask), "
        f"per docs/extraction-spec.md §3.3.\n\n"
        f"Best contiguous-5 layer window by mean IoU on "
        f"{df['case_idx'].nunique()} VinDr-CXR cases:\n\n"
        f"**L{best_start}-L{best_end}** (mean IoU = {best_iou:.4f})\n\n"
        f"Native grid used: {extractor.native_grid}.\n\n"
        f"## All metrics per layer\n\n{table}\n\n"
        f"## Lock-in step\n\n"
        f"Replace the placeholder in `docs/extraction-spec.md` §Q1 "
        f"(MAIRA-2 row) with:\n\n"
        f"> **L{best_start}-L{best_end}** (frozen 2026-MM-DD by 50-case pilot, "
        f"mean IoU = {best_iou:.4f}). Native grid {extractor.native_grid}. "
        f"See `data/pilot/maira2/layer_window.json`.\n"
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
        fig.suptitle("MAIRA-2 layer scan vs VinDr-CXR bbox alignment "
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
    print(f"[local] launching MAIRA-2 layer pilot: n_cases={n_cases}, seed={seed}")
    result = run_pilot.remote(n_cases=n_cases, seed=seed)
    print("\n[local] result:")
    print(json.dumps(result, indent=2))
    print("\n[local] cross-model comparison once locked:")
    print("  Compare L0-L4 mean metrics with MedGemma (real signal) and")
    print("  LLaVA-Med (near-random). MAIRA-2 is interesting because it")
    print("  ALSO emits explicit bboxes — that's a separate experiment.")

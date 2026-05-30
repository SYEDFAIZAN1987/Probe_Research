"""02 — MedGemma layer-selection pilot (Modal port, BBoxProbe v1).

Resolves the candidate layer range in docs/extraction-spec.md §Q1 for
MedGemma by running attention extraction on 50 VinDr-CXR cases across
ALL 32 decoder layers, then picking the best contiguous-5-layer
window by mean KL alignment with the radiologist bbox mask.

Procedure:
  1. Load MedGemma-4B-IT in bf16 on a single L40S (attn_implementation
     "eager" so output_attentions actually materializes).
  2. Sample 50 cases from /data/vindr/train.csv that have at least
     one bbox-annotated finding (class_id != 14).
  3. For each case:
       a. Load the PNG from /data/vindr/train_png/{image_id}.png.
       b. Pick the primary class (the one with the largest bbox
          area in the case).
       c. Build a canonical-template "report":
            "There is <class_name> visible in this chest radiograph."
       d. Teacher-force MedGemma with that report; capture
          out.attentions across all 32 decoder layers.
       e. Aggregate (heads → mean, content tokens → mean) to one
          attention vector per (case, layer) over the 256 image
          tokens. Reshape to 16x16 (MedGemma native grid).
       f. Rasterize the union of bboxes for the primary class to
          16x16 as a probability mass.
       g. Compute KL(attn || bbox-mask) per (case, layer).
  4. Aggregate mean KL per layer across all cases. Pick the
     contiguous-5-layer window that minimizes mean KL via a rolling
     window. Lower KL = better alignment.
  5. Write outputs to /data/pilot/medgemma/:
       layer_kl.parquet         per-(case, layer) KL values
       layer_window.json        consumed by MedGemmaExtractor at
                                inference time via default_layers()
       recommendation.md        human-readable copy-paste for the
                                spec doc

Run:
    modal run scripts/modal/02_medgemma_layer_pilot.py
    modal run scripts/modal/02_medgemma_layer_pilot.py --n-cases 100

After it finishes, commit the chosen window into
docs/extraction-spec.md §Q1 (MedGemma row), replacing the
"L11-22 candidate; pilot to pick best 5" placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# ----------------------------------------------------------------------- #
# Modal app + image
# ----------------------------------------------------------------------- #

app = modal.App("gazeprobe-medgemma-layer-pilot")

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
    )
    # Make our project src/ importable inside the container.
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
    timeout=2 * 60 * 60,    # 2 h cap; 50 cases × 32 layers ~10 min on L40S
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
    import string
    import traceback

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    # Ensure HF token is in the env (Modal Secret writes it for us, but
    # some transformers versions look at HF_HOME-derived paths only).
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # NLTK stopwords (used by base extractor's content_token_mask)
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    from src.attn.extract_medgemma import MedGemmaExtractor
    from src.metrics.alignment import kl_div
    from src.metrics.rasterize import rasterize_bbox_to_grid

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Phase 1 — Load + filter VinDr train.csv
    # ------------------------------------------------------------------ #
    train_csv_path = Path("/data/vindr/train.csv")
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"{train_csv_path} not on volume. Run "
            "scripts/modal/00_download_vindr.py first."
        )
    train = pd.read_csv(train_csv_path)
    print(f"[*] train.csv: {len(train)} annotation rows, "
          f"{train['image_id'].nunique()} unique image_ids")

    # VinDr class_id 14 = "No finding" → no bbox; drop.
    with_findings = train[train["class_id"] != 14].dropna(
        subset=["x_min", "y_min", "x_max", "y_max"]
    )
    candidate_ids = with_findings["image_id"].unique()
    print(f"[*] {len(candidate_ids)} image_ids have at least one bbox")

    n_take = min(n_cases, len(candidate_ids))
    sampled_ids = rng.choice(candidate_ids, size=n_take, replace=False)
    print(f"[*] sampled {n_take} cases for the pilot")

    # ------------------------------------------------------------------ #
    # Phase 2 — Load MedGemma
    # ------------------------------------------------------------------ #
    print("[*] loading MedGemma-4B-IT in bf16 (attn_implementation=eager)...")
    extractor = MedGemmaExtractor()
    extractor.load()  # bf16 by default per the locked spec
    n_layers = extractor.model.config.text_config.num_hidden_layers
    print(f"[*] model loaded; n_decoder_layers = {n_layers}")

    # Verify attentions actually materialize on a smoke forward
    # (cheaper to fail here than mid-loop)
    smoke_img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    smoke_result = extractor.extract_teacher_forced(
        smoke_img, "There is consolidation visible in this chest radiograph."
    )
    if smoke_result.attentions is None or len(smoke_result.attentions) == 0:
        raise RuntimeError(
            "Smoke forward returned no attentions. Confirm "
            "attn_implementation='eager' is honored by the installed "
            "transformers version."
        )
    print(f"[*] smoke forward OK: {len(smoke_result.attentions)} layers, "
          f"img_pos count = {smoke_result.image_token_positions.numel()}")

    # ------------------------------------------------------------------ #
    # Phase 3 — Per-case extraction
    # ------------------------------------------------------------------ #
    rows = []
    skipped = 0
    failed = 0
    for ci, image_id in enumerate(sampled_ids):
        try:
            img_path = Path(f"/data/vindr/train_png/{image_id}.png")
            if not img_path.exists():
                skipped += 1
                continue
            image = Image.open(img_path).convert("RGB")
            image_hw = (image.size[1], image.size[0])  # (H, W)

            # Pick the case's primary class as the one with the largest
            # bbox area. Build a union mask over all bboxes of that class.
            case_rows = with_findings[with_findings["image_id"] == image_id].copy()
            case_rows["area"] = (
                (case_rows["x_max"] - case_rows["x_min"])
                * (case_rows["y_max"] - case_rows["y_min"])
            )
            primary = case_rows.loc[case_rows["area"].idxmax()]
            class_name = str(primary["class_name"])
            same_class = case_rows[case_rows["class_name"] == class_name]
            bboxes = same_class[["x_min", "y_min", "x_max", "y_max"]].values.tolist()

            bbox_mask = rasterize_bbox_to_grid(
                bboxes,
                image_hw=image_hw,
                grid_edge=extractor.native_grid[0],
                as_probability=True,
            )

            # Teacher-force the model with a canonical template that
            # mentions the primary class. Content-token attention to
            # the image tokens is the signal.
            template = (
                f"There is {class_name} visible in this chest radiograph."
            )
            result = extractor.extract_teacher_forced(image, template)

            a_start, a_end = result.assistant_range
            img_pos = result.image_token_positions
            content_mask = extractor.content_token_mask(result)
            if content_mask.sum() == 0:
                print(f"  [WARN] case {ci} ({image_id}): no content tokens")
                skipped += 1
                continue

            G = extractor.native_grid[0]
            for layer_idx, attn in enumerate(result.attentions):
                a = attn[0].mean(dim=0)                          # (q, k)
                a = a[a_start:a_end][content_mask][:, img_pos]   # (n_content, n_img)
                if a.numel() == 0:
                    continue
                a = a.float().mean(dim=0).cpu().numpy()          # (n_img,)
                a_grid = a.reshape(G, G)
                a_grid = a_grid / (a_grid.sum() + 1e-12)
                kl = kl_div(a_grid, bbox_mask)
                rows.append({
                    "case_idx": ci,
                    "image_id": str(image_id),
                    "class_name": class_name,
                    "layer": layer_idx,
                    "kl": float(kl),
                })

            if (ci + 1) % 5 == 0:
                print(f"  [{ci+1}/{n_take}] done; rows so far: {len(rows)}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] case {ci} ({image_id}): {e}", file=sys.stderr)
            traceback.print_exc()

    if not rows:
        raise RuntimeError(
            "Pilot collected zero rows. Check: (a) PNGs at /data/vindr/train_png/, "
            "(b) train.csv schema, (c) MedGemma loaded correctly."
        )

    print(f"\n[*] phase 3 done. rows: {len(rows)}, "
          f"skipped: {skipped}, failed: {failed}")

    # ------------------------------------------------------------------ #
    # Phase 4 — Aggregate + pick best layer window
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(rows)
    out_dir = Path("/data/pilot/medgemma")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "layer_kl.parquet", index=False)
    print(f"[OK] wrote {out_dir / 'layer_kl.parquet'}")

    mean_per_layer = df.groupby("layer")["kl"].mean().sort_index()
    rolling = mean_per_layer.rolling(window=5, center=False).mean()
    best_end = int(rolling.idxmin())
    best_start = best_end - 4
    best_window = list(range(best_start, best_end + 1))
    min_kl = float(rolling.min())

    print(f"\n[REC] best contiguous-5 layer window: "
          f"L{best_start}-L{best_end} (mean KL = {min_kl:.4f})")
    print("\nMean KL per layer:")
    for layer, kl in mean_per_layer.items():
        marker = "  ← in window" if layer in best_window else ""
        print(f"  L{layer:2d}: {kl:.4f}{marker}")

    # JSON sidecar read by MedGemmaExtractor.default_layers()
    window_path = out_dir / "layer_window.json"
    window_path.write_text(json.dumps({
        "model": extractor.model_id,
        "layers": best_window,
        "mean_kl": min_kl,
        "n_cases": int(df["case_idx"].nunique()),
        "all_mean_kl_per_layer": {int(k): float(v) for k, v in mean_per_layer.items()},
    }, indent=2))
    print(f"[OK] wrote {window_path}")

    # Human-readable recommendation file
    rec_path = out_dir / "recommendation.md"
    rec_path.write_text(
        f"# MedGemma layer-pilot recommendation (BBoxProbe v1)\n\n"
        f"Best contiguous-5 layer window by mean KL(attn || bbox) on "
        f"{df['case_idx'].nunique()} VinDr-CXR cases:\n\n"
        f"**L{best_start}-L{best_end}** (mean KL = {min_kl:.4f})\n\n"
        f"Lock this into `docs/extraction-spec.md` §Q1 (MedGemma row), "
        f"replacing the 'L11-22 candidate; pilot to pick best 5' "
        f"placeholder.\n\n"
        f"## Mean KL per layer\n\n"
        + "".join(f"- L{layer:2d}: {kl:.4f}"
                 + ("  *(in chosen window)*" if layer in best_window else "")
                 + "\n"
                 for layer, kl in mean_per_layer.items())
    )
    print(f"[OK] wrote {rec_path}")

    # Layer-scan plot (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(mean_per_layer.index, mean_per_layer.values, marker="o")
        ax.axvspan(best_start - 0.5, best_end + 0.5, color="tab:green",
                   alpha=0.2, label=f"recommended L{best_start}-L{best_end}")
        ax.set_xlabel("decoder layer")
        ax.set_ylabel("mean KL(attn || bbox)")
        ax.set_title("MedGemma layer scan vs VinDr-CXR bbox alignment")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "layer_kl_scan.png", dpi=120)
        plt.close(fig)
        print(f"[OK] wrote {out_dir / 'layer_kl_scan.png'}")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    data_vol.commit()

    return {
        "n_cases_used": int(df["case_idx"].nunique()),
        "skipped": int(skipped),
        "failed": int(failed),
        "n_layers_scanned": int(n_layers),
        "best_window": best_window,
        "mean_kl_in_window": min_kl,
        "mean_kl_per_layer": {int(k): float(v) for k, v in mean_per_layer.items()},
    }


# ----------------------------------------------------------------------- #
# Local entrypoint
# ----------------------------------------------------------------------- #

@app.local_entrypoint()
def main(n_cases: int = 50, seed: int = 0):
    print(f"[local] launching MedGemma layer pilot: n_cases={n_cases}, seed={seed}")
    result = run_pilot.remote(n_cases=n_cases, seed=seed)
    print("\n[local] result:")
    print(json.dumps(result, indent=2))
    print("\n[local] next step: paste the 'best_window' into "
          "docs/extraction-spec.md §Q1 (MedGemma row), then commit.")

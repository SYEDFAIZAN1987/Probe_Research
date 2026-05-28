"""02 - MedGemma per-layer attention-vs-gaze pilot.

Resolves the candidate layer range L11-22 in docs/extraction-spec.md
§Q1 to a frozen best-5 selection. Procedure:

1.  Load MedGemma-4B-IT in 4-bit nf4 on a single A40.
2.  Take 50 REFLACX cases with a frontal CXR + fixations + transcription.
3.  TEACHER-FORCED forward pass per case: feed [system + user (image +
    "Describe this X-ray") + assistant (REFLACX dictation)], capture
    output_attentions across all 32 decoder layers in one forward.
    (Faster and lower-variance than autoregressive generation for a
    layer-selection pilot.)
4.  For each layer, build a per-case 16x16 attention map by:
        a.  Slicing attention from assistant-token positions (queries)
            to image-token positions (keys).
        b.  Mean across heads.
        c.  Mean across content tokens of each sentence
            (drop punctuation + stopwords).
        d.  Mean across sentences.
        e.  Reshape from 256 image tokens to 16x16.
5.  Rasterize REFLACX fixations to 16x16 via Gaussian KDE.
6.  Compute KL(attn || gaze) per (case, layer).
7.  Aggregate: mean KL per layer, find the contiguous-5 layer window
    that minimizes mean KL. Print the recommendation; save parquet +
    a layer-vs-KL line plot.

Run:
    python scripts/02_medgemma_layer_pilot.py \\
        --reflacx-root /path/to/reflacx \\
        --mimic-jpg-root /path/to/mimic-cxr-jpg \\
        --out-dir data/pilot/medgemma \\
        --n-cases 50

After this lands, manually edit docs/extraction-spec.md §Q1's
MedGemma row to lock the chosen layer window with a citation back to
data/pilot/medgemma/layer_kl.parquet.

IMPORTANT items to verify on first GPU run (all marked TODO inline):
  - The image-token detection in find_image_token_positions().
    Confirm count == 256 and positions are contiguous.
  - That bnb 4-bit + attn_implementation="eager" lets attentions
    materialize (SDPA backend swallows attention weights silently).
  - That MedGemma's apply_chat_template lays out
    [system | user (text + image) | assistant] in that order.
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reflacx-root", required=True, type=Path)
    ap.add_argument("--mimic-jpg-root", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data/pilot/medgemma"), type=Path)
    ap.add_argument("--n-cases", default=50, type=int)
    ap.add_argument("--model-id", default="google/medgemma-4b-it")
    ap.add_argument(
        "--native-grid", default=16, type=int,
        help="MedGemma image-token grid edge. 16 = 256 tokens (default).",
    )
    ap.add_argument("--seed", default=0, type=int)
    return ap.parse_args()


# --------------------------------------------------------------------- #
# Case selection
# --------------------------------------------------------------------- #

def select_cases(
    reflacx_root: Path,
    mimic_jpg_root: Path,
    n: int,
    seed: int,
) -> list[dict]:
    """Return `n` validated case dicts with image_path, fixations_df,
    transcription_text, dicom_id. Filters to frontal-only cases that
    have non-empty fixations and a non-empty transcription."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []

    # Combine all phases. REFLACX phase distinctions don't matter for
    # the layer pilot (we just want enough variety).
    frames = []
    for phase in (1, 2, 3):
        p = reflacx_root / f"metadata_phase_{phase}.csv"
        if p.exists():
            frames.append(pd.read_csv(p).assign(_phase=phase))
    if not frames:
        raise FileNotFoundError(f"No metadata_phase_*.csv under {reflacx_root}")
    meta = pd.concat(frames, ignore_index=True)

    # Shuffle deterministically
    meta = meta.sample(frac=1, random_state=seed).reset_index(drop=True)

    main_data = reflacx_root / "main_data"
    if not main_data.exists():
        raise FileNotFoundError(f"{main_data} not found")

    for _, row in meta.iterrows():
        if len(out) >= n:
            break
        dicom_id = str(row.get("dicom_id"))
        # Locate trial dir. REFLACX trial dirs are commonly named by trial id
        # which differs from dicom_id. We look up by the column the metadata
        # provides for trial identification.
        trial_id_col = next(
            (c for c in row.index if "id" in c.lower() and c.lower() != "dicom_id"),
            None,
        )
        trial_id = str(row[trial_id_col]) if trial_id_col else dicom_id
        trial_dir = main_data / trial_id
        if not trial_dir.exists():
            continue
        fix_p = trial_dir / "fixations.csv"
        tx_p = trial_dir / "transcription.txt"
        if not (fix_p.exists() and tx_p.exists()):
            continue
        try:
            fix = pd.read_csv(fix_p)
            tx = tx_p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if fix.empty or not tx:
            continue
        # Image
        subj = row.get("subject_id")
        study = row.get("study_id")
        if pd.isna(subj) or pd.isna(study):
            continue
        sid = str(int(subj))
        img_path = (
            mimic_jpg_root / "files" / f"p{sid[:2]}" / f"p{sid}"
            / f"s{int(study)}" / f"{dicom_id}.jpg"
        )
        if not img_path.exists():
            continue
        out.append({
            "trial_id": trial_id,
            "dicom_id": dicom_id,
            "image_path": img_path,
            "fixations": fix,
            "transcription": tx,
        })
    if len(out) < n:
        print(
            f"[WARN] only {len(out)}/{n} cases passed filters; proceeding with what we have",
            file=sys.stderr,
        )
    return out


# --------------------------------------------------------------------- #
# Image-token detection
# --------------------------------------------------------------------- #

def find_image_token_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Return indices of image-placeholder tokens in input_ids[0].

    For MedGemma we expect a single contiguous block of 256 image tokens.
    The script asserts that loudly so a silent layout change can't slip
    through unnoticed.
    """
    ids = input_ids[0]
    mask = (ids == image_token_id)
    positions = mask.nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise RuntimeError(
            f"No image tokens (id={image_token_id}) found in input_ids. "
            "TODO: confirm the image-token id for this transformers version."
        )
    # Check contiguity
    if positions.numel() > 1:
        gaps = positions[1:] - positions[:-1]
        if not (gaps == 1).all():
            raise RuntimeError(
                f"Image tokens are not contiguous in input_ids: positions={positions.tolist()}"
            )
    return positions


def detect_assistant_token_range(
    input_ids: torch.Tensor,
    processor,
    last_n_to_skip: int = 0,
) -> tuple[int, int]:
    """Heuristic: assume the assistant's content begins immediately after
    the last `<start_of_turn>model\\n` (or equivalent) sequence and ends
    at the final eos. This is fragile across template versions — we use
    a defensive fallback: assume the assistant range is everything after
    the last image token plus a small offset for the assistant-role
    preamble. Confirm on first run.
    """
    # TODO confirm on first run: this is the fragile part. The cleanest
    # fix is to apply_chat_template separately for [system+user] and the
    # full [system+user+assistant], then assistant range = tokens that
    # only appear in the full version.
    image_ids = (input_ids[0] == processor.tokenizer.convert_tokens_to_ids("<image_soft_token>"))
    if image_ids.any():
        last_image = image_ids.nonzero(as_tuple=True)[0][-1].item()
    else:
        last_image = 0
    start = last_image + 1
    end = input_ids.shape[-1] - last_n_to_skip
    return start, end


# --------------------------------------------------------------------- #
# Per-sentence content-token mask
# --------------------------------------------------------------------- #

# Lazy import to keep CLI fast.
_STOPWORDS: set[str] | None = None


def get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def content_token_mask(
    assistant_token_ids: torch.Tensor,
    processor,
) -> torch.Tensor:
    """Return a boolean mask over assistant tokens: True for content
    tokens (drop punctuation + stopwords + special tokens)."""
    sw = get_stopwords()
    mask = torch.ones(assistant_token_ids.shape[-1], dtype=torch.bool)
    for i, tok_id in enumerate(assistant_token_ids.tolist()):
        tok = processor.tokenizer.decode([tok_id]).strip().lower()
        if not tok or tok in sw or all(c in string.punctuation for c in tok):
            mask[i] = False
        # Drop sub-word leading-space artifacts that are just punctuation
        if tok in {"</s>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>"}:
            mask[i] = False
    return mask


# --------------------------------------------------------------------- #
# Gaze rasterization
# --------------------------------------------------------------------- #

def rasterize_gaze_to_grid(
    fixations: pd.DataFrame,
    image_hw: tuple[int, int],
    grid_edge: int,
    sigma_frac: float = 0.03,
) -> np.ndarray:
    """Gaussian KDE rasterization of fixation points to (grid, grid).

    sigma_frac: KDE bandwidth as a fraction of image diagonal. 0.03 ≈
    ~1° visual angle for typical reading distance; refine per the
    REFLACX reference protocol once docs/extraction-spec.md §Q3 confirms
    the recommended σ.
    """
    from scipy.ndimage import gaussian_filter

    H, W = image_hw
    x_col = next(c for c in fixations.columns if c.lower().startswith("x"))
    y_col = next(c for c in fixations.columns if c.lower().startswith("y"))

    # Build a high-res hit map, smooth, then bin to grid.
    hi = np.zeros((H, W), dtype=np.float32)
    xs = fixations[x_col].clip(0, W - 1).astype(int).to_numpy()
    ys = fixations[y_col].clip(0, H - 1).astype(int).to_numpy()
    # Weight by fixation duration if available, else uniform.
    dur_col = next(
        (c for c in fixations.columns if "duration" in c.lower() or "dwell" in c.lower()),
        None,
    )
    w = fixations[dur_col].to_numpy().astype(np.float32) if dur_col else np.ones_like(xs, dtype=np.float32)
    np.add.at(hi, (ys, xs), w)

    sigma_px = sigma_frac * np.hypot(H, W)
    smoothed = gaussian_filter(hi, sigma=sigma_px)

    # Bin to grid via mean pooling
    bin_h = H // grid_edge
    bin_w = W // grid_edge
    smoothed = smoothed[: bin_h * grid_edge, : bin_w * grid_edge]
    grid = smoothed.reshape(grid_edge, bin_h, grid_edge, bin_w).mean(axis=(1, 3))
    # Normalize to a probability distribution
    grid = grid + 1e-12
    grid = grid / grid.sum()
    return grid.astype(np.float32)


# --------------------------------------------------------------------- #
# KL divergence
# --------------------------------------------------------------------- #

def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), both probability distributions on the same grid."""
    p = p + 1e-12
    q = q + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (np.log(p) - np.log(q))).sum())


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- Load model in 4-bit nf4 -------------------------------------
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"[*] loading {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",   # required to get output_attentions
    )
    model.eval()
    n_layers = model.config.text_config.num_hidden_layers
    print(f"[*] model loaded; n_decoder_layers={n_layers}")

    # Image token id — Gemma 3 uses <image_soft_token>; TODO confirm.
    try:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    except Exception:
        raise RuntimeError(
            "Could not find <image_soft_token> — confirm the actual image-placeholder "
            "token name in this version of the MedGemma processor."
        )

    # ---- Load cases --------------------------------------------------
    cases = select_cases(args.reflacx_root, args.mimic_jpg_root, args.n_cases, args.seed)
    print(f"[*] running pilot on {len(cases)} cases")

    # ---- Per-case forward and KL -------------------------------------
    rows = []
    for ci, case in enumerate(cases):
        try:
            image = Image.open(case["image_path"]).convert("RGB")
            image_hw = (image.size[1], image.size[0])  # (H, W)
            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "Describe this X-ray"},
                     {"type": "image", "image": image},
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": case["transcription"]}]},
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,   # we already supply the assistant turn
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                out = model(**inputs, output_attentions=True, return_dict=True)
            attns = out.attentions   # tuple of n_layers tensors

            # Image positions
            img_pos = find_image_token_positions(inputs["input_ids"], image_token_id)
            if img_pos.numel() != args.native_grid ** 2:
                print(
                    f"[WARN] case {ci}: expected {args.native_grid**2} image tokens, "
                    f"got {img_pos.numel()}; skipping"
                )
                continue

            # Assistant range
            a_start, a_end = detect_assistant_token_range(inputs["input_ids"], processor)
            assistant_ids = inputs["input_ids"][0, a_start:a_end]
            mask = content_token_mask(assistant_ids, processor)
            if mask.sum() == 0:
                print(f"[WARN] case {ci}: no content tokens in assistant range; skipping")
                continue

            # Gaze raster
            gaze = rasterize_gaze_to_grid(case["fixations"], image_hw, args.native_grid)

            # Per-layer KL
            for layer_idx, attn in enumerate(attns):
                # attn: (batch=1, heads, q, k)
                a = attn[0]                                            # (heads, q, k)
                a = a.mean(dim=0)                                      # (q, k)
                a = a[a_start:a_end][mask][:, img_pos]                 # (n_content, n_img)
                a = a.float().mean(dim=0)                              # (n_img,)
                grid = a.reshape(args.native_grid, args.native_grid).cpu().numpy()
                grid = grid / (grid.sum() + 1e-12)
                kl = kl_div(grid, gaze)
                rows.append({
                    "case_idx": ci,
                    "dicom_id": case["dicom_id"],
                    "layer": layer_idx,
                    "kl": kl,
                })
            if ci % 5 == 0:
                print(f"  [{ci+1}/{len(cases)}] done")
        except Exception as e:
            print(f"[ERROR] case {ci} ({case['dicom_id']}): {e}", file=sys.stderr)

    if not rows:
        print("[FATAL] no rows collected", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(args.out_dir / "layer_kl.parquet", index=False)
    print(f"[OK] wrote {args.out_dir / 'layer_kl.parquet'}")

    # ---- Pick best contiguous-5 window --------------------------------
    mean_per_layer = df.groupby("layer")["kl"].mean().sort_index()
    rolling = mean_per_layer.rolling(window=5, center=False).mean()
    best_end = int(rolling.idxmin())
    best_start = best_end - 4
    print(
        f"[REC] best contiguous-5 layer window: L{best_start}–L{best_end} "
        f"(mean KL = {rolling.min():.4f})"
    )
    print("\nMean KL per layer:")
    print(mean_per_layer.to_string())

    # ---- Plot --------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(mean_per_layer.index, mean_per_layer.values, marker="o")
        ax.axvspan(best_start - 0.5, best_end + 0.5, color="tab:green", alpha=0.2,
                   label=f"recommended L{best_start}–L{best_end}")
        ax.set_xlabel("decoder layer")
        ax.set_ylabel("mean KL(attn || gaze)")
        ax.set_title("MedGemma layer scan vs REFLACX gaze")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / "layer_kl_scan.png", dpi=120)
        plt.close(fig)
        print(f"[OK] wrote {args.out_dir / 'layer_kl_scan.png'}")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # ---- Recommendation file ------------------------------------------
    rec_path = args.out_dir / "recommendation.md"
    rec_path.write_text(
        f"# MedGemma layer-pilot recommendation\n\n"
        f"Best contiguous-5 layer window by mean KL(attn || gaze) on "
        f"{df['case_idx'].nunique()} REFLACX cases:\n\n"
        f"**L{best_start}–L{best_end}** (mean KL = {rolling.min():.4f})\n\n"
        f"Lock this into `docs/extraction-spec.md` §Q1 (MedGemma row).\n"
    )
    print(f"[OK] wrote {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""03b — LLaVA-Med prompt-sensitivity check.

Replays the 03_llavamed_layer_pilot.py recipe on the **same 50 cases**
(same seed), but swaps the short synthetic teacher-forcing template
for a longer Q&A-style assistant turn that more closely matches
LLaVA-Med's instruction-tuning distribution.

**Question being asked:** is LLaVA-Med's apparent near-random
attention-bbox alignment (see /data/pilot/llavamed/random_summary.md
— L0-L4 below random p95 on CC, NSS, IoU) a real property of the
model, or an artifact of the short "There is X visible..." prompt
the original pilot used?

**Decision rule on completion:**
  - If the naturalistic prompt's L0-L4 metrics also fall below the
    24×24 random p95 → the finding is robust to prompt choice. The
    paper bullet "LLaVA-Med's cross-attention is statistically
    indistinguishable from uniform-random against radiologist bbox
    locations" stands.
  - If the naturalistic prompt clears p95 on CC/NSS where the short
    one didn't → original was prompt-artifact. Report the
    naturalistic numbers as the "real" finding and note the
    sensitivity in §7 limitations.

Differences from 03_llavamed_layer_pilot.py:
  - Inline subclass `LLaVAMedNaturalisticExtractor` overrides
    prepare_inputs to construct a longer prompt
  - User turn is a question rather than an instruction
  - Assistant turn embeds the class label in a 3-sentence clinical
    statement instead of a single 6-word template
  - Outputs land under /data/pilot/llavamed_naturalistic/

Run:
    modal run scripts/modal/03b_llavamed_prompt_sensitivity.py

Expected cost: ~$0.45. Expected runtime: ~10-12 min on L40S.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


# ----------------------------------------------------------------------- #
# Modal app + image
# ----------------------------------------------------------------------- #

app = modal.App("gazeprobe-llavamed-prompt-sensitivity")

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
    from src.attn.base import TeacherForcedInputs
    from src.metrics.alignment import auc_attn_gaze, cc, iou_topk, kl_div, nss
    from src.metrics.rasterize import rasterize_bbox_to_grid

    # ------------------------------------------------------------------ #
    # Naturalistic-prompt extractor — same class, different prep.
    # ------------------------------------------------------------------ #
    class LLaVAMedNaturalisticExtractor(LLaVAMedExtractor):
        """Subclass that uses a longer, Q&A-style prompt for the
        teacher-forcing pass. Inherits all the post-expansion
        bookkeeping from the parent's extract_teacher_forced."""

        # Q&A user turn: a question rather than an instruction.
        USER_TEXT = (
            "What is the main abnormality visible in this chest "
            "radiograph? Please describe what you observe and any "
            "clinically significant features."
        )

        @staticmethod
        def _build_assistant_text(class_name: str) -> str:
            # 3-sentence clinical statement embedding the class name
            # twice. Matches LLaVA-Med training-distribution style
            # better than the single-sentence original.
            return (
                f"The main abnormality visible in this chest radiograph "
                f"is {class_name}. The {class_name} can be identified "
                f"upon careful examination of the relevant anatomical "
                f"regions. This finding is clinically significant and "
                f"warrants further evaluation in the appropriate "
                f"clinical context."
            )

        def prepare_inputs(self, image, ground_truth_report):
            # ground_truth_report here is the class_name passed by the
            # main loop — the assistant text is built from it.
            class_name = ground_truth_report
            assistant_text = self._build_assistant_text(class_name)

            prompt_user = f"[INST] <image>\n{self.USER_TEXT} [/INST]"
            eos = self.processor.tokenizer.eos_token or "</s>"
            prompt_full = f"{prompt_user} {assistant_text}{eos}"

            inputs_user = self.processor(
                images=[image], text=prompt_user, return_tensors="pt",
            )
            inputs_full = self.processor(
                images=[image], text=prompt_full, return_tensors="pt",
            ).to(self.model.device, dtype=torch.bfloat16)

            ids_full = inputs_full["input_ids"][0]
            ids_user = inputs_user["input_ids"][0].to(ids_full.device)
            prefix = min(ids_full.numel(), ids_user.numel())
            for i in range(prefix):
                if ids_full[i].item() != ids_user[i].item():
                    prefix = i
                    break
            if prefix == 0:
                raise RuntimeError(
                    "Naturalistic prompt: assistant-range diff produced "
                    "no common prefix. Inspect the prompt construction."
                )
            end = ids_full.numel()
            eos_id = self.processor.tokenizer.eos_token_id
            while end > prefix and ids_full[end - 1].item() == eos_id:
                end -= 1

            img_token_id = self._resolve_image_token_id()
            img_pos_pre = (ids_full == img_token_id).nonzero(as_tuple=True)[0]
            if img_pos_pre.numel() == 0:
                raise RuntimeError(
                    f"Naturalistic prompt: no image-token in input_ids."
                )

            return TeacherForcedInputs(
                inputs=dict(inputs_full),
                image_token_positions=img_pos_pre,
                assistant_range=(prefix, end),
            )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Phase 1 — Same case selection as 03
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
    print(f"[*] sampled {n_take} cases (seed={seed} — same as 03)")

    # ------------------------------------------------------------------ #
    # Phase 2 — Load via naturalistic subclass
    # ------------------------------------------------------------------ #
    print("[*] loading LLaVA-Med-1.5 (naturalistic prompt variant)...")
    extractor = LLaVAMedNaturalisticExtractor()
    extractor.load()
    n_layers = extractor.model.config.text_config.num_hidden_layers \
        if hasattr(extractor.model.config, "text_config") \
        else extractor.model.config.num_hidden_layers
    print(f"[*] model loaded; n_decoder_layers = {n_layers}")

    smoke_img = Image.new("RGB", (336, 336), color=(127, 127, 127))
    smoke_result = extractor.extract_teacher_forced(smoke_img, "consolidation")
    if smoke_result.attentions is None or len(smoke_result.attentions) == 0:
        raise RuntimeError("smoke forward returned no attentions")

    img_pos_count = smoke_result.image_token_positions.numel()
    expected_img = extractor.native_grid[0] * extractor.native_grid[1]
    a_start_smoke, a_end_smoke = smoke_result.assistant_range
    smoke_assistant_len = a_end_smoke - a_start_smoke
    print(f"[*] smoke forward OK: {len(smoke_result.attentions)} layers, "
          f"img_pos count = {img_pos_count} (expected {expected_img}), "
          f"assistant_len = {smoke_assistant_len} tokens "
          f"(original 03 was ~12; longer naturalistic prompt should be ~40+)")

    # ------------------------------------------------------------------ #
    # Phase 3 — Same extraction loop as 03
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

            # Pass class_name as the "ground_truth_report" param; the
            # naturalistic subclass uses it to construct the assistant
            # text.
            result = extractor.extract_teacher_forced(image, class_name)

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
        raise RuntimeError("zero rows collected — re-check naturalistic prep")

    print(f"\n[*] phase 3 done. rows: {len(rows)}, "
          f"skipped: {skipped}, failed: {failed}")

    # ------------------------------------------------------------------ #
    # Phase 4 — Same aggregation + comparison to original LLaVA-Med
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(rows)
    out_dir = Path("/data/pilot/llavamed_naturalistic")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "layer_kl.parquet", index=False)

    metric_cols = ["kl", "iou", "auc", "cc", "nss"]
    per_layer = df.groupby("layer")[metric_cols].mean().sort_index()

    iou_rolling = per_layer["iou"].rolling(window=5, center=False).mean()
    best_end = int(iou_rolling.idxmax())
    best_start = best_end - 4
    best_window = list(range(best_start, best_end + 1))
    best_iou = float(iou_rolling.max())

    print(f"\n[REC] best window by IoU (primary): "
          f"L{best_start}-L{best_end} (mean IoU = {best_iou:.4f})")

    print("\nPer-layer means (naturalistic prompt):")
    print(per_layer.to_string(float_format="%.4f"))

    # ------------------------------------------------------------------ #
    # Side-by-side comparison against the original LLaVA-Med pilot
    # ------------------------------------------------------------------ #
    orig_path = Path("/data/pilot/llavamed/layer_kl.parquet")
    print()
    print("=" * 80)
    print("L0-L4 comparison: original short prompt vs naturalistic Q&A")
    print("=" * 80)
    if orig_path.exists():
        orig_df = pd.read_parquet(orig_path)
        orig_L0_L4 = orig_df[orig_df["layer"].isin([0, 1, 2, 3, 4])]
        new_L0_L4 = df[df["layer"].isin([0, 1, 2, 3, 4])]
        comparison = {}
        print(f"{'metric':>6} | {'short prompt':>13} | {'naturalistic':>13} | {'delta':>8}")
        print("-" * 60)
        for m in metric_cols:
            orig_mean = float(orig_L0_L4[m].mean())
            new_mean = float(new_L0_L4[m].mean())
            delta = new_mean - orig_mean
            comparison[m] = {
                "short_prompt": orig_mean,
                "naturalistic": new_mean,
                "delta": delta,
            }
            print(f"{m:>6} | {orig_mean:>13.4f} | {new_mean:>13.4f} | {delta:>+8.4f}")
    else:
        comparison = None
        print("[WARN] original LLaVA-Med pilot parquet not found at "
              f"{orig_path} — no side-by-side comparison")

    # Also compare to the 24×24 random baseline
    rand_path = Path("/data/pilot/llavamed/random_baseline.parquet")
    rand_comparison = {}
    if rand_path.exists():
        rand_df = pd.read_parquet(rand_path)
        print()
        print("=" * 80)
        print("Naturalistic L0-L4 vs 24×24 random baseline (p95)")
        print("=" * 80)
        print(f"{'metric':>6} | {'naturalistic':>13} | {'random p95':>11} | {'verdict':>20}")
        print("-" * 70)
        for m in metric_cols:
            new_mean = float(df[df["layer"].isin([0, 1, 2, 3, 4])][m].mean())
            p95 = float(rand_df[m].quantile(0.95))
            higher_better = m != "kl"
            if higher_better:
                verdict = "above p95 (real)" if new_mean > p95 else "below p95 (random)"
            else:
                verdict = "above p95 (real)" if new_mean < p95 else "below p95 (random)"
            rand_comparison[m] = {
                "naturalistic": new_mean,
                "random_p95": p95,
                "verdict": verdict,
            }
            print(f"{m:>6} | {new_mean:>13.4f} | {p95:>11.4f} | {verdict:>20}")

    sensitivity_path = out_dir / "sensitivity_summary.md"
    md_lines = [
        "# LLaVA-Med prompt sensitivity check",
        "",
        "Compares the L0-L4 metrics from two teacher-forcing prompts:",
        "",
        "- **short**: original `03_llavamed_layer_pilot.py` —",
        "  `There is {class_name} visible in this chest radiograph.`",
        "- **naturalistic**: this script — multi-sentence Q&A that matches",
        "  LLaVA-Med's instruction-tuning distribution more closely.",
        "",
        "## Original vs naturalistic",
        "",
        "| metric | short prompt | naturalistic | delta |",
        "|--------|--------------|--------------|-------|",
    ]
    if comparison:
        for m, c in comparison.items():
            md_lines.append(
                f"| {m} | {c['short_prompt']:.4f} | "
                f"{c['naturalistic']:.4f} | {c['delta']:+.4f} |"
            )

    if rand_comparison:
        md_lines += [
            "",
            "## Naturalistic L0-L4 vs 24×24 random p95",
            "",
            "| metric | naturalistic | random p95 | verdict |",
            "|--------|--------------|------------|---------|",
        ]
        for m, c in rand_comparison.items():
            md_lines.append(
                f"| {m} | {c['naturalistic']:.4f} | "
                f"{c['random_p95']:.4f} | {c['verdict']} |"
            )

    sensitivity_path.write_text("\n".join(md_lines))
    print(f"\n[OK] wrote {sensitivity_path}")

    data_vol.commit()

    return {
        "n_cases_used": int(df["case_idx"].nunique()),
        "skipped": int(skipped),
        "failed": int(failed),
        "n_layers_scanned": int(per_layer.shape[0]),
        "smoke_assistant_len_tokens": int(smoke_assistant_len),
        "best_window": best_window,
        "mean_iou_in_window": best_iou,
        "vs_short_prompt": comparison,
        "vs_random_p95": rand_comparison,
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
    print(f"[local] launching LLaVA-Med prompt-sensitivity pilot: "
          f"n_cases={n_cases}, seed={seed}")
    result = run_pilot.remote(n_cases=n_cases, seed=seed)
    print("\n[local] result:")
    print(json.dumps(result, indent=2))
    print("\n[local] interpretation:")
    print("  - If 'vs_random_p95' shows AUC/CC/NSS still below random p95 →")
    print("    near-random finding is ROBUST to prompt choice. Publishable.")
    print("  - If naturalistic prompt clears p95 where short didn't →")
    print("    original was prompt-artifact. Use naturalistic numbers as 'real'.")

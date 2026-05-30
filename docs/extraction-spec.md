# Attention-extraction spec — BBoxProbe v1

Resolves the open methodology questions flagged in
[`paper/skeleton.md`](../paper/skeleton.md). Frozen for week 2
implementation; revisit only with explicit reason.

**Pivot note (2026-05-29):** this project pivoted from GazeProbe v2
(REFLACX gaze) to BBoxProbe v1 (VinDr-CXR bbox) to escape the
PhysioNet credentialing wait. The gaze-version spec is preserved
at git tag `gaze-v0.1`. The methodology decisions below were
updated where the ground-truth signal change required it; the rest
carry over verbatim.

## Summary

| # | Question | Decision |
|---|---|---|
| 1 | Which attention to extract, from which layer? | Text→image self-attention from LLM decoder layers, mid-third (LLaVA-Med L14–18, MedGemma L11–22, MAIRA-2 L14–18). Mean across heads. Default = "layer mean of best 5 layers"; calibrate per model on a 50-case pilot in week 2. |
| 2 | Per-finding aggregation? | **Autoregressive generation** + per-finding-sentence aggregation. Each model generates a report; we identify sentences mentioning a VinDr disease class and align attention from that sentence's content tokens against the bbox for that class. Content-token mean (drop punctuation + stopwords) within each finding-sentence. |
| 3 | Bbox rasterization grid? | Both: model-native grid AND a common 56×56 cross-model grid. Bbox → binary mask (filled rectangle) → renormalized to a probability distribution for the KL metric, kept as a binary mask for IoU/AUC. |
| 4 | RadGraph version? | **RadGraph-XL** (Delbrouck et al., ACL Findings 2024). VinDr-CXR has no reference reports, so RadGraph-F1 is computed model-vs-model (e.g. MAIRA-2 as anchor) and against the VinDr class labels reduced to a canonical-template "reference report." Note this caveat in the paper. |
| 5 | Hallucination metric? | **RaTEScore + RadGraph-XL** as primary metrics. Skip RadNLI (not used by closest baseline). Skip human eval (out of scope). Report CheXbert-F1 as a secondary number for older-literature comparability with a footnoted caveat. |
| 6 | Compute precision? | **bf16 on L40S 48 GB (or larger)**. No bitsandbytes 4-bit. All three models fit comfortably: LLaVA-Med 7B ≈ 14 GB, MedGemma 4B ≈ 8 GB, MAIRA-2 7B ≈ 14 GB. Decided 2026-05-28 after Modal smoke test confirmed L40S availability. |

Plus one bonus decision driven by the MAIRA-2 architecture:

| # | Bonus | Decision |
|---|---|---|
| 0 | MAIRA-2 bbox output as second grounding signal? | **Yes — even cleaner under BBoxProbe.** MAIRA-2 emits explicit bounding-box tokens. We report (a) MAIRA-2 attention ↔ VinDr radiologist bbox, and (b) MAIRA-2 emitted bbox ↔ VinDr radiologist bbox. Both signals are now in the same format (bbox vs bbox) → direct IoU comparison. The side-finding "is MAIRA-2's emitted bbox better-aligned with the radiologist than its own attention is?" is itself publishable. |

---

## Q1: Which attention to extract, from which layer?

**Decision.** Extract **text-token-to-image-token self-attention** from the
LLM decoder (these are LLaVA-family models — there is no separate
cross-attention block; visual tokens are prepended to the text token
sequence and seen via causal self-attention). For each model, default to
the **mean of the middle 5 layers** identified below, averaging across
all heads. Report layer-mean as the primary number; also report each
layer's standalone alignment in the appendix for transparency.

**Per-model defaults:**

| Model | LLM | Decoder layers | Default layer set | Why |
|---|---|---|---|---|
| LLaVA-Med-1.5 | Vicuna-7B | 32 | **14–18** | Mechanistic interpretability work on LLaVA-1.5-7B identifies localization heads concentrated at L14–18; specifically L14H13 and L14H24 ([arxiv 2503.06287](https://arxiv.org/html/2503.06287), [arxiv 2411.10950](https://arxiv.org/html/2411.10950v1)). |
| MedGemma-4B | Gemma-3-4B | **34** (corrected from earlier prior of 32) | **L0–L4** (frozen 2026-05-30 by 50-case pilot + random-baseline sanity check). | All 5 metrics independently picked L0–L4. AUC 0.685 vs random p95 0.656 → above-random; CC 0.165 vs random p95 0.108 → above-random; NSS 0.95 vs random p95 0.50 → ~2× above-random. IoU 0.045 is at the 76.5th percentile of random — meaningfully above random mean but below the random p95, indicating "broad-but-correct" attention (positively correlated with bbox location, but not tightly peaked). This is *opposite* of the LLaVA-1.5 prior of L14–18; report as a paper bullet. See `data/pilot/medgemma/{layer_window.json,random_summary.md}`. |
| MAIRA-2 | Vicuna-7B | 32 | **14–18** | LLaVA-family; same prior as LLaVA-Med. RAD-DINO vision encoder doesn't change the LLM-side attention pattern. |

**Why not single-best-head.** Single localization heads (L14H24 etc.)
have been reported in interpretability papers, but (a) they're identified
on natural images, not CXRs, (b) using a single head conflates "what we
extract" with "what we tune the extraction to," which weakens the
faithfulness claim. Layer-mean is more conservative and reproducible.

**Implementation.** Use `transformers` model loaded in **bf16** with
`attn_implementation="eager"` and `output_attentions=True`. The SDPA
backend silently drops attention weights — "eager" is the required
workaround regardless of precision. Extract attention matrices of
shape `[heads, query_tokens, key_tokens]`, slice to
`query=text_tokens, key=image_tokens`. See `src/attn/`.

**Open pilot for week 2.** For MedGemma specifically: run extraction on
50 VinDr-CXR cases at *every* layer (1–32), compute alignment-with-
bbox-mask KL per layer, pick the best 5 contiguous, freeze. Document
the selected layer set in this file before running the full
2,000-case pass.

---

## Q2: Per-finding aggregation

**Decision.** Each model generates a report autoregressively for each
case. We identify finding-sentences and align attention from each
finding-sentence's content tokens against the bbox for the
corresponding VinDr disease class.

1. Generate the model's report (greedy, `do_sample=False`).
2. Split into sentences via scispacy `en_core_sci_sm`.
3. For each sentence, match against the 14 VinDr disease-class
   keywords (with a small synonym table per class — e.g.
   "consolidation" matches "airspace opacity," "infiltrate").
4. For each matched (sentence, class) pair, extract attention from the
   sentence's content tokens (drop punctuation + stopwords + special)
   to the image tokens. Mean across content tokens, mean across the
   spec-frozen mid-layer set, mean across heads.
5. Compare against the VinDr bbox for that class (multi-rater union
   by default; report per-rater spread as appendix).

**Sentences with no matched class** still receive an extraction (used
in qualitative figures and a "generic-attention baseline") but don't
enter the per-pathology table.

**Why generate, not teacher-force.** VinDr-CXR has no reference
dictation; teacher-forcing needs SOMETHING as the assistant content.
The alternatives (synthetic template, gold-class-derived report) bias
attention toward terms the model wouldn't otherwise produce. Free
generation measures attention as it actually deploys.

**Cost implication.** Generation is ~3× slower than teacher-forced
forward. To stay within the Modal $30/month budget we **subset to
~2,000 cases per model** (drawn from the 18k by stratified sampling
across disease classes to keep per-class N reasonable). Statistical
power for cross-model comparison and per-class breakdown remains
strong for common classes; rare classes (Pneumothorax, Pleural
thickening) thin out — acknowledge in §7 limitations.

**Why not the period/EOS token.** Sentence-final tokens are
well-known attention sinks. Same reasoning as GazeProbe v2.

**Why content-token-mean.** Symmetric across pathologies, requires no
per-class weighting, robust to sentence-length variance.

---

## Q3: Bbox rasterization grid

**Decision.** Report alignment at **two grids in parallel**:

1. **Native grid.** Each model gets its own grid matching its vision
   encoder's token layout. Attention maps are reshaped to 2D at this
   grid; bbox is rasterized to the same grid as a filled-rectangle
   binary mask.
2. **Common grid (56×56).** Both attention and bbox mask upsampled
   (bilinear / nearest, respectively) to 56×56. Enables direct
   cross-model comparison.

**Per-model native grid:**

| Model | Vision encoder | Patch geometry | Native grid |
|---|---|---|---|
| LLaVA-Med-1.5 | CLIP-ViT-L/14 @ 336² | 24×24 patches | **24×24** |
| MedGemma-4B | MedSigLIP-400M @ 896² | 256 tokens = 16×16 | **16×16** |
| MAIRA-2 | RAD-DINO ViT-B (verify patch size on load) | TBD | **TBD — confirm in week 1** |

**Bbox rasterization.** Rectangles in original-image coordinates
(VinDr CSV ships `x_min, y_min, x_max, y_max` per finding) → resize
to grid by setting every grid cell whose center lies inside the
rectangle to 1, others to 0. For KL we renormalize to a probability
distribution; for IoU and Pointing-Game we keep the binary mask.

**Multi-rater bbox handling.** VinDr-CXR ships up to 3 radiologists per
image per finding. Default: **union mask** (any rater's bbox counts).
Report per-rater spread as an appendix table. The union choice is
generous to the model and conservative against an "audit overclaims"
critique.

**Why both grids.** Native grid keeps the model's representation
honest; common grid keeps the cross-model table apples-to-apples.

**Implementation.** `src/metrics/rasterize.py`. Two helpers:
```python
def rasterize_bbox_to_grid(bboxes, image_hw, grid_edge) -> np.ndarray:  # [G, G]
def rasterize_gaze_to_grid(fixations, ...) -> np.ndarray:               # kept for future
def reshape_attention_to_grid(attn_1d, native_hw) -> np.ndarray:
def upsample_to_common_grid(grid_2d, target_hw=(56, 56)) -> np.ndarray:
```

---

## Q4: RadGraph version

**Decision.** **RadGraph-XL** ([Delbrouck et al., ACL Findings
2024](https://aclanthology.org/2024.findings-acl.765/)) via the
`radgraph` PyPI package or the Stanford-AIMI GitHub release.

**Why XL over the original.**
- Look & Mark (our closest baseline) uses RadGraph-XL. Matching is
  necessary for comparability.
- Higher inter-annotator agreement than the original RadGraph
  (Jain et al. 2021).
- Better calibrated for multi-anatomy reports (CXR + CT/MR);
  CXR-only behavior is the relevant slice here.

**Caveat.** CheXpert-F1 also reported as a secondary number for older
report-generation papers' comparability, **footnoted** with the Look &
Mark observation that CheXpert-F1 correlates poorly with expert
ratings on multi-sentence reports.

**Implementation.** `src/eval/radgraph.py`. Cache per-report parses to
disk; never re-parse.

---

## Q5: Hallucination metric

**Decision.** **RaTEScore** ([Zhao et al., 2024](https://arxiv.org/abs/2406.16845))
as the primary hallucination-sensitive metric, in addition to RadGraph-XL F1.

**Why RaTEScore over RadNLI.** RaTEScore is what Look & Mark uses; it
explicitly handles clinical entities, negations, and synonyms — the
exact failure modes of a hallucinating CXR-report generator. RadNLI is
older, pair-classification only, and tuned to a now-narrow distribution.

**Skip human eval.** Out of scope for a single-author 8-week probe.
Acknowledge as a limitation in §7.

**Skip BLEU/ROUGE.** Multiple 2024–2026 papers confirm these correlate
poorly with expert ratings for radiology. Do not report.

**Implementation.** `src/eval/ratescore.py` (HF model). Cache scores per
report to disk.

---

## Q0 (bonus): MAIRA-2 bbox output

**Decision.** Treat MAIRA-2's generated bbox tokens as a **second
grounding signal**, audited independently against the VinDr radiologist
bbox.

**Why.** MAIRA-2 is the only one of the three models that produces an
explicit, model-internal localization output via its 100×100 tokenized
bbox grid. Skipping this would mean evaluating MAIRA-2 *only* through
attention while ignoring the signal the model itself thinks is most
important for grounding.

**What this adds to the headline tables.**
- Table 2 (models × metric × baseline) gets a column for "MAIRA-2 (attn)"
  and a column for "MAIRA-2 (bbox)" — same model, two grounding signals.
- Both columns are now in the same FORMAT as the ground truth (bbox).
  IoU between MAIRA-2's emitted bbox and the VinDr radiologist bbox is
  a direct, naturally-interpretable number.
- Table 2 caption explicitly asks: **is MAIRA-2's emitted bbox better-
  aligned with the radiologist bbox than its own attention is?** This
  is itself a publishable side-finding, easier to land than the analogous
  GazeProbe question (which compared bbox to gaze — different formats).

**Implementation.** MAIRA-2's generate call emits bbox tokens in the
report string. Parse them, rasterize as a filled rectangle on the
native grid, and run the same alignment metrics as for the
attention-derived map.

---

## What this spec does NOT decide

- ~~**4-bit vs. fp16 attention delta.**~~ **Resolved 2026-05-28** by
  Q6 above: bf16 on L40S 48 GB. No quantization, no concern.
- **Exact prompt template per model.** Week 1 work; each model has a
  preferred CXR-report prompt format from its model card / paper. Use
  the model-card prompt verbatim, document in `docs/prompts.md`.
- **What counts as a "sentence."** Use scispaCy `en_core_sci_sm` for
  sentence splitting — handles "1.5cm" / "T2-weighted" / abbreviations
  better than default sentencizers. Document the version pin.
- **VinDr class → keyword synonym table.** Week 1 work. Each of the 14
  VinDr classes gets a small set of clinical synonyms (e.g.
  "Consolidation" matches "consolidation", "airspace opacity",
  "infiltrate"). Commit the table to `docs/vindr-class-synonyms.md`
  so the matching is auditable and reviewable.
- **Stratified 2k-case subset.** Week 1 work. Sample 2,000 cases per
  model so each of the 14 classes is represented, with oversampling
  for rare classes (Pneumothorax, Pleural thickening) capped at
  available N. Document the seed and the per-class N in
  `docs/subset-sampling.md`.

---

## References

### Methodology
- [arxiv 2503.06287](https://arxiv.org/html/2503.06287) — Your LVLM Only Needs A Few Attention Heads For Visual Grounding
- [arxiv 2411.10950](https://arxiv.org/html/2411.10950v1) — Mechanistic Interpretability of LLaVA in VQA
- [arxiv 2403.06764](https://arxiv.org/html/2403.06764v3) — Image Is Worth 1/2 Tokens After Layer 2 (attention-sink evidence)

### Models
- [arxiv 2406.04449](https://arxiv.org/html/2406.04449v1) — MAIRA-2 paper
- [arxiv 2507.05201](https://arxiv.org/html/2507.05201v2) — MedGemma Technical Report

### Dataset
- [Nguyen et al., Sci Data 2022](https://www.nature.com/articles/s41597-022-01498-w) — VinDr-CXR dataset paper
- [Kaggle competition release](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) — our actual data source

### Closest scoop-check threats (cleared 2026-05-29)
- [arxiv 2510.11196](https://arxiv.org/abs/2510.11196) — Reasoning faithfulness via perturbations. *Different signal class.*
- [arxiv 2510.19599](https://arxiv.org/abs/2510.19599) — XBench: CLIP-style VLMs only. *Different model class.*
- [arxiv 2511.00504](https://arxiv.org/abs/2511.00504) — VinDr-CXR-VQA: new dataset, single-model bench. *Different scope.*
- [arxiv 2505.22222](https://arxiv.org/html/2505.22222) — Look & Mark: gaze+bbox prompt-level ICL intervention, not an audit. *Different framework.*

### Eval
- [Delbrouck et al., ACL Findings 2024](https://aclanthology.org/2024.findings-acl.765/) — RadGraph-XL
- [arxiv 2406.16845](https://arxiv.org/abs/2406.16845) — RaTEScore

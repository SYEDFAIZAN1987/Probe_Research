# Attention-extraction spec

Resolves the five open methodology questions flagged in
[`paper/skeleton.md`](../paper/skeleton.md). Frozen for week 2
implementation; revisit only with explicit reason.

## Summary

| # | Question | Decision |
|---|---|---|
| 1 | Which attention to extract, from which layer? | Text→image self-attention from LLM decoder layers, mid-third (LLaVA-Med L14–18, MedGemma L11–22, MAIRA-2 L14–18). Mean across heads. Default = "layer mean of best 5 layers"; calibrate per model on a 50-case pilot in week 2. |
| 2 | Per-sentence aggregation? | Mean over content tokens of the sentence (drop punctuation and stopwords). Sanity-check against "finding-noun tokens only" on a 100-case subset, report delta in appendix. |
| 3 | Gaze rasterization grid? | Both: model-native grid AND a common 56×56 cross-model grid. Bilinear upsampling; gaze rasterized via Gaussian KDE with σ from the REFLACX reference protocol. |
| 4 | RadGraph version? | **RadGraph-XL** (Delbrouck et al., ACL Findings 2024) — matches Look & Mark for comparability, avoids CheXpert-F1's expert-correlation issues. |
| 5 | Hallucination metric? | **RaTEScore + RadGraph-XL** as primary metrics. Skip RadNLI (not used by closest baseline). Skip human eval (out of scope). Report CheXbert-F1 as a secondary number for older-literature comparability with a footnoted caveat. |

Plus one bonus decision driven by the MAIRA-2 architecture:

| # | Bonus | Decision |
|---|---|---|
| 0 | MAIRA-2 bbox output as second grounding signal? | **Yes.** MAIRA-2 emits explicit bounding-box tokens during generation. Treat these as a parallel grounding channel: report (a) attention↔gaze alignment and (b) bbox↔gaze alignment separately. This is a free methodological lever — MAIRA-2 gets two signal columns in the headline table. |

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
| MedGemma-4B | Gemma-3-4B | 32 (h=4096, 32 heads) | **11–22 candidate; pilot to pick best 5** | No published mechanistic study on Gemma yet. Default to a wider mid-band, then run a 50-case pilot in week 2 to pick the best 5 by alignment-with-gaze score. Document the choice and lock it. |
| MAIRA-2 | Vicuna-7B | 32 | **14–18** | LLaVA-family; same prior as LLaVA-Med. RAD-DINO vision encoder doesn't change the LLM-side attention pattern. |

**Why not single-best-head.** Single localization heads (L14H24 etc.)
have been reported in interpretability papers, but (a) they're identified
on natural images, not CXRs, (b) using a single head conflates "what we
extract" with "what we tune the extraction to," which weakens the
faithfulness claim. Layer-mean is more conservative and reproducible.

**Implementation.** Use `transformers` model with
`output_attentions=True` plus a forward-hook fallback for any model
that lazy-prunes attention outputs in 4-bit mode. Extract attention
matrices of shape `[heads, query_tokens, key_tokens]`, slice to
`query=text_tokens, key=image_tokens`. See `src/attn/` (to be
implemented week 2).

**Open pilot for week 2.** For MedGemma specifically: run extraction on
50 REFLACX cases at *every* layer (1–32), compute alignment-with-gaze
KL per layer, pick the best 5 contiguous, freeze. Document the
selected layer set in this file before running the full 3,032-case pass.

---

## Q2: Per-sentence aggregation

**Decision.** For each generated sentence in the report, extract one
attention map per (model, sentence) by:

1. Tokenize the generated sentence with the model's tokenizer.
2. Drop punctuation tokens and English stopwords (NLTK list +
   model-specific special tokens like `<bos>`, `<image>`, `<eos>`).
3. Take the **mean attention** of remaining content tokens.

**Why not the period/EOS token.** Sentence-final tokens are well-known
attention sinks; using them as a single signal biases toward whatever
the model happens to "park" attention on at sentence boundaries.

**Why not just the noun finding token (e.g., "pneumonia").** Requires
a per-pathology keyword list, which biases evaluation toward the very
pathologies REFLACX labels — a circular setup. We run this as a
sanity-check ablation on 100 cases (drop-stopwords vs. noun-only) and
report the delta in the appendix; if the delta is large, lift the
ablation into the main results.

**Why content-token-mean.** It's symmetric across pathologies, requires
no keyword list, and is robust to sentence-length variance because it's
a mean rather than a sum.

---

## Q3: Gaze rasterization grid

**Decision.** Report alignment at **two grids in parallel**:

1. **Native grid.** Each model gets its own grid matching its vision
   encoder's token layout. Attention maps are reshaped to 2D at this
   grid; gaze is rasterized to the same grid via Gaussian KDE.
2. **Common grid (56×56).** Both attention and gaze upsampled
   (bilinear) to 56×56. Enables direct cross-model comparison.

**Per-model native grid:**

| Model | Vision encoder | Patch geometry | Native grid |
|---|---|---|---|
| LLaVA-Med-1.5 | CLIP-ViT-L/14 @ 336² | 24×24 patches | **24×24** |
| MedGemma-4B | MedSigLIP-400M @ 896² | 256 tokens = 16×16 | **16×16** |
| MAIRA-2 | RAD-DINO ViT-B (verify patch size on load) | TBD | **TBD — confirm in week 1** |

**Gaussian KDE bandwidth.** Use the σ recommended by the REFLACX
reference protocol (Lanfredi et al., 2022). If their codebase ships a
helper, call it directly. If not, σ corresponding to ~1° of visual
angle at typical reading distance is the eye-tracking-literature
default.

**Why both grids.** Native grid keeps the model's representation
honest; common grid keeps the cross-model table apples-to-apples.
Reporting only native makes cross-model alignment numbers
non-comparable; reporting only common discards model-specific
geometric fidelity.

**Implementation.** `src/metrics/rasterize.py`. Function signature:
```python
def rasterize_gaze(fixations, grid_hw, sigma=...) -> np.ndarray:  # [H, W]
def reshape_attention(attn_1d, native_hw) -> np.ndarray:           # [H, W]
def to_common_grid(attn_2d, target_hw=(56, 56)) -> np.ndarray:
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
grounding signal**, audited independently against gaze.

**Why.** MAIRA-2 is the only one of the three models that produces an
explicit, model-internal localization output via its 100×100 tokenized
bbox grid. Skipping this would mean evaluating MAIRA-2 *only* through
attention while ignoring the signal the model itself thinks is most
important for grounding. That weakens the paper.

**What this adds to the headline tables.**
- Table 2 (models × metric × baseline) gets a column for "MAIRA-2 (attn)"
  and a column for "MAIRA-2 (bbox)" — same model, two grounding signals.
- Table 2 caption explicitly compares the two: is MAIRA-2's emitted
  bbox a more faithful gaze-aligned signal than its internal attention?
  This is itself a publishable side-finding.

**Implementation.** Already partly free: MAIRA-2's generate call emits
the bbox tokens in the report string. Parse them, rasterize as a
filled rectangle on the native grid, and run the same alignment
metrics as for attention.

---

## What this spec does NOT decide

- **4-bit vs. fp16 attention delta.** Open empirical question — does
  bitsandbytes nf4 quantization shift the attention distribution enough
  to invalidate the audit? Plan: run 100 cases of LLaVA-Med in both
  precisions in week 2; report the delta as an appendix table. If
  the delta is large, repeat the full pass in fp16 (budget allows
  if needed).
- **Exact prompt template per model.** Week 1 work; each model has a
  preferred CXR-report prompt format from its model card / paper. Use
  the model-card prompt verbatim, document in `docs/prompts.md`.
- **What counts as a "sentence."** Use scispaCy `en_core_sci_sm` for
  sentence splitting — handles "1.5cm" / "T2-weighted" / abbreviations
  better than default sentencizers. Document the version pin.

---

## References

- [arxiv 2503.06287](https://arxiv.org/html/2503.06287) — Your LVLM Only Needs A Few Attention Heads For Visual Grounding
- [arxiv 2411.10950](https://arxiv.org/html/2411.10950v1) — Mechanistic Interpretability of LLaVA in VQA
- [arxiv 2403.06764](https://arxiv.org/html/2403.06764v3) — Image Is Worth 1/2 Tokens After Layer 2 (attention-sink evidence)
- [arxiv 2406.04449](https://arxiv.org/html/2406.04449v1) — MAIRA-2 paper
- [arxiv 2507.05201](https://arxiv.org/html/2507.05201v2) — MedGemma Technical Report
- [arxiv 2505.22222](https://arxiv.org/html/2505.22222) — Look & Mark (closest baseline)
- [Delbrouck et al., ACL Findings 2024](https://aclanthology.org/2024.findings-acl.765/) — RadGraph-XL
- [arxiv 2406.16845](https://arxiv.org/abs/2406.16845) — RaTEScore

# Paper skeleton — GazeProbe

Working title:
**How Faithful Is Medical-VLM Attention? A Cross-Model Audit Against Radiologist Gaze**

Target: 6–8 pages, workshop format (ML4H 2026 / iMIMIC 2026). arXiv
tech report independent of workshop outcome.

This file is a working outline. Each section has (a) what it does,
(b) bullet-level content, (c) what figure/table lives there. Convert to
LaTeX (`paper/main.tex`) once a workshop style file is committed.

---

## Abstract (≈ 200 words)

- One sentence on the gap: medical VLMs produce confident CXR reports,
  attention faithfulness vs. radiologist behavior is unmeasured across
  released checkpoints.
- One sentence on what we do: cross-model audit of LLaVA-Med-1.5,
  MedGemma-4B, MAIRA-2 against REFLACX gaze, 3,032 cases.
- Two sentences on findings (placeholder until experiments land):
  *"We find a large, model-specific, pathology-dependent alignment gap.
  For [model X] on [pathology Y], cross-attention is statistically
  indistinguishable from a uniform prior."*
- One sentence on why it matters: alignment correlates with downstream
  factuality — the gap is not cosmetic, it predicts report errors.
- One sentence on what this is not: not a new method; not a new training
  recipe. The probe motivates Phase 2 work on targeted intervention.

---

## 1. Introduction (≈ 1 page)

**Move 1 — Establish the territory.** Medical VLMs (LLaVA-Med, MedGemma,
MAIRA-2) have reached strong CXR-reporting numbers on RadGraph-F1 and
CheXbert. They are increasingly used as black boxes — clinicians read the
generated report, not the model's internal state.

**Move 2 — Open the gap.** Cross-attention is *the* interface between the
image encoder and the language head. If it is unfaithful — pointing
elsewhere than the diagnostically relevant region — the report can be
right *despite* the visual evidence, not *because* of it. That makes the
system fragile under distribution shift.

**Move 3 — Why gaze.** Radiologist gaze (REFLACX) is the closest available
ground-truth proxy for "where a domain expert spent attention while
forming a finding." Aligning model attention with gaze is one operational
definition of visual-evidence faithfulness.

**Move 4 — Why now, why these three models.** LLaVA-Med-1.5 (general
adapter), MedGemma-4B (Google), MAIRA-2 (grounded reporting) span the
architectural diversity of open medical VLMs in 2026. They have not been
compared on this axis.

**Move 5 — Contributions.**
1. First cross-model audit of attention–gaze alignment for medical VLMs
   on REFLACX.
2. Per-pathology breakdown identifying where attention is below random.
3. Correlation analysis: alignment vs. report factuality vs.
   hallucination rate, per case.
4. Released code + HuggingFace demo showing the three models'
   attention maps side-by-side with gaze overlay.

**Move 6 — What this is not.** Not a new gaze-supervised pretraining
(CoGaze, GazeX, Eyes-on-the-Image already do that). Not a new
training-free intervention with gaze (Look & Mark already does that).
This is the diagnostic measurement that motivates future intervention.

---

## 2. Related work (≈ 0.75 page)

Group into four buckets. Cite tightly; do not turn this into a survey.

**2.1 Medical VLMs for CXR reporting.** LLaVA-Med (Li et al., 2024),
MedGemma (Google DeepMind, 2025), MAIRA-2 (Bannur et al., 2024). Note
that none publish faithfulness numbers.

**2.2 Gaze-supervised training of medical VLMs.** CoGaze
([2603.26049](https://arxiv.org/abs/2603.26049)), GazeX
([2604.14316](https://arxiv.org/abs/2604.14316)), Eyes-on-the-Image
([2508.13068](https://arxiv.org/abs/2508.13068)),
Thinking-with-Gaze ([2603.06697](https://arxiv.org/html/2603.06697)),
RadEyeVideo ([2507.09097](https://arxiv.org/abs/2507.09097)). *Gap:*
all train new models; none audit released checkpoints.

**2.3 Training-free gaze-based intervention.** Look & Mark
([2505.22222](https://arxiv.org/abs/2505.22222)): prompt-level ICL with
gaze + bbox on LLaVA-Med and others. *Gap:* shows intervention helps,
but does not measure the underlying alignment that the intervention
is correcting.

**2.4 Faithfulness and reliability probing of medical foundation
models.** MediConfusion ([2409.15477](https://arxiv.org/abs/2409.15477)),
"Medical Context Distorts Decisions in Clinical VLMs"
([2605.17436](https://arxiv.org/html/2605.17436)). *Gap:* both probe
text-side robustness, not visual-attention faithfulness against expert
gaze.

---

## 3. Setup (≈ 0.75 page)

**3.1 Models.** Released checkpoints, bf16, inference only. Hardware:
Modal L40S 48 GB. For each model: what layer produces the
cross-attention we extract, which token positions in the generated
report we condition on (per-sentence aggregation).

**3.2 Dataset.** REFLACX 3,032 cases, 5 radiologists. Gaze traces are
fixation polygons with timestamps; we rasterize to a heatmap at the
model's vision-encoder patch grid. Bounding boxes (Lanfredi labels) used
as a secondary ground-truth signal.

**3.3 Alignment metrics.**
- **KL(attn ∥ gaze)** — gaze rasterized to same grid, both row-stochastic.
- **AUC(attn, gaze>τ)** — treat thresholded gaze as binary, attention as
  score.
- **IoU(top-k attn, bbox)** — discrete grounding metric.
- **NSS, CC** (eye-tracking-standard metrics) — for comparability with
  the saliency literature.

**3.4 Baselines.** Random attention, uniform, CLIP-image-text similarity,
Grad-CAM at the same layer.

**Table 1.** Models × what-we-extract spec sheet.

---

## 4. Methodology — attention extraction (≈ 0.5 page)

Per-model engineering: which hook, which layers, which heads aggregated,
which sentence-token positions used for conditioning. Crucial for
reproducibility because this varies by architecture. State the choice
and the justification in 2–3 sentences per model.

**Figure 1.** Pipeline diagram: CXR → model → per-sentence cross-attention
map → grid-aligned to REFLACX gaze rasterization → alignment metric.

---

## 5. Results (≈ 2 pages — the meat)

**5.1 Headline alignment table.** Mean per-case KL / AUC / IoU per model,
with 95 % bootstrap CIs vs. each baseline. Expected shape: the
medical-trained models beat random but not by as much as the literature's
self-reported attention figures imply.

**Table 2.** Models × metric × baseline.

**5.2 Per-pathology alignment.** Heatmap (models × 13 pathologies) of
mean alignment. Identify which cells are at-or-below the random baseline.

**Figure 2.** Heatmap. The headline figure of the paper.

**5.3 Correlation with downstream factuality.** Per case, regress
RadGraph-F1 on alignment score; same for NLI-hallucination rate.
Stratify by pathology.

**Figure 3.** Two scatter plots (RadGraph-F1 vs. alignment;
hallucination rate vs. alignment), one per model overlaid.

**5.4 Inter-radiologist variance ceiling.** REFLACX has 5 radiologists;
compute radiologist-vs-radiologist alignment as the ceiling. Report
each model as a fraction of the radiologist-pair ceiling — this contains
the "model attention can't be more aligned than another radiologist"
sanity bound.

**Table 3.** Radiologist-pair ceiling vs. each model.

---

## 6. Qualitative analysis (≈ 0.5 page)

Eight case studies: two best-aligned, two worst-aligned, two showing
inter-model disagreement, two where attention is high but the report
is wrong (faithfulness without correctness, or correctness without
faithfulness).

**Figure 4.** 4×2 grid of overlays.

---

## 7. Discussion (≈ 0.5 page)

- What the alignment gap means clinically.
- Why per-pathology variance matters more than the headline number.
- The "right answer for the wrong reason" cases (5.3 + qualitative) —
  alignment is necessary, not sufficient, for trust.
- Limitations: 2D only; English reports only; one dataset; gaze is a
  proxy for attention but not for reasoning. bf16 inference (no
  quantization) keeps the attention distribution at full numerical
  fidelity.

---

## 8. Conclusion + future work (≈ 0.25 page)

- The probe is the contribution. The gap is real, measurable,
  pathology-specific.
- Future work (one paragraph, deliberately vague to leave room for the
  thesis): a decode-time cross-attention intervention designed *for the
  specific failure cells this probe identifies*, benchmarked against
  Look & Mark.

---

## Appendix (uncounted pages, arXiv only)

- A. Full per-model attention-extraction details (hooks, layers, code
  pointer).
- B. ~~4-bit vs. fp16 attention-distribution delta on a 100-case subset.~~
  Removed: bf16 is now the only inference precision (no quantization).
- C. All 13 pathologies' per-model alignment numbers with CIs.
- D. NLI-hallucination eval setup (model, prompts, thresholds).
- E. Radiologist-pair alignment statistics, per pathology.

---

## Methodology decisions (resolved)

All five open methodology questions are resolved in
[`../docs/extraction-spec.md`](../docs/extraction-spec.md). Headline
decisions:

1. **Attention layers:** LLaVA-Med L14–18, MAIRA-2 L14–18 (LLaVA-family
   localization-head evidence); MedGemma L11–22 candidate with a week-2
   50-case pilot to pick best 5.
2. **Per-sentence aggregation:** mean over content tokens (drop
   punctuation + stopwords). Noun-only as ablation on 100 cases.
3. **Gaze grid:** both native per-model grids AND a common 56×56 grid;
   Gaussian-KDE rasterization with REFLACX reference σ.
4. **RadGraph:** XL (matches Look & Mark).
5. **Hallucination:** RaTEScore + RadGraph-XL (matches Look & Mark).
   Skip RadNLI, skip BLEU/ROUGE, skip human eval.

**Bonus decision (not in original five):** MAIRA-2's emitted bounding
boxes are treated as a *second* grounding signal, audited independently
against gaze. Adds a column to Table 2 — itself a publishable
side-finding (is bbox a better gaze proxy than attention for the model
that emits both?).

## Open empirical questions deferred to week 2

- ~~4-bit nf4 vs fp16 attention delta on a 100-case LLaVA-Med subset.~~
  Resolved: no quantization, bf16 throughout.
- MedGemma layer pilot (which 5 of 32 layers carry the best gaze
  alignment signal).
- Exact prompt template per model (use model-card default; document
  in `docs/prompts.md`).

# Paper skeleton — BBoxProbe v1

Working title:
**How Faithful Is Medical-VLM Attention? A Cross-Model Audit Against Radiologist Bounding-Box Annotations**

Target: 6–8 pages, primary target **iMIMIC 2026** (MICCAI Workshop on
Interpretability of MachIne intelligence in Medical Image Computing —
the methodological fit is direct: medical-VLM interpretability is the
workshop's exact scope). Fallback: ML4H 2026 (NeurIPS workshop).
arXiv tech report independent of workshop outcome.

When the iMIMIC 2026 CFP is released (typically late spring), check
the page limit (recent years: 8 pages LNCS including references) and
adapt this skeleton's word budgets accordingly. Springer LNCS style
file required.

This file is a working outline. Each section has (a) what it does,
(b) bullet-level content, (c) what figure/table lives there. Convert to
LaTeX (`paper/main.tex`) once a workshop style file is committed.

Project lineage note: this paper subsumes the gaze-based predecessor
(GazeProbe, git tag `gaze-v0.1`); we pivoted to bbox audit on
VinDr-CXR after PhysioNet credentialing review introduced an
indefinite wait. Methodology, code, metrics, infrastructure all
transferred unchanged.

---

## Abstract (≈ 200 words)

- One sentence on the gap: medical VLMs produce confident CXR
  reports, but attention faithfulness against domain-expert spatial
  grounding is unmeasured across released checkpoints.
- One sentence on what we do: cross-model audit of LLaVA-Med-1.5,
  MedGemma-4B, and MAIRA-2 against VinDr-CXR radiologist bounding
  boxes, on a 2,000-case stratified subset of 14 disease classes.
- Two sentences on findings (placeholder until experiments land):
  *"We find a large, model-specific, class-dependent alignment gap.
  For [model X] on [class Y], cross-attention overlap with the
  radiologist bbox is statistically indistinguishable from a uniform
  prior."*
- One sentence on the MAIRA-2 side-finding (placeholder):
  *"For MAIRA-2 specifically, the model's emitted bounding box is
  better aligned with the radiologist annotation than its own
  internal cross-attention is, suggesting the grounding head is more
  faithful than the attention pattern that produced it."*
- One sentence on why it matters: alignment correlates with
  downstream factuality — the gap is not cosmetic, it predicts
  report errors.
- One sentence on what this is not: not a new method; not a new
  training recipe. The probe motivates Phase 2 work on targeted
  intervention.

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

**Move 3 — Why bbox.** Radiologist-drawn bounding boxes are the
shared gold standard for spatial grounding across the medical-imaging
literature. A bbox marks "where the finding is" — a strictly
necessary (if not sufficient) target for any model that claims to
reason about that finding visually. Aligning model attention with
the bbox is one operational definition of visual-evidence
faithfulness; unlike gaze, it does not require eye-tracking
infrastructure and is directly comparable to model-emitted
grounding boxes (relevant for MAIRA-2).

**Move 4 — Why now, why these three models.** LLaVA-Med-1.5 (general
adapter), MedGemma-4B (Google's medical Gemma), MAIRA-2 (grounded
reporting with emitted bboxes) span the architectural diversity of
open medical VLMs in 2026. None has been compared on attention
faithfulness, and the three-way comparison is what makes
model-specific failure modes visible.

**Move 5 — Contributions.**
1. First cross-model audit of attention–bbox alignment for medical
   VLMs on VinDr-CXR (Kaggle release, 14 disease classes,
   multi-rater).
2. Per-class breakdown identifying where attention is at-or-below a
   random baseline.
3. Correlation analysis: per-case alignment vs. RadGraph-XL F1 and
   RaTEScore.
4. Side-finding: for MAIRA-2, comparison between its emitted bbox
   and its internal attention, both vs. the radiologist bbox.
5. Released code + HuggingFace demo showing the three models'
   attention maps + MAIRA-2's emitted bbox + the gold radiologist
   bbox side-by-side.

**Move 6 — What this is not.** Not a new bbox-supervised pretraining
(many CXR detectors already do that). Not a new training-free
intervention (Look & Mark already pairs prompt-level bbox/gaze hints
with medical VLMs). Not a perturbation-based faithfulness audit
([arxiv 2510.11196](https://arxiv.org/abs/2510.11196) covers that
angle). Not a CLIP-style VLM benchmark
([arxiv 2510.19599 XBench](https://arxiv.org/abs/2510.19599) covers
the encoder-only direction). This is the cross-model, autoregressive-
LVLM, spatial-alignment audit — the cell of the design matrix that
remains open.

---

## 2. Related work (≈ 0.75 page)

Group into four buckets. Cite tightly; do not turn this into a
survey.

**2.1 Medical VLMs for CXR reporting.** LLaVA-Med (Li et al.,
2024), MedGemma (Google DeepMind, 2025), MAIRA-2 (Bannur et al.,
2024). Note that none publish attention-grounding faithfulness
numbers against radiologist bboxes.

**2.2 Spatial-grounding benchmarks for medical VLMs.** XBench
([2510.19599](https://arxiv.org/abs/2510.19599)) benchmarks
encoder-only CLIP-style VLMs (BioVIL, MedCLIP, etc.) on
radiologist-annotated regions. VinDr-CXR-VQA
([2511.00504](https://arxiv.org/abs/2511.00504)) extends VinDr-CXR
with VQA pairs and benchmarks MedGemma alone. *Gap:* neither audits
cross-attention of multiple autoregressive LLM-based VLMs.

**2.3 Training-free interventions for grounding.** Look & Mark
([2505.22222](https://arxiv.org/abs/2505.22222)): prompt-level ICL with
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
report we condition on (per-finding-sentence aggregation).

**3.2 Dataset.** VinDr-CXR (Kaggle "VinBigData Chest X-ray
Abnormalities Detection" release), 18,000 frontal CXRs with bbox
annotations across 14 disease classes by 17 radiologists (3 readers
per image). We use a stratified 2,000-case subset per model, with
per-class oversampling for rare conditions capped at available N.
Multi-rater bboxes handled by union; per-rater spread reported as
appendix. See `docs/extraction-spec.md` §Q3 + §"What this spec does
NOT decide" for subset-sampling and class-synonym table details.

**3.3 Alignment metrics.**
- **IoU(top-k attn, bbox)** — primary metric for BBoxProbe since bbox
  is naturally a region.
- **KL(attn ∥ bbox-mask-normalized)** — both renormalized to
  probability distributions.
- **AUC(attn, bbox-binary)** — attention as score, bbox interior as
  positive class.
- **NSS, CC** — saliency-literature standards for completeness with
  prior gaze-audit literature.
- **MAIRA-2 bonus column:** IoU(emitted bbox, radiologist bbox).

**3.4 Baselines.** Random attention, uniform, CLIP-image-text
similarity, Grad-CAM at the same layer.

**Table 1.** Models × what-we-extract spec sheet.

---

## 4. Methodology — attention extraction (≈ 0.5 page)

Per-model engineering: which hook, which layers, which heads aggregated,
which sentence-token positions used for conditioning. Crucial for
reproducibility because this varies by architecture. State the choice
and the justification in 2–3 sentences per model.

**Figure 1.** Pipeline diagram: CXR → model autoregressive generation
→ per-finding-sentence cross-attention map → grid-aligned to
VinDr-CXR bbox-mask rasterization → alignment metric. Side branch:
MAIRA-2 generated-bbox parse → same grid → IoU vs radiologist bbox.

---

## 5. Results (≈ 2 pages — the meat)

**5.1 Headline alignment table.** Mean per-case KL / AUC / IoU per model,
with 95 % bootstrap CIs vs. each baseline. Expected shape: the
medical-trained models beat random but not by as much as the literature's
self-reported attention figures imply.

**Table 2.** Models × metric × baseline.

**5.2 Per-class alignment.** Heatmap (models × 14 VinDr disease
classes) of mean IoU. Identify which (model × class) cells are
at-or-below the random baseline.

**Figure 2.** Heatmap. The headline figure of the paper.

**5.3 Correlation with downstream factuality.** Per case, regress
RadGraph-XL F1 on alignment score; same for RaTEScore. Stratify by
disease class.

**Figure 3.** Two scatter plots (RadGraph-XL F1 vs. alignment;
RaTEScore vs. alignment), one per model overlaid.

**5.4 Inter-radiologist variance ceiling.** VinDr-CXR has 3
radiologists per image. Compute radiologist-vs-radiologist bbox IoU
(by union and by per-rater pairwise) as the ceiling. Report each
model as a fraction of the radiologist-pair ceiling — this contains
the "model attention can't be more aligned than another radiologist"
sanity bound. Per-rater spread is small for well-defined classes
(consolidation, effusion) and larger for diffuse/subtle classes —
expected and itself a reportable observation.

**Table 3.** Radiologist-pair ceiling vs. each model.

**5.5 MAIRA-2 dual-signal comparison.** Same metrics on MAIRA-2's
emitted bbox vs. the radiologist bbox, compared to MAIRA-2's
internal attention vs. the radiologist bbox. Direct bbox-vs-bbox IoU
is the natural metric.

**Table 4.** MAIRA-2 attention vs. emitted bbox vs. radiologist bbox.

---

## 6. Qualitative analysis (≈ 0.5 page)

Eight case studies: two best-aligned, two worst-aligned, two showing
inter-model disagreement, two where attention is high but the report
is wrong (faithfulness without correctness, or correctness without
faithfulness). Each case carries one paragraph of radiologist
interpretation from Dr. Anuradha, anchoring the automated metrics
in clinical reasoning.

**Figure 4.** 4×2 grid of overlays + radiologist commentary.

---

## 7. Radiologist evaluation (≈ 1.25 pages, new)

Full protocol in [`docs/radiologist-eval-protocol.md`](../docs/radiologist-eval-protocol.md);
methods section here summarizes.

**7.1 Setup.** Board-certified radiologist (Dr. Anuradha, co-author),
blinded to model identity, rates a stratified 100-case × 3-model
subset along a 1–5 Likert clinical-significance scale plus structured
error counts (missed / hallucinated / misclassified / stylistic).
Calibration session against 10 shared cases before the main rating
phase. Plus: pairwise heatmap-preference task on 200 pairs;
independent bbox re-annotation on 50 cases.

**7.2 Likert distribution.** Per-model histogram of Likert scores
plus mean ± SD. Identifies which model produces clinically
acceptable reports most often.

**Figure 5.** Likert distributions × 3 models.

**7.3 Pairwise heatmap preference.** For each (model attention vs.
VinDr bbox) and (MAIRA-2 emitted bbox vs. VinDr bbox) pair, count
preferences. Binomial test against 50%.

**Table 5.** Preference rates per pair-type.

**7.4 Failure-mode taxonomy.** Distribution of attention failures
across the categories defined collaboratively with Dr. Anuradha
(wrong lung field / correct lobe wrong region / off-anatomy /
diffuse / anatomically-plausible distractor). Chi-squared across
models. Identifies which failure mode dominates per model — a
medically actionable finding.

**Table 6.** Failure-mode counts × 3 models.

**7.5 Independent bbox re-annotation (inter-rater ceiling).** Dr.
Anuradha's bboxes on a 50-case subset, compared against the VinDr
consensus. Per-class IoU + κ. The "model attention can't be more
aligned than another radiologist" sanity bound for §5.4.

**Table 7.** Inter-rater ceiling vs. per-model alignment, per class.

---

## 8. Automated-metric validation (≈ 0.5 page, new)

Spearman correlation between Dr. Anuradha's per-report Likert
scores and: (a) RadGraph-XL F1, (b) RaTEScore, (c) per-report
attention-bbox alignment. With 95 % bootstrap CIs.

**Why this matters.** If the attention-bbox alignment score
correlates strongly with radiologist judgment (Spearman ≥ 0.4 is
the threshold used in saliency-vs-clinical-relevance papers), we've
shown that attention faithfulness is *clinically* meaningful, not
just numerically convenient. If correlation is weak, the paper
honestly reports that the metric is internally consistent but
clinically under-specified — also a publishable finding.

**Figure 6.** Three scatter plots (Likert vs. each metric), with
Spearman ρ + CI on each panel.

---

## 9. Discussion (≈ 0.5 page)

- What the alignment gap means clinically.
- Why per-pathology variance matters more than the headline number.
- The "right answer for the wrong reason" cases (5.3 + qualitative) —
  alignment is necessary, not sufficient, for trust.
- Limitations: 2D only; English reports only; one dataset; gaze is a
  proxy for attention but not for reasoning. bf16 inference (no
  quantization) keeps the attention distribution at full numerical
  fidelity.

---

## 10. Conclusion + future work (≈ 0.25 page)

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
   Bbox → binary mask rasterization (filled rectangle) renormalized
   to a probability distribution for KL.
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

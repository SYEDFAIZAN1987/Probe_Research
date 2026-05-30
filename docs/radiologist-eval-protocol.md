# Radiologist evaluation protocol

For Dr. Anuradha to review and approve before scheduling time. The
protocol is the contract: it bounds the time commitment, defines the
rubric so ratings are reproducible, and fixes the authorship +
publication terms in writing.

---

## 1. Purpose

GazeProbe / BBoxProbe v1 is a faithfulness audit of three open
medical vision–language models (LLaVA-Med-1.5, MedGemma-4B,
MAIRA-2) generating chest X-ray reports. Automated metrics
(RadGraph-XL F1, RaTEScore, attention–bbox overlap) tell us part of
the story; radiologist evaluation tells us whether those metrics
correspond to *clinical* judgment. The radiologist evaluation
provides three things the automated pipeline cannot:

1. A clinically grounded "truth" against which to validate our
   automated metrics.
2. A taxonomy of how the models actually fail, in medical terms.
3. A radiologist-vs-radiologist agreement ceiling for the
   attention-vs-bbox claim.

---

## 2. Authorship and credit

**Dr. Anuradha is a confirmed co-author** on the resulting paper.
Order: TBD (typically last-author for the senior medical voice
unless preference differs). Her contributions meet the ICMJE
authorship criteria across:

- Substantial contribution to study design (failure-mode taxonomy,
  rubric calibration).
- Substantial contribution to data acquisition / analysis
  (clinical-significance grading, pairwise preferences, bbox
  re-annotation).
- Drafting / critical revision of the manuscript (medical-claims
  section, limitations, clinical-relevance discussion).
- Final approval and public accountability for the medical claims.

Any colleague of Dr. Anuradha who joins:
- If they participate in rubric design or interpretation in
  addition to rating → co-author.
- If they only perform rating tasks under the locked rubric →
  acknowledged in Acknowledgments with their institutional
  affiliation, by their preference.

---

## 3. IRB / ethics

**Open question, to be confirmed with Dr. Anuradha's institution.**

The work involves radiologists evaluating model-generated outputs
(text reports + attention heatmaps + bounding boxes) on de-identified
chest X-rays from the VinDr-CXR public dataset (Kaggle release,
Vietnamese-cohort, fully de-identified by the dataset's original
creators). **No patient charts, no PHI, no clinical-decision use.**

Most institutions classify this as either (a) not human-subjects
research, or (b) exempt research under "secondary use of
de-identified data." A one-email check with the institutional IRB
office should resolve this in 1–3 business days. **Do not begin
rating until the IRB position is confirmed in writing.**

Template email to IRB office is included in [§9](#9-irb-email-template).

---

## 4. Time budget (Dr. Anuradha, all three tiers)

| Phase | Activity | Estimated hours |
|---|---|---|
| 4.0 | Protocol review + sign-off on this document | 1 |
| 4.1 | Calibration session: 10 shared cases against rubric, plus discussion to align edge cases | 2 |
| 4.2 | **Tier 1a — Clinical-significance grading.** 100 cases × 3 models = 300 generated reports. Likert + structured error counts. ~3 min/report. | 15 |
| 4.3 | **Tier 1b — Qualitative-case commentary.** 8 qualitative-figure cases, one paragraph each. | 2 |
| 4.4 | **Tier 2a — Failure-mode taxonomy.** Inspect ~30 attention-failure examples, define clinical categories collaboratively with first author. | 3 |
| 4.5 | **Tier 2b — Pairwise preference.** 200 pairs (model attention vs. VinDr bbox; MAIRA-2 emitted bbox vs. VinDr bbox). ~30 sec/pair. | 2 |
| 4.6 | **Tier 3 — Independent bbox re-annotation.** 50-case subset, all visible findings. ~5 min/case. | 4 |
| 4.7 | Manuscript review: read draft, suggest edits to clinical sections, sign off on final claims. | 3 |
| | **Total** | **~32 hours over 6 weeks** |

**Hard cap.** If actual time exceeds 40 hours, we cut scope (likely
reduce the 100-case-per-model Tier-1 subset to 75). The 40-hour
ceiling is built into the protocol; this is a research collaboration,
not a clinical workload.

---

## 5. Rubric

### 5.1 Clinical-significance Likert (per generated report)

Each rater scores the report against the source CXR + VinDr-CXR
ground-truth bboxes on a single 1–5 Likert scale:

| Score | Anchor |
|---|---|
| **5** | Clinically accurate. All major findings present in the bbox set are described correctly; no clinically significant hallucinated findings; phrasing acceptable for a clinical report. |
| **4** | Largely correct. Minor omissions or stylistic issues only. A radiologist reading this report alongside the image would not be misled about management. |
| **3** | Generally correct but with notable omissions or inaccuracies. A reader using this report alone (without the image) would have a meaningfully incomplete picture. |
| **2** | One major error or several moderate errors. Could lead to incorrect downstream action if relied upon. |
| **1** | Multiple major errors. Misleading; clinically dangerous if acted upon. |

### 5.2 Structured error counts (per report)

Independent of the Likert score, count:

- **Missed findings**: VinDr bbox-labeled abnormalities the report does not mention.
- **Hallucinated findings**: Report mentions findings not supported by the image.
- **Misclassified findings**: Report correctly identifies a finding but assigns wrong location, laterality, severity, or category.
- **Stylistic-only issues**: Phrasing is unusual but content is correct (these do NOT contribute to Likert score).

### 5.3 Pairwise preference (§4.5)

For each pair, two heatmap overlays are shown on the same CXR
alongside the radiologist bbox(es). Response options:

- **A is more clinically aligned**
- **B is more clinically aligned**
- **A and B are equivalent**
- **Both are clinically wrong**

Order randomized per pair. Heatmap source (model attention vs.
MAIRA-2 emitted bbox vs. random baseline) is blinded.

### 5.4 Failure-mode taxonomy (§4.4)

Categories — finalized collaboratively during the taxonomy
session, expected to land around:

- **Wrong lung field** (right vs. left) — clinically severe
- **Correct lobe, wrong sub-region** — clinically moderate
- **Off-anatomy** (attention on margins, support devices, text overlays) — severe
- **Diffuse / non-localizing** (no clear peak) — mild
- **Anatomically plausible distractor** (heart shadow, hilum, costophrenic angle when the finding is elsewhere) — moderate
- **Other / uncategorized** — escape hatch

### 5.5 Bbox re-annotation (§4.6)

For 50 cases drawn from the stratified subset (covering the 14
disease classes), the radiologist draws bboxes independently for
all visible findings without seeing the original VinDr annotations.
We then compute:

- **Per-class IoU** between Dr. Anuradha's bbox and the VinDr
  consensus bbox (her bbox is treated as one more reader; we don't
  privilege either).
- **Per-class κ** for finding presence/absence.

This gives the inter-rater ceiling referenced in paper §5.4.

---

## 6. Case sampling strategy

### 6.1 Tier-1 sample (300 reports)

- 100 cases per model (LLaVA-Med, MedGemma, MAIRA-2).
- Same 100 underlying CXRs across models so per-case cross-model
  comparison is direct. → Dr. Anuradha rates 3 reports per CXR.
- 100 cases stratified across the 14 VinDr disease classes: ~6
  cases per class, oversampled for rare classes (Pneumothorax,
  Pleural thickening) up to available N.
- Sampling seed and per-class N committed to
  `docs/subset-sampling.md` so the sample is reproducible.

### 6.2 Tier-2 pair sample (200 pairs)

- Drawn from the same 100 cases.
- For each (case, model) cell where there's at least one VinDr
  bbox, generate two heatmap-vs-bbox panels and one pairwise
  comparison.

### 6.3 Tier-3 bbox sample (50 cases)

- Subset of the 100, weighted to cover all 14 classes.

### 6.4 Blinding

- Model identity hidden in all rating screens (raters see "Report
  A" / "Heatmap A", not "MAIRA-2"). Mapping reconstructed at
  analysis time.
- Order of (case, model) presentation randomized per rater.

---

## 7. Calibration session (§4.1)

Before the main rating phase begins, Dr. Anuradha and the first
author review **10 shared cases** together:

1. First author shows the case, the VinDr bbox, and a generated
   report from one model.
2. Dr. Anuradha rates on the rubric while talking through her
   reasoning.
3. First author records any rubric ambiguities that surface.
4. Edge cases discussed; the rubric in §5 is amended if needed.

Goal: the rubric should be stable after the calibration session, so
the remaining ~290 reports rate consistently. If two or more
substantive rubric amendments are needed, we add a second
calibration session before continuing.

If colleagues of Dr. Anuradha join, the calibration is re-run with
them included so all raters share one interpretation of the rubric.

---

## 8. Statistical analysis plan

### 8.1 Validation of automated metrics against Dr. Anuradha's ratings

- Spearman rank correlation between per-report Likert scores and:
  - RadGraph-XL F1 (computed against canonical-template reference)
  - RaTEScore
  - Per-report attention-bbox alignment (mean IoU across the
    report's mentioned findings)
- Reported with 95% bootstrap CIs.

### 8.2 Inter-rater agreement

- If single rater (Dr. Anuradha only): per-rater κ against VinDr's
  original consensus bbox for the Tier-3 re-annotation subset.
- If multiple raters: full inter-rater Krippendorff's α on the
  Likert scores; per-rater agreement table.

### 8.3 Pairwise preference

- For each (heatmap-source-A vs. heatmap-source-B) comparison:
  binomial test against 50% baseline, FDR-adjusted across pairs.

### 8.4 Failure-mode distribution

- Counts per category × per model, normalized to total Tier-1
  errors. Chi-squared test for distribution differences across
  models.

---

## 9. IRB email template

Subject: "Quick IRB-status check — radiologist evaluation of AI outputs on de-identified public dataset"

> Dear IRB Office,
>
> I'm writing to confirm the IRB status of a research activity I'm collaborating on.
>
> **What:** A board-certified radiologist (myself) is being asked to evaluate text reports and visual attention maps generated by three publicly available vision–language models on chest X-rays.
>
> **Source data:** The chest X-rays come from the publicly released VinDr-CXR dataset (Kaggle "VinBigData Chest X-ray Abnormalities Detection" competition release), de-identified by the original publishers. No patient identifiers, no clinical-decision use, no contact with patients.
>
> **My role:** Rate AI-generated reports and attention overlays against the dataset's published bounding-box annotations.
>
> **Output:** A research publication evaluating the models. I will be a co-author.
>
> Does this activity require IRB approval at our institution, or does it qualify as exempt / non-human-subjects research?
>
> Happy to provide more details if helpful.
>
> Best regards,
> Dr. Anuradha [last name]

---

## 10. Open questions for Dr. Anuradha

1. **Author order** preference (last-author senior position, or other).
2. **Affiliation** to list under her name.
3. **Email address** for the corresponding-author exchange (paper revisions etc.).
4. **Conflicts of interest** to disclose (any relevant industry advisory roles, etc.).
5. **Colleagues** — names and availability if/when they're recruited.
6. **IRB outcome** once the email above is sent.
7. **Schedule** — is the ~32 hours best spread across 6 weeks evenly (~5 hours/week), or front-loaded / back-loaded?

---

## 11. Document version + sign-off

- **v0.1** — drafted 2026-05-30 by first author.
- **Awaiting Dr. Anuradha's review.** When she signs off (or proposes amendments), bump to v1.0 and freeze. Subsequent amendments require both authors' agreement.

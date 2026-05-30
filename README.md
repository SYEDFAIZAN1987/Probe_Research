# BBoxProbe — How Faithful Is Medical-VLM Attention?

![tests](https://github.com/SYEDFAIZAN1987/Probe_Research/actions/workflows/test.yml/badge.svg)

A cross-model audit of attention–bbox alignment in three released
medical vision–language models, evaluated against radiologist-drawn
bounding-box annotations on chest X-rays.

> **Status:** work in progress.
> **Primary target venue:** iMIMIC 2026 — the [Interpretability of
> MachIne intelligence in Medical Image Computing](https://imimic-workshop.com/)
> workshop at MICCAI 2026.
> **Stretch:** MICCAI 2026 main conference. The stretch became
> realistic on 2026-05-30 when Dr. Anuradha (board-certified
> radiologist) joined the project as a co-author with a Tier-3
> evaluation commitment (clinical-significance grading + failure-mode
> taxonomy + pairwise preferences + independent bbox re-annotation).
> **Fallback:** ML4H 2026 (NeurIPS workshop).
> arXiv tech report posted regardless of workshop outcome.
>
> **Project lineage:** pivoted on 2026-05-29 from GazeProbe (REFLACX
> radiologist gaze) to BBoxProbe (VinDr-CXR radiologist bbox) to
> escape an indefinite PhysioNet credentialing wait. The gaze-version
> state is preserved at git tag `gaze-v0.1`.

## One-line thesis

Open medical VLMs produce confident chest-X-ray reports, but nobody
has measured how well their internal cross-attention aligns with
**where radiologists drew the abnormality on the image** — and a
systematic per-pathology audit across LLaVA-Med-1.5, MedGemma-4B, and
MAIRA-2 reveals a large, model-specific, pathology-dependent
alignment gap that correlates with downstream report errors.

## Headline questions

1. How well does each model's per-finding-sentence cross-attention
   align with the VinDr-CXR radiologist bbox for that finding?
2. How does that alignment vary across the 14 VinDr-CXR disease
   classes?
3. Does higher attention–bbox alignment predict better downstream
   report factuality (RadGraph-XL F1) and lower hallucination rate
   (RaTEScore)?
4. For which (model × class) cells is attention no better than random
   — or actively *anti*-correlated with the radiologist bbox?

## What this is *not*

- **Not a new training method.** All three VLMs are evaluated as
  released; no fine-tuning, no LoRA.
- **Not a new bbox-based intervention.** Look & Mark
  ([arXiv 2505.22222](https://arxiv.org/abs/2505.22222)) already
  occupies the prompt-level gaze+bbox intervention lane. This work
  measures the gap; closing it is future work.
- **Not multilingual, not 3D, not pediatric.** Scope is VinDr-CXR
  adult-CXR with the Kaggle competition release's 14 disease classes.

## Models audited

| Model | Params | Release | License | Precision |
|---|---|---|---|---|
| LLaVA-Med-1.5 | 7B | Microsoft, 2024 | MSR research | bf16 |
| MedGemma-4B-IT | 4B | Google DeepMind, 2025 | Gemma terms | bf16 |
| MAIRA-2 | ~7B | Microsoft, 2024 | MSR research | bf16 |

## Dataset

[VinDr-CXR](https://vindr.ai/cxr) (Nguyen et al., Sci Data 2022), via
the [Kaggle "VinBigData Chest X-ray Abnormalities Detection"
competition release](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)
— 18,000 frontal chest radiographs with bounding-box annotations
across 14 disease classes by a pool of 17 radiologists (3 readers per
image).

**Access:** free with Kaggle account + acceptance of the competition
DUA. No CITI training, no review wait.

The full 22-class VinDr-CXR release is on PhysioNet under credentialed
access; we deliberately use the smaller, immediately-available Kaggle
subset. The 14-class label set covers all common CXR pathologies
including consolidation, pleural effusion, cardiomegaly, atelectasis,
pneumothorax, and nodule/mass.

## Experiments

| # | Experiment | Output |
|---|---|---|
| 1 | Inference + autoregressive generation on all 3 models × ~2,000 VinDr cases (stratified subset) — extract per-finding-sentence cross-attention maps | `data/attn/{model}/{image_id}.pt` |
| 2 | Alignment metrics (attn↔bbox IoU, KL, AUC, NSS, CC) per (case, class, model) | `data/metrics/per_finding.parquet` |
| 3 | Baselines: random attention, uniform, CLIP-similarity, Grad-CAM | `data/baselines/*.parquet` |
| 4 | Correlation: per-case alignment vs. RadGraph-XL F1 vs. RaTEScore | `paper/figures/fig_correlation.pdf` |
| 5 | Attention-pathology map: (model × class) cells where attention is at-or-below random | `paper/tables/tab_pathology.tex` |
| 6 | Qualitative figure: 8 case studies with best/worst alignment side-by-side, annotated with radiologist commentary | `paper/figures/fig_qualitative.pdf` |
| **7** | **Radiologist evaluation** — clinical-significance Likert grading, pairwise preference, failure-mode taxonomy, independent bbox re-annotation. See [`docs/radiologist-eval-protocol.md`](docs/radiologist-eval-protocol.md). | `data/radiologist/*.csv` + `paper/tables/tab_human_eval.tex` |
| **8** | **Automated-metric validation** — Spearman correlation between radiologist Likert and (RadGraph-XL F1, RaTEScore, attention-bbox IoU) | `paper/figures/fig_metric_validation.pdf` |

**Bonus (MAIRA-2 only):** MAIRA-2's emitted bbox tokens are treated
as a second grounding signal and audited independently against the
radiologist bbox. Both are now bbox-vs-bbox → direct IoU. The side-
question "is the model's emitted bbox better-aligned with the
radiologist than its own attention is?" is itself publishable.

## Compute budget

Modal Starter plan: $30/month, per-second billing, no rollover.
Project spans 2 calendar months → effective $60 budget.

| Phase | GPU-hours @ L40S | $ |
|---|---|---|
| Layer pilot (MedGemma, 50 cases) | 0.5 | $1.00 |
| Attention extraction (3 models × 2k cases) | ~3 | ~$6 |
| Generation pass for downstream eval | ~5 | ~$10 |
| Baselines + analysis (CPU-feasible) | 0 | $0 |
| Debug + restart buffer | 2 | ~$4 |
| **Total** | **~10 hr** | **~$21** |

Plenty of headroom in one month's $30. Cross-month buffer if a
debugging spiral happens.

Hard rules:
- bf16 inference (no quantization). All three models fit comfortably
  on L40S 48 GB.
- Attention maps cached to Modal Volume on first pass; never re-run.
- Stop pods immediately when a function returns. Per-second billing
  punishes idleness less than RunPod but it still adds up.

## Timeline (8 weeks, solo)

| Week | Goal | Done-when |
|---|---|---|
| 1 | VinDr-CXR Kaggle download to Modal Volume; env reproduces one LLaVA-Med inference on a 10-case subset; class-synonym table committed | `data/vindr/`populated + `docs/vindr-class-synonyms.md` |
| 2 | Attention extraction wrappers for all 3 models validated on first GPU run; MedGemma layer pilot completed and frozen | layer-pilot recommendation merged into `docs/extraction-spec.md` §Q1 |
| 3 | Full attention extraction + generation pass on 2,000-case subset × 3 models | `data/attn/` populated |
| 4 | Alignment + baselines: experiments 2 and 3 | per-finding metrics frozen |
| 5 | Correlation + attention-pathology map: experiments 4 and 5; radiologist-eval protocol signed off + IRB check + Gradio labeling interface live | win/loss table frozen + radiologist rating phase begins |
| 6 | Qualitative figures + RadGraph-XL + RaTEScore eval; radiologist rating in progress (asynchronous) | all automated results frozen |
| 7 | Radiologist rating completes; automated-metric validation analysis; draft 6–8 page arXiv tech report with co-author | PDF on arXiv |
| 8 | Polish GitHub repo + HuggingFace Space demo + 1-page summary PDF; final co-author manuscript review | application-ready |

## Deliverables

1. **arXiv tech report** (6–8 pages, workshop format).
2. **GitHub repo** with `reproduce.sh`. No new model weights (probe-only).
3. **HuggingFace Space demo:** upload a CXR, see all three models'
   attention maps + MAIRA-2's emitted bbox + the radiologist bbox
   side-by-side.
4. **1-page project summary PDF** for the MBZUAI application.

## Repository layout

```
.
├── README.md           # this file
├── docs/
│   ├── extraction-spec.md             # methodology decisions (frozen)
│   ├── prompts.md                     # per-model prompt templates
│   ├── radiologist-eval-protocol.md   # co-author Dr. Anuradha's protocol
│   ├── vindr-class-synonyms.md        # week-1 deliverable
│   └── subset-sampling.md             # week-1 deliverable
├── paper/
│   ├── skeleton.md     # outline + section bullets
│   ├── figures/        # generated figures
│   └── tables/         # generated LaTeX tables
├── scripts/
│   ├── modal/          # Modal-app entry points
│   └── *.py            # local CLI scripts (00_*, 01_*, ...)
├── src/
│   ├── attn/           # attention-extraction wrappers per model
│   ├── metrics/        # alignment metrics + rasterization
│   └── eval/           # RadGraph-XL, RaTEScore
├── tests/              # 35+ unit tests; pure-numpy, no GPU
├── data/               # gitignored: attention caches, bbox maps, metrics
├── notebooks/          # exploratory only
├── pyproject.toml
├── requirements.txt
└── reproduce.sh        # end-to-end from cached attentions to figures
```

## Setup

```bash
git clone https://github.com/SYEDFAIZAN1987/Probe_Research.git
cd Probe_Research
python -m venv venv
.\venv\Scripts\Activate.ps1     # PowerShell on Windows
# source venv/bin/activate      # bash on Linux/macOS
pip install -r requirements.txt
```

For the GPU phase, the project is deployed via [Modal](https://modal.com/):

```bash
pip install modal
python -m modal setup           # browser OAuth
```

Then create two persistent Modal Volumes (one for HuggingFace model
cache, one for VinDr-CXR data), upload your Kaggle API token as a
Modal Secret, and run the downloader:

```bash
modal volume create gazeprobe-hf-cache
modal volume create gazeprobe-data
modal run scripts/modal/00_download_vindr.py
```

GPU: Modal L40S 48 GB (per-second billing) or AMD MI300X 192 GB if
the [AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html)
free-credits application is approved. RTX 4090 24 GB is **not**
sufficient in bf16 — LLaVA-Med-1.5 and MAIRA-2 OOM with long context.

## Phase 2 (future thesis pitch, not in this 8-week scope)

This probe identifies *which* (model × class) cells suffer the
largest attention–bbox gap. The natural Phase-2 thesis is a
decode-time cross-attention intervention designed specifically for
those failure cells, benchmarked against Look & Mark's prompt-level
injection. The temporal/scanpath gaze dimension (REFLACX, if
credentialing comes through) is another natural extension.

## License

Code: MIT (planned).
No model weights are released by this repo; pointers only.
See `DISCLAIMER.md` for clinical-use prohibitions per the MAIRA-2
MSRLA terms.

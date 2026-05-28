# GazeProbe — How Faithful Is Medical-VLM Attention?

A cross-model audit of attention–gaze alignment in three released medical
vision–language models, evaluated against radiologist eye-tracking on
chest X-rays.

> **Status:** work in progress. Target venue: ML4H 2026 (NeurIPS workshop)
> or iMIMIC 2026 (MICCAI workshop). arXiv tech report regardless of
> workshop acceptance.

## One-line thesis

Open medical VLMs produce confident chest-X-ray reports, but nobody has
measured how well their internal cross-attention aligns with **where
radiologists actually look** — and a systematic per-pathology audit across
LLaVA-Med-1.5, MedGemma-4B, and MAIRA-2 reveals a large, model-specific,
pathology-dependent alignment gap that correlates with downstream report
errors.

## Headline questions

1. How well does each model's per-sentence cross-attention align with
   REFLACX radiologist gaze fixations?
2. How does that alignment vary across the 13 REFLACX pathologies?
3. Does higher attention–gaze alignment predict better downstream report
   factuality (RadGraph-F1) and lower hallucination rate
   (NLI on Radiology-NLI)?
4. For which (model × pathology) cells is attention no better than
   random — or actively *anti*-correlated with gaze?

## What this is *not*

- **Not a new training method.** All three VLMs are evaluated as
  released; no fine-tuning, no LoRA.
- **Not a new gaze-based intervention.** Look & Mark
  ([arXiv 2505.22222](https://arxiv.org/abs/2505.22222)) and
  CoGaze ([arXiv 2603.26049](https://arxiv.org/abs/2603.26049)) already
  own that lane. This work measures the gap; closing it is future work.
- **Not multilingual, not 3D, not pediatric.** Scope is REFLACX 2D
  adult-CXR English-language reports.

## Models audited

| Model | Params | Release | License | Quant |
|---|---|---|---|---|
| LLaVA-Med-1.5 | 7B | Microsoft, 2024 | MSR research | 4-bit (bnb) |
| MedGemma-4B-IT | 4B | Google DeepMind, 2025 | Gemma terms | 4-bit (bnb) |
| MAIRA-2 | ~7B | Microsoft, 2024 | MSR research | 4-bit (bnb) |

## Dataset

[REFLACX](https://physionet.org/content/reflacx-xray-localization/1.0.0/)
(Lanfredi et al., 2022) — 3,032 chest-radiograph cases with per-radiologist
fixation traces, bounding-box annotations, and transcribed dictation.
Built on top of MIMIC-CXR images.

**Access:** PhysioNet credentialed access required (CITI training
+ application). Lead time ≈ 1 week; **start this on day one.**

## Experiments

| # | Experiment | Output |
|---|---|---|
| 1 | Inference on all 3 models × 3,032 cases — per-sentence cross-attention maps extracted and cached to disk | `data/attn/{model}/{case_id}.pt` |
| 2 | Alignment metrics (attn↔gaze KL, attn↔gaze AUC, attn↔bbox IoU) per case, per model | `data/metrics/per_case.parquet` |
| 3 | Baselines: random attention, uniform, CLIP-similarity, Grad-CAM | `data/baselines/*.parquet` |
| 4 | Correlation: per-case alignment vs. RadGraph-F1 vs. NLI-hallucination rate | `paper/figures/fig_correlation.pdf` |
| 5 | Attention-pathology map: (model × pathology) cells where attention is at-or-below random | `paper/tables/tab_pathology.tex` |
| 6 | Qualitative figure: 8 case studies with best/worst alignment side-by-side | `paper/figures/fig_qualitative.pdf` |

## Compute budget

| | Hours (A40) | $ at $0.39/hr | $ at $0.55/hr |
|---|---|---|---|
| Planned inference + analysis | ~40 | ~$16 | ~$22 |
| 50% buffer (debugging, restarts) | +20 | +$8 | +$11 |
| **Total cap** | **~60 hr** | **~$24** | **~$33** |

Hard rules:
- 4-bit quantization for inference (bitsandbytes / AWQ).
- Attention extraction cached to disk on first pass; never re-run.
- RunPod community / spot instances; tolerate preemption (resume from
  case-id checkpoint).

## Timeline (8 weeks, solo)

| Week | Goal | Done-when |
|---|---|---|
| 1 | REFLACX downloaded + format-explored (fixations, bboxes, case→image map); env reproduces one LLaVA-Med inference number on a 10-case subset | one working notebook + a `data/reflacx_explore.ipynb` documenting the data schema |
| 2 | Attention extraction wrappers for all 3 models; alignment metric code | metrics computed on 100-case subset |
| 3 | Full-dataset attention extraction (all 3 models × 3,032 cases) | `data/attn/` populated |
| 4 | Alignment + baselines: experiments 2 and 3 | per-case metrics frozen |
| 5 | Correlation analysis + attention-pathology map: experiments 4 and 5 | win/loss table frozen |
| 6 | Qualitative figures + RadGraph-F1 + NLI-hallucination eval | all results frozen |
| 7 | Draft 6–8 page arXiv tech report | PDF on arXiv |
| 8 | Polish GitHub repo + HuggingFace Space demo + 1-page summary PDF | application-ready |

## Deliverables

1. **arXiv tech report** (6–8 pages, workshop format).
2. **GitHub repo** with `reproduce.sh`. No new model weights (probe-only).
3. **HuggingFace Space demo:** upload a CXR, see all three models'
   attention maps side-by-side with the REFLACX gaze overlay.
4. **1-page project summary PDF** for the MBZUAI application.

## Repository layout (planned)

```
.
├── README.md           # this file
├── docs/
│   └── extraction-spec.md  # methodology decisions (frozen, week 2 target)
├── paper/
│   ├── skeleton.md     # outline + section bullets (current)
│   ├── figures/        # generated figures
│   └── tables/         # generated LaTeX tables
├── scripts/            # numbered Python scripts (00_*, 01_*, ...)
├── src/
│   ├── attn/           # attention-extraction wrappers per model
│   ├── metrics/        # alignment metrics + baselines
│   └── eval/           # RadGraph-F1, NLI-hallucination
├── data/               # gitignored: attention caches, gaze maps, metrics
├── notebooks/          # exploratory only
├── requirements.txt
└── reproduce.sh        # end-to-end from cached attentions to figures
```

## Setup (placeholder)

```bash
git clone <repo>.git && cd Probe_Research
python -m venv venv && source venv/bin/activate     # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
huggingface-cli login                                # MedGemma is gated
# PhysioNet credentialed dataset — download REFLACX manually after approval
```

GPU: single A40 48 GB on RunPod. RTX 4090 24 GB is **not** sufficient
(LLaVA-Med-1.5 and MAIRA-2 OOM at 4-bit + long context). 

## Phase 2 (future thesis pitch, not in this 8-week scope)

This probe identifies *which* (model × pathology) cells suffer the largest
attention–gaze gap. The natural Phase-2 thesis is a decode-time
cross-attention intervention designed specifically for those failure cells,
benchmarked against Look & Mark's prompt-level injection. Out of scope
here.

## License

Code: MIT (planned).
No model weights are released by this repo; pointers only.

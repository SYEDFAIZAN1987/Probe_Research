"""Per-model attention-extraction wrappers.

The abstract base lives in `base.py`. Concrete subclasses for the three
audited models live in sibling files:

    extract_medgemma.py    — google/medgemma-4b-it
    extract_llavamed.py    — microsoft/llava-med-v1.5-mistral-7b  (TODO)
    extract_maira2.py      — microsoft/maira-2                    (TODO)

The two TODOs are deliberately deferred: their input-prep code is
fundamentally different from MedGemma's (LLaVA mistral_instruct conv
template; MAIRA-2's custom format_and_preprocess_reporting_input),
and writing them ahead of a real model-load test would be guesswork.
"""

from src.attn.base import AttentionExtractor, TeacherForcedInputs, ExtractionResult
from src.attn.extract_medgemma import MedGemmaExtractor

__all__ = [
    "AttentionExtractor",
    "ExtractionResult",
    "MedGemmaExtractor",
    "TeacherForcedInputs",
]

"""Per-model attention-extraction wrappers.

The abstract base lives in `base.py`. Concrete subclasses for the three
audited models live in sibling files:

    extract_medgemma.py    — google/medgemma-4b-it             (validated against pilot)
    extract_llavamed.py    — microsoft/llava-med-v1.5-mistral-7b  (UNTESTED, first-load required)
    extract_maira2.py      — microsoft/maira-2                     (UNTESTED, first-load required)

The two UNTESTED extractors are committed as structural scaffolding so
the abstract base's interface is exercised in code review and so the
known-fragile assumptions are documented inline at the point of use.
They MUST NOT be relied on for numbers before being validated against
a real model load — every fragile assumption is marked with an inline
`# UNTESTED:` comment.
"""

from src.attn.base import AttentionExtractor, TeacherForcedInputs, ExtractionResult
from src.attn.extract_llavamed import LLaVAMedExtractor
from src.attn.extract_maira2 import MAIRA2Extractor
from src.attn.extract_medgemma import MedGemmaExtractor

__all__ = [
    "AttentionExtractor",
    "ExtractionResult",
    "LLaVAMedExtractor",
    "MAIRA2Extractor",
    "MedGemmaExtractor",
    "TeacherForcedInputs",
]

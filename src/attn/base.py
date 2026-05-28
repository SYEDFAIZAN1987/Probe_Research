"""Abstract base for per-model attention extraction.

Each concrete subclass overrides three model-specific pieces:
  - load():            model + processor loading
  - prepare_inputs():  build the teacher-forced input tensors and
                       locate image-token positions + assistant range
  - default_layers():  the layer set frozen in docs/extraction-spec.md
                       §Q1 for this model

The base class provides the shared protocol:
  - extract_teacher_forced(): one forward pass with output_attentions,
                              packaged into an ExtractionResult
  - content_token_mask():     drops punctuation and stopwords from
                              the assistant range
  - per_sentence_maps():      produces a list of (sentence, attn_grid)
                              per the extraction spec §Q2

The base class does NOT pick a "best" layer aggregation strategy —
that's a downstream choice made by the caller (the pilot picks
mean-of-5-mid-layers; downstream eval scripts may pick differently).
"""

from __future__ import annotations

import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # avoid hard torch import at module load time
    import torch
    from PIL.Image import Image


# --------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------- #

@dataclass
class TeacherForcedInputs:
    """Inputs ready for a single forward pass that captures attentions
    over both the prompt and the assistant-generated tokens.

    inputs: kwargs passable to `model(**inputs)`. Must include
        input_ids; image features come via the model's image processor.
    image_token_positions: 1D LongTensor of indices in input_ids[0]
        that correspond to image-placeholder tokens.
    assistant_range: (a_start, a_end) half-open interval over
        input_ids[0] covering the assistant-content tokens.
    """
    inputs: dict[str, Any]
    image_token_positions: "torch.Tensor"
    assistant_range: tuple[int, int]


@dataclass
class ExtractionResult:
    """The packaged output of one teacher-forced extraction pass."""
    attentions: tuple["torch.Tensor", ...]   # per-layer (B=1, H, q, k)
    input_ids: "torch.Tensor"                # (1, q)
    image_token_positions: "torch.Tensor"    # (n_image,)
    assistant_range: tuple[int, int]
    native_grid: tuple[int, int]             # (H, W) for reshaping attn → 2D
    generated_text: str                      # assistant content used
    model_id: str
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- #
# Stopword cache
# --------------------------------------------------------------------- #

_STOPWORDS: set[str] | None = None


def _stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        try:
            from nltk.corpus import stopwords
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


# --------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------- #

class AttentionExtractor(ABC):
    """Abstract attention extractor.

    Concrete subclasses:
        MedGemmaExtractor — see extract_medgemma.py
        LLaVAMedExtractor — TODO (deferred per __init__.py)
        MAIRA2Extractor   — TODO (deferred per __init__.py)
    """

    model_id: str
    native_grid: tuple[int, int]

    def __init__(self, model_id: str, native_grid: tuple[int, int]):
        self.model_id = model_id
        self.native_grid = native_grid
        self.model = None
        self.processor = None

    # ------------------------------------------------------------------ #
    # Abstract — model-specific
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self, *, quant_config=None, attn_implementation: str = "eager") -> None:
        """Load self.model and self.processor.

        Subclasses must set attn_implementation='eager' (or equivalent)
        so output_attentions actually materializes — the SDPA backend
        silently drops attention weights under 4-bit quantization.
        """
        ...

    @abstractmethod
    def prepare_inputs(
        self,
        image: "Image",
        ground_truth_report: str,
    ) -> TeacherForcedInputs:
        """Build inputs for a teacher-forced forward pass with the
        provided report as the assistant content."""
        ...

    @abstractmethod
    def default_layers(self) -> list[int]:
        """Return the frozen layer set for this model from
        docs/extraction-spec.md §Q1. For MedGemma this is the pilot's
        output; for LLaVA-Med / MAIRA-2 this is L14–18 by default."""
        ...

    # ------------------------------------------------------------------ #
    # Shared protocol
    # ------------------------------------------------------------------ #

    def extract_teacher_forced(
        self,
        image: "Image",
        ground_truth_report: str,
    ) -> ExtractionResult:
        """Run one forward pass with output_attentions=True."""
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError(f"{type(self).__name__}.load() must be called first")

        prep = self.prepare_inputs(image, ground_truth_report)

        with torch.inference_mode():
            out = self.model(**prep.inputs, output_attentions=True, return_dict=True)

        if out.attentions is None:
            raise RuntimeError(
                "model returned attentions=None. Confirm attn_implementation "
                "was set to 'eager' at load time — SDPA silently drops them."
            )

        return ExtractionResult(
            attentions=tuple(out.attentions),
            input_ids=prep.inputs["input_ids"],
            image_token_positions=prep.image_token_positions,
            assistant_range=prep.assistant_range,
            native_grid=self.native_grid,
            generated_text=ground_truth_report,
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------ #
    # Aggregation helpers
    # ------------------------------------------------------------------ #

    def content_token_mask(
        self,
        result: ExtractionResult,
    ) -> "torch.Tensor":
        """Bool mask over the assistant range: True for content tokens
        (drop punctuation, stopwords, special tokens).

        Implements docs/extraction-spec.md §Q2: per-sentence aggregation
        uses content tokens only.
        """
        import torch

        a_start, a_end = result.assistant_range
        ids = result.input_ids[0, a_start:a_end]
        sw = _stopwords()
        special = {
            self.processor.tokenizer.bos_token or "",
            self.processor.tokenizer.eos_token or "",
            self.processor.tokenizer.pad_token or "",
            "<start_of_turn>", "<end_of_turn>", "<image_soft_token>",
        }
        special.discard("")

        mask = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        for i, tok_id in enumerate(ids.tolist()):
            tok = self.processor.tokenizer.decode([tok_id]).strip()
            tok_l = tok.lower()
            if not tok:
                mask[i] = False
            elif tok in special:
                mask[i] = False
            elif tok_l in sw:
                mask[i] = False
            elif all(c in string.punctuation for c in tok):
                mask[i] = False
        return mask

    def aggregate_to_grid(
        self,
        result: ExtractionResult,
        *,
        layer_indices: list[int] | None = None,
        head_aggregation: str = "mean",
    ) -> np.ndarray:
        """Aggregate text→image attention across the given layers and
        heads into a single (H, W) probability map at the model's
        native grid.

        Procedure (per docs/extraction-spec.md):
          1. Restrict to layers in `layer_indices` (default: self.default_layers()).
          2. Mean across heads (head_aggregation="mean").
          3. Restrict queries to assistant-range × content-token-mask.
          4. Restrict keys to image-token positions.
          5. Mean over remaining query tokens.
          6. Mean across layers.
          7. Reshape to (G, G), normalize.
        """
        import torch
        from src.metrics.rasterize import reshape_attention_to_grid

        if head_aggregation != "mean":
            raise NotImplementedError(f"head_aggregation={head_aggregation!r}")

        layers = layer_indices if layer_indices is not None else self.default_layers()
        a_start, a_end = result.assistant_range
        mask = self.content_token_mask(result)
        img_pos = result.image_token_positions

        per_layer_maps = []
        for li in layers:
            if li < 0 or li >= len(result.attentions):
                raise IndexError(f"layer index {li} out of range for {len(result.attentions)} layers")
            attn = result.attentions[li][0]              # (heads, q, k)
            attn = attn.mean(dim=0)                       # (q, k)
            attn = attn[a_start:a_end][mask][:, img_pos]  # (n_content, n_img)
            if attn.numel() == 0:
                continue
            attn = attn.float().mean(dim=0)               # (n_img,)
            per_layer_maps.append(attn.cpu().numpy())

        if not per_layer_maps:
            raise RuntimeError("no content tokens survived masking; cannot aggregate")

        stacked = np.stack(per_layer_maps, axis=0).mean(axis=0)  # (n_img,)
        H, W = result.native_grid
        return reshape_attention_to_grid(stacked, (H, W))

    # ------------------------------------------------------------------ #
    # Per-sentence aggregation — for the audit's per-finding analysis
    # ------------------------------------------------------------------ #

    def per_sentence_maps(
        self,
        result: ExtractionResult,
        sentences: list[tuple[int, int]],
        *,
        layer_indices: list[int] | None = None,
    ) -> list[np.ndarray]:
        """For each (sentence_start, sentence_end) span in the assistant
        range (indices relative to the assistant range), return the 2D
        attention map averaged over content tokens of that sentence.

        Sentence spans come from the caller — produced by scispacy
        sentence splitting on result.generated_text and mapped back to
        token positions. That mapping is a caller responsibility because
        tokenizer-to-character offsets vary per model.
        """
        import torch
        from src.metrics.rasterize import reshape_attention_to_grid

        layers = layer_indices if layer_indices is not None else self.default_layers()
        a_start, a_end = result.assistant_range
        full_mask = self.content_token_mask(result)
        img_pos = result.image_token_positions
        H, W = result.native_grid

        out_maps: list[np.ndarray] = []
        for s_start, s_end in sentences:
            local_mask = torch.zeros_like(full_mask)
            local_mask[s_start:s_end] = full_mask[s_start:s_end]
            if local_mask.sum() == 0:
                out_maps.append(np.full((H, W), np.nan, dtype=np.float32))
                continue

            per_layer = []
            for li in layers:
                attn = result.attentions[li][0].mean(dim=0)         # (q, k)
                attn = attn[a_start:a_end][local_mask][:, img_pos]  # (n_sent, n_img)
                attn = attn.float().mean(dim=0)                      # (n_img,)
                per_layer.append(attn.cpu().numpy())
            stacked = np.stack(per_layer, axis=0).mean(axis=0)
            out_maps.append(reshape_attention_to_grid(stacked, (H, W)))
        return out_maps

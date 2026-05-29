"""MedGemma-4B-IT concrete attention extractor.

Mirrors the inline logic in scripts/02_medgemma_layer_pilot.py but
factored as a subclass of AttentionExtractor. The pilot remains the
canonical single-file artifact for layer-window calibration; this
class is what week-3's full-3,032-case extraction uses.

Default layer set: read from a local override file written by the
pilot if present (`data/pilot/medgemma/layer_window.json`), else the
spec's candidate range L11–22 mid-point window L14–18.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.attn.base import AttentionExtractor, TeacherForcedInputs

if TYPE_CHECKING:
    from PIL.Image import Image


# Path the pilot writes its frozen layer recommendation to.
_PILOT_LAYER_WINDOW = Path("data/pilot/medgemma/layer_window.json")


class MedGemmaExtractor(AttentionExtractor):
    """Extractor for `google/medgemma-4b-it`.

    Native grid: 16×16 (256 image tokens from MedSigLIP @ 896²).
    """

    def __init__(self, model_id: str = "google/medgemma-4b-it"):
        super().__init__(model_id=model_id, native_grid=(16, 16))

    # ------------------------------------------------------------------ #

    def load(
        self,
        *,
        quant_config=None,
        dtype=None,
        attn_implementation: str = "eager",
    ) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        load_kwargs = dict(
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        if quant_config is not None:
            # Explicit override path — keeps 4-bit available for
            # opt-in testing on smaller GPUs.
            load_kwargs["quantization_config"] = quant_config
        else:
            # Default per docs/extraction-spec.md §Q6: bf16, no quant.
            load_kwargs["torch_dtype"] = dtype if dtype is not None else torch.bfloat16

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **load_kwargs,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #

    def _resolve_image_token_id(self) -> int:
        """Layered lookup: model.config first, then tokenizer."""
        tok_id = getattr(self.model.config, "image_token_index", None)
        if tok_id is None:
            tok_id = getattr(getattr(self.model.config, "text_config", None),
                             "image_token_index", None)
        if isinstance(tok_id, int) and tok_id > 0:
            return tok_id
        candidate = self.processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
        if candidate is not None and candidate != self.processor.tokenizer.unk_token_id:
            return candidate
        return -1

    def _find_image_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find the contiguous block of image-placeholder tokens.

        Tries the resolved image_token_id; falls back to scanning for a
        contiguous run of 256 identical tokens.
        """
        tok_id = self._resolve_image_token_id()
        ids = input_ids[0]
        expected = self.native_grid[0] * self.native_grid[1]

        if tok_id >= 0:
            positions = (ids == tok_id).nonzero(as_tuple=True)[0]
        else:
            positions = torch.tensor([], dtype=torch.long, device=ids.device)
            for start in range(ids.numel() - expected + 1):
                window = ids[start : start + expected]
                if (window == window[0]).all():
                    positions = torch.arange(start, start + expected, device=ids.device)
                    break

        if positions.numel() != expected:
            raise RuntimeError(
                f"MedGemma: expected {expected} image tokens, found {positions.numel()}. "
                "Print processor.tokenizer.special_tokens_map and model.config to debug."
            )
        gaps = positions[1:] - positions[:-1]
        if positions.numel() > 1 and not (gaps == 1).all():
            raise RuntimeError(
                f"MedGemma: image tokens not contiguous: positions={positions.tolist()[:10]}..."
            )
        return positions

    # ------------------------------------------------------------------ #

    def prepare_inputs(
        self,
        image: "Image",
        ground_truth_report: str,
    ) -> TeacherForcedInputs:
        messages_user_only = [
            {"role": "system",
             "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user",
             "content": [
                 {"type": "text", "text": "Describe this X-ray"},
                 {"type": "image", "image": image},
             ]},
        ]
        messages_full = messages_user_only + [
            {"role": "assistant",
             "content": [{"type": "text", "text": ground_truth_report}]},
        ]

        full = self.processor.apply_chat_template(
            messages_full, add_generation_prompt=False,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        short = self.processor.apply_chat_template(
            messages_user_only, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )

        # Assistant range — longest common prefix
        ids_full = full["input_ids"][0]
        ids_short = short["input_ids"][0].to(ids_full.device)
        prefix = min(ids_full.numel(), ids_short.numel())
        for i in range(prefix):
            if ids_full[i].item() != ids_short[i].item():
                prefix = i
                break
        if prefix == 0:
            raise RuntimeError(
                "MedGemma: assistant-range diff failed — no common prefix. "
                "Inspect the processor chat_template config."
            )
        end = ids_full.numel()
        eos = self.processor.tokenizer.eos_token_id
        while end > prefix and ids_full[end - 1].item() == eos:
            end -= 1

        img_pos = self._find_image_positions(full["input_ids"])

        return TeacherForcedInputs(
            inputs=dict(full),
            image_token_positions=img_pos,
            assistant_range=(prefix, end),
        )

    # ------------------------------------------------------------------ #

    def default_layers(self) -> list[int]:
        """Read the pilot's frozen window if present, else fall back to
        the mid-band the spec named as the starting candidate."""
        if _PILOT_LAYER_WINDOW.exists():
            data = json.loads(_PILOT_LAYER_WINDOW.read_text())
            return list(data["layers"])
        # Spec's starting band, centered on L14-18
        return [14, 15, 16, 17, 18]

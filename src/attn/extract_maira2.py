"""MAIRA-2 concrete attention extractor.

**UNTESTED — written from the HF model card and paper.** MAIRA-2's
input pipeline is fundamentally different from MedGemma and LLaVA-Med:

  - It does NOT use a chat template; instead it ships a custom method
    `processor.format_and_preprocess_reporting_input(...)` that takes
    structured fields (frontal/lateral views + clinical context strings
    + grounding toggle) and returns ready-to-run input tensors.
  - The processor handles all formatting end-to-end. There is no
    user-facing prompt string we can edit.
  - For teacher-forced extraction we must build the prompt via the
    processor, then APPEND the gold report tokens manually.

Specific items requiring first-load confirmation (marked `# UNTESTED:`
inline):

  1. The native image grid size. RAD-DINO ViT-B is reportedly used at
     518² with patch 14, which would give 37×37 = 1369 tokens. The
     constructor uses `(37, 37)` as a guess — REPLACE if the actual
     image-token count differs.
  2. The image-placeholder token id. MAIRA-2's processor may use a
     custom token name or rely on internal expansion.
  3. Whether `AutoModelForCausalLM` is the right model class (needs
     `trust_remote_code=True` because of the custom processor).
  4. Whether the gold-report-tokens-appended teacher-forcing approach
     produces sensible attention. MAIRA-2 was trained to PRODUCE
     grounded reports with bbox tokens; teacher-forcing with the
     un-bbox-annotated VinDr-CXR dictation is methodologically different
     from training. Acknowledge in the paper.

VinDr-CXR-specific note: VinDr-CXR (Kaggle release) provides only
frontal views, no structured indication / technique / comparison /
prior fields, and no reference dictation reports.
We pass empty strings for those — a known deviation from MAIRA-2's
intended use, called out in §7 of the paper skeleton.

Default layers L14–18 per docs/extraction-spec.md §Q1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.attn.base import AttentionExtractor, TeacherForcedInputs

if TYPE_CHECKING:
    from PIL.Image import Image


class MAIRA2Extractor(AttentionExtractor):
    """Extractor for `microsoft/maira-2`.

    Backbone: Vicuna-7B + RAD-DINO-MAIRA-2 (ViT-B, ~87M).
    Native image grid: **PROVISIONAL (37, 37)** — confirm on first
    load by counting image-placeholder tokens in the prompt input_ids.
    Default attention layers: 14–18.
    """

    def __init__(self, model_id: str = "microsoft/maira-2"):
        # UNTESTED: native_grid is a guess from RAD-DINO ViT-B at 518²
        # with patch 14. Re-instantiate with the verified grid after
        # first model load:
        #   ext = MAIRA2Extractor()
        #   ext.load()
        #   # examine prompt input_ids, count image tokens
        #   ext.native_grid = (verified_h, verified_w)
        super().__init__(model_id=model_id, native_grid=(37, 37))

    # ------------------------------------------------------------------ #

    def load(
        self,
        *,
        quant_config=None,
        dtype=None,
        attn_implementation: str = "eager",
    ) -> None:
        # UNTESTED: AutoModelForCausalLM + trust_remote_code is the
        # path documented on the model card for MAIRA-2. If the
        # processor's `format_and_preprocess_reporting_input` is not
        # exposed, fall back to importing from the model's own module
        # after `trust_remote_code=True` triggers code download.
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True,
        )

        load_kwargs = dict(
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = dtype if dtype is not None else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **load_kwargs,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #

    def _find_image_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find image-placeholder positions in the prompt input_ids.

        UNTESTED: MAIRA-2's image placeholder token id is not documented
        in the model card. Three fallback layers, in order:

          1. `self.processor.image_token_id` or similar attribute
          2. `self.model.config.image_token_index`
          3. Contiguous-block scan for any run of `expected` identical
             tokens; the largest such block is presumed to be the image.

        After first successful load, replace this with the verified
        single-attribute lookup.
        """
        expected = self.native_grid[0] * self.native_grid[1]
        ids = input_ids[0]

        for attr in ("image_token_id", "image_token_index"):
            tok_id = getattr(self.processor, attr, None)
            if isinstance(tok_id, int) and tok_id >= 0:
                positions = (ids == tok_id).nonzero(as_tuple=True)[0]
                if positions.numel() == expected:
                    return positions

        tok_id = getattr(self.model.config, "image_token_index", None)
        if isinstance(tok_id, int) and tok_id >= 0:
            positions = (ids == tok_id).nonzero(as_tuple=True)[0]
            if positions.numel() == expected:
                return positions

        # Fallback: largest contiguous run of identical tokens.
        # Slow on large sequences; only used until the right attribute
        # path is identified.
        best_start, best_len, best_tok = -1, 0, None
        run_start = 0
        for i in range(1, ids.numel() + 1):
            if i == ids.numel() or ids[i].item() != ids[run_start].item():
                run_len = i - run_start
                if run_len > best_len:
                    best_len = run_len
                    best_start = run_start
                    best_tok = ids[run_start].item()
                run_start = i
        if best_len >= expected:
            return torch.arange(best_start, best_start + expected, device=ids.device)

        raise RuntimeError(
            f"MAIRA-2: could not locate {expected} contiguous image-token "
            f"positions in input_ids of length {ids.numel()}. Longest "
            f"identical run found: {best_len} of token id {best_tok}. "
            "Likely fix: print `processor.tokenizer.special_tokens_map` "
            "and `dir(processor)` to find the actual image-token attribute "
            "MAIRA-2 uses, then update _find_image_positions accordingly."
        )

    # ------------------------------------------------------------------ #

    def prepare_inputs(
        self,
        image: "Image",
        ground_truth_report: str,
    ) -> TeacherForcedInputs:
        """Build teacher-forced inputs via the custom MAIRA-2 processor,
        then append the gold report tokens.

        UNTESTED — see module docstring. The two-stage build:

          1. Call `format_and_preprocess_reporting_input` with empty
             clinical-context fields and `get_grounding=False` (because
             VinDr-CXR gold reports don't have bbox tokens).
          2. Tokenize the gold report separately, append to input_ids,
             extend attention_mask. Image embeddings carry through
             because they're keyed to input positions in the prompt.
        """
        # UNTESTED: empty-string handling for indication/technique/etc.
        # The processor MAY require these to be None instead of "".
        # If empty strings produce a malformed prompt, swap to None.
        prompt_inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication="",
            technique="",
            comparison="",
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        )

        prompt_input_ids = prompt_inputs["input_ids"]
        device = self.model.device

        # Tokenize the gold report without adding special tokens — those
        # already terminate the prompt portion built by the processor.
        # UNTESTED: confirm `add_special_tokens=False` is correct for
        # this tokenizer. If a BOS is required, this will produce a
        # systematically shifted assistant range.
        report_ids = self.processor.tokenizer(
            ground_truth_report,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(prompt_input_ids.device)

        full_input_ids = torch.cat([prompt_input_ids, report_ids], dim=1)
        a_start = prompt_input_ids.shape[1]
        a_end = full_input_ids.shape[1]

        # Build the final inputs dict — copy processor outputs, override
        # input_ids and attention_mask to the concatenated lengths.
        inputs: dict = {}
        for k, v in prompt_inputs.items():
            inputs[k] = v
        inputs["input_ids"] = full_input_ids
        if "attention_mask" in prompt_inputs:
            extra_mask = torch.ones_like(report_ids)
            inputs["attention_mask"] = torch.cat(
                [prompt_inputs["attention_mask"], extra_mask], dim=1,
            )

        # Move tensors to device + bf16 where appropriate
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if v.dtype in (torch.float32, torch.float16):
                    inputs[k] = v.to(device=device, dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to(device=device)

        img_pos = self._find_image_positions(full_input_ids)

        return TeacherForcedInputs(
            inputs=inputs,
            image_token_positions=img_pos,
            assistant_range=(a_start, a_end),
        )

    # ------------------------------------------------------------------ #

    def default_layers(self) -> list[int]:
        return [14, 15, 16, 17, 18]

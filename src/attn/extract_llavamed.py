"""LLaVA-Med-1.5 concrete attention extractor.

**UNTESTED — written from the HF model card alone.** Validate on
first real model load before relying on numbers from this extractor.
Specific items that need confirmation, marked with `# UNTESTED:`
inline:

  1. The HF model class — model card uses the `image-text-to-text`
     pipeline, strongly suggesting `LlavaForConditionalGeneration`,
     but the LLaVA library's own loader may be required for the
     mistral_instruct conv template to behave correctly.
  2. The image-token expansion behavior. LLaVA models sometimes use a
     single `<image>` placeholder that the model expands internally to
     576 embeddings (CLIP-ViT-L/14 @ 336² = 24×24 patches), and
     sometimes pre-expand in `input_ids`. The two cases require
     different handling for attention extraction.
  3. The chat-template-vs-mistral_instruct path: the HF processor may
     or may not support `apply_chat_template` for this model. If not,
     fall back to manual construction:
        `<s>[INST] <image>\\n{user_text} [/INST] {report}</s>`

Default layers L14–18 per docs/extraction-spec.md §Q1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.attn.base import AttentionExtractor, TeacherForcedInputs

if TYPE_CHECKING:
    from PIL.Image import Image


class LLaVAMedExtractor(AttentionExtractor):
    """Extractor for `microsoft/llava-med-v1.5-mistral-7b`.

    Backbone: Mistral-7B-Instruct + CLIP-ViT-L/14 @ 336².
    Native image grid: 24×24 (576 image tokens).
    Default attention layers: 14–18.
    """

    def __init__(self, model_id: str = "microsoft/llava-med-v1.5-mistral-7b"):
        super().__init__(model_id=model_id, native_grid=(24, 24))

    # ------------------------------------------------------------------ #

    def load(
        self,
        *,
        quant_config=None,
        dtype=None,
        attn_implementation: str = "eager",
    ) -> None:
        # UNTESTED: LlavaForConditionalGeneration is the most likely
        # class given the HF card's pipeline example. If this fails,
        # try AutoModelForCausalLM with trust_remote_code=True or use
        # the LLaVA library's loader.
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        load_kwargs = dict(
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        else:
            load_kwargs["torch_dtype"] = dtype if dtype is not None else torch.bfloat16

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs,
        )
        self.model.eval()

    # ------------------------------------------------------------------ #

    def _resolve_image_token_id(self) -> int:
        """Layered: model.config → tokenizer lookup."""
        tok_id = getattr(self.model.config, "image_token_index", None)
        if isinstance(tok_id, int) and tok_id > 0:
            return tok_id
        # UNTESTED: LLaVA's standard placeholder is "<image>", but
        # LLaVA-Med may use a different special token.
        candidate = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        unk = self.processor.tokenizer.unk_token_id
        if candidate is not None and candidate != unk:
            return candidate
        return -1

    def _find_image_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find image-placeholder positions in input_ids.

        UNTESTED: LLaVA has two known modes:

        (a) Pre-expansion: the processor emits `expected=576` contiguous
            identical image tokens in input_ids. Easy — same path as
            MedGemma.

        (b) Single-token: input_ids contains exactly ONE `<image>` token
            which the model expands internally via
            `model.get_input_embeddings` during the forward. In this
            case, attention extraction needs to operate on the
            POST-EXPANSION sequence (longer than input_ids by 575).
            Hugging Face's LLaVA forward exposes the expanded positions
            via `inputs_embeds` shaping; we'd need to either (i) compute
            inputs_embeds ourselves and use `inputs_embeds=` rather than
            `input_ids=`, or (ii) read the model's first-layer
            attention shape and infer the expanded image span.

        This method handles (a) directly and raises a clear
        NotImplementedError for (b) so the failure is loud and the fix
        is obvious. On the first successful model load, run a one-cell
        diagnostic to determine which mode is in effect:

            >>> ids = inputs["input_ids"][0]
            >>> (ids == llava.config.image_token_index).sum()
            # tensor(1)  → mode (b)
            # tensor(576) → mode (a)
        """
        tok_id = self._resolve_image_token_id()
        if tok_id < 0:
            raise RuntimeError(
                "LLaVA-Med: image_token_id lookup failed. "
                "Inspect `processor.tokenizer.special_tokens_map` and "
                "`model.config.image_token_index`."
            )
        ids = input_ids[0]
        expected = self.native_grid[0] * self.native_grid[1]   # 576
        positions = (ids == tok_id).nonzero(as_tuple=True)[0]

        if positions.numel() == expected:
            gaps = positions[1:] - positions[:-1]
            if not (gaps == 1).all():
                raise RuntimeError(
                    "LLaVA-Med: image-token block not contiguous: "
                    f"positions={positions.tolist()[:10]}..."
                )
            return positions

        if positions.numel() == 1:
            raise NotImplementedError(
                "LLaVA-Med returned a single image-token placeholder. "
                "Attention extraction needs the post-expansion 576-token "
                "image span. Two fixes:\n"
                "  (i) compute inputs_embeds yourself via "
                "self.model.get_input_embeddings() and pass "
                "inputs_embeds= to model.forward; the image embeddings "
                "occupy positions [single_pos, single_pos+576).\n"
                "  (ii) hook the model's vision-language projector to "
                "read the expanded sequence length on each forward.\n"
                "Both require first-load confirmation of which path the "
                "current transformers version uses for LLaVA."
            )

        raise RuntimeError(
            f"LLaVA-Med: expected 1 or {expected} image-token positions, "
            f"found {positions.numel()}."
        )

    # ------------------------------------------------------------------ #

    def prepare_inputs(
        self,
        image: "Image",
        ground_truth_report: str,
    ) -> TeacherForcedInputs:
        """Build teacher-forced inputs using the HF chat-template path.

        UNTESTED: this assumes LLaVA-Med's processor supports
        `apply_chat_template` with role+content lists. If
        `apply_chat_template` errors, the fallback is to construct the
        mistral_instruct string manually:

            <s>[INST] <image>
            {user_text} [/INST] {report}</s>

        and tokenize via `self.processor(...)`.
        """
        messages_user_only = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": "Describe this chest radiograph. Cover the lungs, "
                         "pleura, heart and mediastinum, and bones / soft "
                         "tissues. Do not produce an impression."},
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

        # Longest-common-prefix → assistant range
        ids_full = full["input_ids"][0]
        ids_short = short["input_ids"][0].to(ids_full.device)
        prefix = min(ids_full.numel(), ids_short.numel())
        for i in range(prefix):
            if ids_full[i].item() != ids_short[i].item():
                prefix = i
                break
        if prefix == 0:
            raise RuntimeError(
                "LLaVA-Med: assistant-range diff failed — no common prefix "
                "between [user] and [user+assistant] tokenizations. "
                "Fall back to manual mistral_instruct construction."
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
        return [14, 15, 16, 17, 18]

"""LLaVA-Med-1.5 concrete attention extractor.

**Validated 2026-05-30** against the chaoyinshe HF-format conversion
(see "checkpoint choice" below). Three pieces of knowledge baked in
from the first failed load:

  1. **Use the converted checkpoint**, not the original microsoft one.
     The original `microsoft/llava-med-v1.5-mistral-7b` uses
     `model_type: llava_mistral` which transformers v5 maps to a class
     with a different state-dict layout (`model.layers.X` vs.
     `model.language_model.layers.X`). Loading it silently produces a
     randomly-initialized model with a LOAD REPORT showing every
     weight as UNEXPECTED/MISSING. The
     `chaoyinshe/llava-med-v1.5-mistral-7b-hf` checkpoint is a
     drop-in HF-format conversion of the same weights — only key
     names differ, the network is byte-identical aside from a
     32000→32064 vocabulary expansion (added image-token-related
     special tokens; attention layers unchanged).
  2. **The chat template needs the {"type": "image"} format** (no
     `image` key in the dict). The HF processor's chat template for
     LLaVA does string concatenation; passing a list under `content`
     breaks with `TypeError: can only concatenate str (not "list")
     to str`. The model card's recipe uses:
         {"role": "user", "content": [
             {"type": "image"},  # no image= key
             {"type": "text", "text": "..."},
         ]}
     Then the image is passed separately to
     processor(images=[image], text=prompt, ...).
  3. **Single-token image expansion.** HF LLaVA puts ONE <image> token
     in input_ids and the model internally replaces it with 576 image
     embeddings during forward. The attentions returned by
     output_attentions=True have shape (..., q, k) with q = k =
     (input_ids_len + 575). To make attention slicing work cleanly
     against the base class's helpers, extract_teacher_forced is
     overridden here to synthesize a post-expansion input_ids
     (replicating the image token 576 times) BEFORE storing in the
     ExtractionResult. Forward pass still gets the original pre-
     expansion input_ids; only the stored copy is expanded so
     downstream slicing and the base class's content_token_mask
     work uniformly.

Default layers per docs/extraction-spec.md §Q1: L14–L18 prior; will be
re-frozen by the layer pilot result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.attn.base import AttentionExtractor, ExtractionResult, TeacherForcedInputs

if TYPE_CHECKING:
    from PIL.Image import Image


# The community HF-format conversion is the right load target. The
# microsoft/ original is the source of the weights but uses a
# model_type identifier transformers can't deserialize cleanly.
DEFAULT_MODEL_ID = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"


class LLaVAMedExtractor(AttentionExtractor):
    """Extractor for the HF-format LLaVA-Med-1.5.

    Backbone: Mistral-7B-Instruct + CLIP-ViT-L/14 @ 336².
    Native image grid: 24×24 (576 image tokens after model-internal
    expansion of a single <image> placeholder in input_ids).
    Default attention layers: 14–18 (pre-pilot; will be replaced by
    the pilot recommendation).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        super().__init__(model_id=model_id, native_grid=(24, 24))

    # ------------------------------------------------------------------ #

    def load(
        self,
        *,
        quant_config=None,
        dtype=None,
        attn_implementation: str = "eager",
    ) -> None:
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
        """Standard HF LLaVA exposes image_token_index on the config."""
        tok_id = getattr(self.model.config, "image_token_index", None)
        if isinstance(tok_id, int) and tok_id > 0:
            return tok_id
        candidate = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        unk = self.processor.tokenizer.unk_token_id
        if candidate is not None and candidate != unk:
            return candidate
        raise RuntimeError(
            "LLaVA-Med: could not resolve image_token_id from either "
            "model.config.image_token_index or the tokenizer's <image>."
        )

    # ------------------------------------------------------------------ #

    def prepare_inputs(
        self,
        image: "Image",
        ground_truth_report: str,
    ) -> TeacherForcedInputs:
        """Build inputs using the chat-template + processor two-step,
        matching the chaoyinshe model card recipe verbatim.

        Note: image-token expansion handling is NOT done here — the
        prep object holds pre-expansion coords. The post-expansion
        bookkeeping happens in the overridden extract_teacher_forced.
        """
        messages_user_only = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text",
                 "text": "Describe this chest radiograph. Cover the "
                         "lungs, pleura, heart and mediastinum, and bones / "
                         "soft tissues. Do not produce an impression."},
            ]},
        ]
        messages_full = messages_user_only + [
            {"role": "assistant",
             "content": [{"type": "text", "text": ground_truth_report}]},
        ]

        prompt_user = self.processor.tokenizer.apply_chat_template(
            messages_user_only, tokenize=False, add_generation_prompt=True,
        )
        prompt_full = self.processor.tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False,
        )

        inputs_user = self.processor(
            images=[image], text=prompt_user, return_tensors="pt",
        )
        inputs_full = self.processor(
            images=[image], text=prompt_full, return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        # Assistant range via longest-common-prefix between the two
        # tokenizations. Both share the same image processing so any
        # divergence is from the assistant content append.
        ids_full = inputs_full["input_ids"][0]
        ids_user = inputs_user["input_ids"][0].to(ids_full.device)
        prefix = min(ids_full.numel(), ids_user.numel())
        for i in range(prefix):
            if ids_full[i].item() != ids_user[i].item():
                prefix = i
                break
        if prefix == 0:
            raise RuntimeError(
                "LLaVA-Med: assistant-range diff failed — no common "
                "prefix between [user] and [user+assistant] tokenizations. "
                "Inspect the processor chat_template config."
            )
        end = ids_full.numel()
        eos = self.processor.tokenizer.eos_token_id
        while end > prefix and ids_full[end - 1].item() == eos:
            end -= 1

        # Image-token position in PRE-expansion input_ids — typically
        # exactly one position; the model expands it internally.
        img_token_id = self._resolve_image_token_id()
        img_pos_pre = (ids_full == img_token_id).nonzero(as_tuple=True)[0]
        if img_pos_pre.numel() == 0:
            raise RuntimeError(
                f"LLaVA-Med: no image-token (id={img_token_id}) in "
                f"input_ids of length {ids_full.numel()}. Check the "
                "chat template's image-placeholder handling."
            )

        return TeacherForcedInputs(
            inputs=dict(inputs_full),
            image_token_positions=img_pos_pre,
            assistant_range=(prefix, end),
        )

    # ------------------------------------------------------------------ #

    def extract_teacher_forced(
        self,
        image: "Image",
        ground_truth_report: str,
    ) -> ExtractionResult:
        """Run a forward pass, then translate pre-expansion coords to
        post-expansion coords so the returned ExtractionResult is
        consistent with the base class's slicing helpers.

        The model's forward internally replaces the single <image>
        token in input_ids with 576 image embeddings, so the
        attention tensors are over a sequence of length
        (input_ids_len + 575). The stored ExtractionResult uses a
        synthetic post-expansion input_ids (image token replicated
        576 times) so input_ids.shape[-1] matches attn.shape[-1] —
        base.content_token_mask and aggregate_to_grid then index
        consistently without any LLaVA-specific awareness.
        """
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError(f"{type(self).__name__}.load() first")

        prep = self.prepare_inputs(image, ground_truth_report)

        with torch.inference_mode():
            out = self.model(**prep.inputs, output_attentions=True, return_dict=True)
        if out.attentions is None or len(out.attentions) == 0:
            raise RuntimeError(
                "LLaVA-Med: out.attentions is None/empty. Confirm "
                "attn_implementation='eager' was passed at load time — "
                "SDPA silently drops attention weights."
            )

        ids_pre = prep.inputs["input_ids"][0]
        img_token_id = self._resolve_image_token_id()
        img_pos_in_pre = (ids_pre == img_token_id).nonzero(as_tuple=True)[0]
        expansion = self.native_grid[0] * self.native_grid[1]  # 576

        if img_pos_in_pre.numel() == 1:
            single_pos = int(img_pos_in_pre[0].item())
            # Synthetic post-expansion input_ids: replace the single
            # image token with `expansion` copies so length matches the
            # attention dims.
            ids_post = torch.cat([
                ids_pre[:single_pos],
                torch.full(
                    (expansion,), img_token_id,
                    dtype=ids_pre.dtype, device=ids_pre.device,
                ),
                ids_pre[single_pos + 1:],
            ])
            img_pos_post = torch.arange(
                single_pos, single_pos + expansion, device=ids_pre.device,
            )
            offset = expansion - 1
            a_start_pre, a_end_pre = prep.assistant_range
            a_start_post = a_start_pre + offset if a_start_pre > single_pos else a_start_pre
            a_end_post = a_end_pre + offset if a_end_pre > single_pos else a_end_pre
        elif img_pos_in_pre.numel() == expansion:
            # Already pre-expanded (some processor versions do this)
            ids_post = ids_pre
            img_pos_post = img_pos_in_pre
            a_start_post, a_end_post = prep.assistant_range
        else:
            raise RuntimeError(
                f"LLaVA-Med: unexpected image-token count "
                f"{img_pos_in_pre.numel()} in input_ids; expected 1 or "
                f"{expansion}."
            )

        # Sanity check: post-expansion length must match attention dims.
        attn_seq_len = out.attentions[0].shape[-1]
        if ids_post.numel() != attn_seq_len:
            raise RuntimeError(
                f"LLaVA-Med: post-expansion ids length ({ids_post.numel()}) "
                f"does not match attention sequence length ({attn_seq_len}). "
                "Image-expansion math is off; inspect the model's forward "
                "expansion factor."
            )

        return ExtractionResult(
            attentions=tuple(out.attentions),
            input_ids=ids_post.unsqueeze(0),
            image_token_positions=img_pos_post,
            assistant_range=(a_start_post, a_end_post),
            native_grid=self.native_grid,
            generated_text=ground_truth_report,
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------ #

    def default_layers(self) -> list[int]:
        """Pre-pilot prior: L14–L18 from LLaVA-1.5-7B interpretability
        literature. Will be overridden by the pilot's recommendation
        once docs/extraction-spec.md §Q1 (LLaVA-Med row) is frozen."""
        return [14, 15, 16, 17, 18]

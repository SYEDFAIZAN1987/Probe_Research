# Per-model prompt templates

Recommended inference prompts pulled from each model's official model
card. **Treat as starting points** — the final templates lock in week 1
after a 10-case validation that the model produces a coherent report on
a known CXR. Document any deviations from this file before running the
full 3,032-case pass.

---

## LLaVA-Med-1.5 (`microsoft/llava-med-v1.5-mistral-7b`)

**Backbone.** Mistral-7B-Instruct. Vision: CLIP-ViT-L/14 @ 336².

**Conversation template.** `mistral_instruct` mode in the LLaVA
library's `llava/conversation.py`. The HF model card does not document
this verbatim. Day-1 verification: load
`from llava.conversation import conv_templates;
conv_templates["mistral_instruct"]` and print, then mirror exactly.

**Expected format** (Mistral instruct convention, to confirm on load):
```
<s>[INST] <image>
{system_or_role_instruction}
{user_question} [/INST]
```

**Working starting prompt for CXR report generation:**
```
<image>
You are an expert radiologist. Read the chest radiograph and produce a
concise findings report covering the lungs, pleura, heart and
mediastinum, and bones / soft tissues. Do not add an impression.
```

**Generation arguments (starting):**
```python
generate(
    max_new_tokens=300,
    do_sample=False,
    temperature=None,
    top_p=None,
)
```

**Quirk.** LLaVA-Med-1.5's `pipeline("image-text-to-text", ...)`
example in the HF card uses an OpenAI-style `messages` list. The
underlying conversation template is still `mistral_instruct`; the
pipeline just wraps it. For attention-extraction work where we need
deterministic token positions, **call the model and processor
directly** rather than through the pipeline wrapper.

---

## MedGemma-4B (`google/medgemma-4b-it`)

**Backbone.** Gemma-3-4B. Vision: MedSigLIP-400M @ 896² → 256 image
tokens (16×16 grid).

**Access.** Gated. Accept the [Health AI Developer Foundation's terms
of use](https://developers.google.com/health-ai-developer-foundations/terms)
on the HF page before first download.

**Recommended messages format (verbatim from model card):**
```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert radiologist."}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this X-ray"},
            {"type": "image", "image": image},   # PIL Image
        ],
    },
]
```

**Processor + generate (verbatim):**
```python
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    out = out[0][input_len:]
decoded = processor.decode(out, skip_special_tokens=True)
```

**Loading.** Load in bf16 (no quantization — `torch_dtype=torch.bfloat16`
on `from_pretrained`). L40S 48 GB has plenty of headroom for a 4 B
model in bf16 (~8 GB weights + activations).

**Reported headline.** Card reports RadGraph-F1 = 21.9 on MIMIC-CXR
out-of-the-box; rises to 30.3 after MIMIC fine-tuning. We expect
similar or slightly lower numbers on VinDr-CXR (Vietnamese-cohort
distribution shift).

---

## MAIRA-2 (`microsoft/maira-2`)

**Backbone.** Vicuna-7B + RAD-DINO ViT-B (87M params), pretrained via
DINOv2 on CXRs.

**Recommended call (verbatim from model card):**
```python
processed_inputs = processor.format_and_preprocess_reporting_input(
    current_frontal=sample_data["frontal"],
    current_lateral=sample_data["lateral"],          # set None if absent
    prior_frontal=None,
    indication=sample_data["indication"],            # str or None
    technique=sample_data["technique"],              # str or None
    comparison=sample_data["comparison"],            # str or None
    prior_report=None,
    return_tensors="pt",
    get_grounding=True,                              # ← bbox output
)

processed_inputs = processed_inputs.to(device)
with torch.no_grad():
    out = model.generate(
        **processed_inputs,
        max_new_tokens=450,                          # 450 for grounded; 300 ungrounded
        use_cache=True,
    )
prompt_len = processed_inputs["input_ids"].shape[-1]
decoded = processor.decode(out[0][prompt_len:], skip_special_tokens=True).lstrip()
prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
```

**Quirks (critical).**
1. **`get_grounding=True` is required** for the second-grounding-signal
   experiment (see [`extraction-spec.md`](extraction-spec.md) §Q0).
   Without it, MAIRA-2 outputs plain text and we lose the bbox channel.
2. **VinDr-CXR has only frontal views** — pass `current_lateral=None`
   and verify the processor handles it. If a lateral is silently
   required, this is a week-1 blocker for MAIRA-2 specifically.
3. **Indication / technique / comparison fields** are part of MAIRA-2's
   input schema. VinDr-CXR provides only bbox annotations + class
   labels, not structured clinical-context fields. Decision: pass
   empty strings to all three; document this as a deviation from
   MAIRA-2's intended use. This may suppress some MAIRA-2 outputs vs.
   its training distribution — acknowledge in the limitations section.
4. **No system prompt or chat template.** The processor handles
   formatting end-to-end. Do not attempt to inject a system prompt.

**Output parsing.** `convert_output_to_plaintext_or_grounded_sequence`
returns a list of `(text, bboxes_or_None)` tuples — bbox is
`(x1, y1, x2, y2)` relative to the **cropped image**, NOT the original.
Day-1 task: write a `bbox_to_native_coords()` helper that takes the
processor's crop metadata and unprojects back to native image
coordinates so they're directly comparable to VinDr-CXR radiologist
bboxes (which are in original-image pixel coordinates).

---

## Cross-model normalization for the audit

Per the extraction spec, after each model produces its report we extract
attention as documented in `docs/extraction-spec.md`. The prompt
templates above are **the inputs**; the attention extraction is the
read on the **internal state** that results. These should be kept
separate in the code: `src/inference/prompts.py` owns prompts,
`src/attn/extract_<model>.py` owns extraction hooks.

## Day-1 validation checklist

For each model, on the first 10 VinDr-CXR cases:

- [ ] Loads on a 48 GB L40S in bf16 without OOM
- [ ] Produces a non-empty report on a known CXR (e.g., a clear pneumonia
      or pleural effusion case from VinDr)
- [ ] Attention tensor of expected shape comes out of the forward hook
- [ ] Generation is deterministic across 3 reruns (`do_sample=False`,
      fixed seed)
- [ ] For MAIRA-2 specifically: at least one bbox token appears in the
      output of a positive-finding case
- [ ] DICOM→PNG conversion artifacts (if any) don't visibly degrade
      report quality vs. native DICOM input (check on 1-2 cases manually)

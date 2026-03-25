

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

import config


def load_tokenizer(model_id: str = config.BASE_MODEL_ID) -> AutoTokenizer:
    """
    Load the tokenizer for the given model.
    Sets pad_token = eos_token when no pad token is defined (common for
    decoder-only models like TinyLlama).
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model_quantised(model_id: str = config.BASE_MODEL_ID) -> AutoModelForCausalLM:
    """
    Load the base causal-LM in 8-bit mode to reduce GPU memory usage.
    Requires bitsandbytes to be installed.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
    )
    return base_model


def load_instruction_model(
    base_model_id: str = config.BASE_MODEL_ID,
    sft_checkpoint: str = config.INSTRUCTION_CKPT_DIR,
) -> AutoModelForCausalLM:
    """
    Load the instruction-tuned model by:
      1. Loading the quantised base model.
      2. Attaching the SFT LoRA adapter from `sft_checkpoint`.
      3. Merging the adapter weights back into the base model so that
         a fresh LoRA can be applied on top for DPO.
    """
    base = load_base_model_quantised(base_model_id)
    peft_model = PeftModel.from_pretrained(base, sft_checkpoint)
    merged_model = peft_model.merge_and_unload()
    return merged_model


def build_lora_config() -> LoraConfig:
    """Return a LoRA configuration built from config.py values."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )


def attach_lora_for_dpo(merged_model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Wrap a fully-merged model with a new LoRA adapter ready for DPO training.
    Returns the PEFT-wrapped model.
    """
    lora_cfg = build_lora_config()
    trainable_model = get_peft_model(merged_model, lora_cfg)
    trainable_model.print_trainable_parameters()
    return trainable_model


def load_aligned_model_for_inference(
    checkpoint_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> AutoModelForCausalLM:
    """
    Load a fully saved DPO-aligned model (non-quantised) for inference.
    Moves the model to `device` and returns it.
    """
    aligned_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
    )
    aligned_model.to(device)
    return aligned_model

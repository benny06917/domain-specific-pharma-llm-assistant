
import argparse
import torch
from transformers import AutoModelForCausalLM

import config
from model_utils import load_tokenizer, load_aligned_model_for_inference


#evaluation prompt 
DEFAULT_EVAL_PROMPT = (
    "Explain how artificial intelligence is improving the process of "
    "drug discovery and development in the pharmaceutical industry."
)


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> str:
    """
    Run greedy-sample generation and return the decoded output
    (with the prompt stripped from the beginning).
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        token_ids = model.generate(
            **encoded,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            repetition_penalty=config.REPETITION_PENALTY,
        )

    full_text   = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    # Strip the prompt prefix so only the generated reply is shown
    reply_only  = full_text[len(prompt):].strip()
    return reply_only


def run_comparison(prompt: str, aligned_ckpt: str) -> None:
    tokenizer = load_tokenizer()
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("PROMPT")
    print("=" * 70)
    print(prompt)

    #SFT baseline
    print("\n" + "─" * 70)
    print("INSTRUCTION-TUNED MODEL (before DPO)")
    print("─" * 70)
    sft_model = load_aligned_model_for_inference(
        config.INSTRUCTION_CKPT_DIR,
        dtype=torch.float16,
        device=device,
    )
    sft_reply = generate_response(sft_model, tokenizer, prompt, device)
    print(sft_reply)
    #free RAM before loading next model
    del sft_model 

    #DPO-aligned model
    print("\n" + "─" * 70)
    print("DPO PREFERENCE-ALIGNED MODEL")
    print("─" * 70)
    aligned_model = load_aligned_model_for_inference(
        aligned_ckpt,
        dtype=torch.float16,
        device=device,
    )
    aligned_reply = generate_response(aligned_model, tokenizer, prompt, device)
    print(aligned_reply)
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO inference comparison")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_EVAL_PROMPT,
        help="Evaluation prompt (default: pharma AI question)",
    )
    parser.add_argument(
        "--aligned_ckpt",
        type=str,
        default=f"{config.DPO_OUTPUT_DIR}/checkpoint-1",
        help="Path to the saved DPO-aligned checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args.prompt, args.aligned_ckpt)

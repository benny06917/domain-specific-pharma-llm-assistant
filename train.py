
import os
import config  # sets WANDB_DISABLED as a side-effect

from trl import DPOTrainer, DPOConfig

from model_utils import load_tokenizer, load_instruction_model, attach_lora_for_dpo
from data_utils import load_preference_dataset


def build_dpo_training_args() -> DPOConfig:
    """Construct DPOConfig from values in config.py."""
    return DPOConfig(
        output_dir=config.DPO_OUTPUT_DIR,
        learning_rate=config.DPO_LEARNING_RATE,
        per_device_train_batch_size=config.DPO_BATCH_SIZE,
        gradient_accumulation_steps=config.DPO_GRAD_ACCUMULATION_STEPS,
        num_train_epochs=config.DPO_NUM_EPOCHS,
        beta=config.DPO_BETA,
        loss_type=config.DPO_LOSS_TYPE,
        remove_unused_columns=False,
        report_to=None,
        logging_dir=None,
    )


def run_dpo_training() -> None:
    print("=" * 60)
    print("  DPO Preference Alignment — TinyLlama")
    print("=" * 60)

    #Tokenizer
    print("\n[1/4] Loading tokenizer …")
    tokenizer = load_tokenizer()

    #Model
    print("[2/4] Loading instruction-tuned model and merging adapter …")
    merged = load_instruction_model()

    print("[3/4] Attaching LoRA adapter for DPO …")
    trainable_model = attach_lora_for_dpo(merged)

    #dataset
    preference_data = load_preference_dataset()

    #Training
    dpo_args = build_dpo_training_args()

    trainer = DPOTrainer(
        model=trainable_model,
        ref_model=None,          
        args=dpo_args,
        train_dataset=preference_data,
        processing_class=tokenizer,
    )

    trainer.train()

    os.makedirs(config.DPO_OUTPUT_DIR, exist_ok=True)
    trainer.save_model(config.DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(config.DPO_OUTPUT_DIR)
    print(f"\n✓ Aligned model saved to '{config.DPO_OUTPUT_DIR}'")


if __name__ == "__main__":
    run_dpo_training()

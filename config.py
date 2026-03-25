

import os

#model
BASE_MODEL_ID        = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
INSTRUCTION_CKPT_DIR = "./checkpoints/instruction-tuned"   # local path to SFT checkpoint

#Dataset
PREFERENCE_DATA_CSV  = "./data/pharma_preference_data.csv"

#LoRA
LORA_RANK            = 8
LORA_ALPHA           = 16
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = ["q_proj", "v_proj"]

#DPO Training
DPO_OUTPUT_DIR               = "./outputs/dpo-aligned"
DPO_LEARNING_RATE            = 2e-5
DPO_BATCH_SIZE               = 1
DPO_GRAD_ACCUMULATION_STEPS  = 8
DPO_NUM_EPOCHS               = 1
DPO_BETA                     = 0.1          # KL-penalty coefficient
DPO_LOSS_TYPE                = "sigmoid"

#Generation
MAX_NEW_TOKENS      = 200
TEMPERATURE         = 0.7
TOP_P               = 0.9
REPETITION_PENALTY  = 1.1

os.environ["WANDB_DISABLED"] = "true"   # disable W&B logging

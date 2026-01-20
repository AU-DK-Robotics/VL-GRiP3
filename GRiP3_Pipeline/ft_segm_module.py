import os
import json
from typing import List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig

# ------------------------------------------------------------------------------
# DATASET PATHS
# ------------------------------------------------------------------------------
TRAIN_IMAGES_DIR = "/GRiP3_Pipeline/dataset/training_dataset_segmentation"  # <-- modificali col tuo path
TRAIN_ANNOTATIONS = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/dataset/training_dataset_segmentation/annotations_train.jsonl"

VALID_IMAGES_DIR = "/GRiP3_Pipeline/dataset/training_dataset_segmentation"
VALID_ANNOTATIONS = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/dataset/training_dataset_segmentation/annotations_valid.jsonl"

# ------------------------------------------------------------------------------
# LOCAL JSONL DATASET CLASS
# ------------------------------------------------------------------------------
class JSONLDataset(Dataset):
    """
    Reads a JSONL file and loads images from the local folder.
    Each line must contain:
    {
      "image": "filename.jpg",
      "prefix": "some prefix text",
      "suffix": "some suffix text"
    }
    """
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        super().__init__()
        self.records = []
        self.image_dir = image_directory_path

        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line.strip())
                    self.records.append(data)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        image_name = item["image"]
        prefix = item["prefix"]
        suffix = item.get("suffix", "")
        img_path = os.path.join(self.image_dir, image_name)
        pil_image = Image.open(img_path).convert("RGB")
        return {
            "image": pil_image,
            "prefix": prefix,
            "suffix": suffix
        }

# ------------------------------------------------------------------------------
# LOAD LOCAL DATASETS
# ------------------------------------------------------------------------------
train_dataset = JSONLDataset(jsonl_file_path=TRAIN_ANNOTATIONS, image_directory_path=TRAIN_IMAGES_DIR)
valid_dataset = JSONLDataset(jsonl_file_path=VALID_ANNOTATIONS, image_directory_path=VALID_IMAGES_DIR)

# ------------------------------------------------------------------------------
# LOAD PROCESSOR
# ------------------------------------------------------------------------------
model_id = "google/paligemma-3b-mix-448"  # Scegli il modello PaliGemma desiderato
processor = PaliGemmaProcessor.from_pretrained(model_id)

# ------------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTION
# ------------------------------------------------------------------------------
def collate_fn(examples: List[Dict[str, Any]]):
    # Prepend the <image> token to signal multi-modal input.
    texts = [f"<image>{ex['prefix']}" for ex in examples]
    images = [ex["image"] for ex in examples]
    targets = [ex["suffix"] for ex in examples]

    batch = processor(
        text=texts,
        images=images,
        suffix=targets,
        return_tensors="pt",
        padding="longest"
    )

    # Convert floating-point tensors to bfloat16 (if GPU supports) and move to device.
    for key, value in batch.items():
        if value.dtype in [torch.float32, torch.float64]:
            batch[key] = value.to(torch.bfloat16)
        batch[key] = batch[key].to(device)
    return batch

# ------------------------------------------------------------------------------
# LORA / QLoRA CONFIGURATION AND DEVICE SETUP
# ------------------------------------------------------------------------------
USE_LORA = True    # Cambia a True se vuoi LoRA
USE_QLORA = False    # Cambia a True se vuoi QLoRA
FREEZE_VISION = True  # Congela la vision tower

device = "cuda" if torch.cuda.is_available() else "cpu"

# Definiamo una config LoRA (vale sia per LoRA che QLoRA)
lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj", "o_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)

if USE_LORA or USE_QLORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
    else:
        # LoRA in full precision
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    # Avvolgiamo il modello con la LoRA
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()
else:
    # Nessun LoRA
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)

# ------------------------------------------------------------------------------
# OPZIONE: Congelare la vision tower, ma lasciare allenabile il multi_modal_projector
# ------------------------------------------------------------------------------
if FREEZE_VISION and hasattr(model, "vision_tower"):
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    # IMPORTANTE: lasciare sbloccato (non congelo) multi_modal_projector,
    # per adattare le feature al tuo caso di segmentazione
    # if hasattr(model, "multi_modal_projector"):
    #     for param in model.multi_modal_projector.parameters():
    #         param.requires_grad = False  # <-- COMMENTATO PER LASCIARLO SBLOCCATO

# ------------------------------------------------------------------------------
# TRAINING ARGUMENTS
# ------------------------------------------------------------------------------
args = TrainingArguments(
    num_train_epochs=15,                  # 10-20 epoche su dataset piccolo
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,        # per un eff. batchsize maggiore
    warmup_steps=50,                      # un po' di warmup in più su dataset piccolo
    learning_rate=1e-5,                   # LR più basso per stabilità
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=10,
    # Se stai usando QLoRA, potresti cambiare in: optim="paged_adamw_8bit"
    optim="adamw_hf",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    push_to_hub=True,
    output_dir="checkpoints/paligemma-segm-module",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# ------------------------------------------------------------------------------
# CREATE TRAINER
# ------------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    args=args
)

# ------------------------------------------------------------------------------
# START TRAINING
# ------------------------------------------------------------------------------
torch.cuda.empty_cache()
print("Starting training...")
trainer.train()
print("Training completed.")

#TUNING CUSTOM PART FOR OBJ SEGMENTATION
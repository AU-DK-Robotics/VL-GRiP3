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
TRAIN_IMAGES_DIR = "/GRiP3_Pipeline/utils/dataset/train_basic_cmd"
TRAIN_ANNOTATIONS = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/utils/dataset/train_basic_cmd/annotations_train.jsonl"

VALID_IMAGES_DIR = "/GRiP3_Pipeline/utils/dataset/train_basic_cmd"
VALID_ANNOTATIONS = "/home/au-robotics/MircoProjects/VL_GRiP3/GRiP3_Pipeline/utils/dataset/train_basic_cmd/annotations_valid.jsonl"

# ------------------------------------------------------------------------------
# LOCAL JSONL DATASET CLASS
# ------------------------------------------------------------------------------
class JSONLDataset(Dataset):
    """
    Legge un file JSONL e carica le immagini dalla cartella locale.
    Ogni riga deve contenere:
    {
      "image": "filename.jpg",
      "prefix": "testo di input",
      "suffix": "output atteso (token di segmentazione + comandi robotici)"
    }
    """
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        super().__init__()
        self.records = []
        self.image_dir = image_directory_path

        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
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
model_id = "google/paligemma-3b-mix-448"  # Scegli il modello desiderato
processor = PaliGemmaProcessor.from_pretrained(model_id)

# ------------------------------------------------------------------------------
# CUSTOM COLLATE FUNCTION
# ------------------------------------------------------------------------------
def collate_fn(examples: List[Dict[str, Any]]):
    # Aggiungi il token <image> all'inizio del testo per indicare input multimodale.
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

    # Converti i tensori floating-point in bfloat16 e sposta tutto sul device.
    for key, value in batch.items():
        if value.dtype in [torch.float32, torch.float64]:
            batch[key] = value.to(torch.bfloat16).to(device)
        else:
            batch[key] = value.to(device)
    return batch

# ------------------------------------------------------------------------------
# LORA / QLoRA CONFIGURATION AND DEVICE SETUP
# ------------------------------------------------------------------------------
USE_LORA = False
USE_QLORA = True   # Usa QLoRA se desiderato
# Imposta FREEZE_VISION a True per congelare la parte visiva (non allenare la segmentazione)
FREEZE_VISION = True
device = "cuda" if torch.cuda.is_available() else "cpu"

if USE_LORA or USE_QLORA:
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
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
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()
else:
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)

# ---- FREEZING DEL MODULO VISIVO ----
if FREEZE_VISION and hasattr(model, "vision_tower"):
    # Congela la vision tower: le feature visive non verranno aggiornate
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    # Se esiste, congela anche il multi_modal_projector
    if hasattr(model, "multi_modal_projector"):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

# ------------------------------------------------------------------------------
# TRAINING ARGUMENTS
# ------------------------------------------------------------------------------
args = TrainingArguments(
    num_train_epochs=10,                        # Dataset piccolo, quindi meno epoch
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,              # Batch virtuale più grande per stabilità
    warmup_steps=20,                            # Warmup più lungo per una transizione graduale
    learning_rate=3e-5,                         # Learning rate leggermente ridotto
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=10,
    optim="adamw_hf",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    push_to_hub=True,                         # Se non vuoi pubblicare su hub, metti False
    output_dir="/checkpoints/paligemma-action-module",
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


#FINE-TUNING ROBOTIC COMMAND SCRIPT.
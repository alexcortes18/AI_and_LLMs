import os
import torch
import numpy as np
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

model_checkpoint = "distilbert-base-uncased"

def print_model_size(path):
    size = 0
    for f in os.scandir(path):
        size += os.path.getsize(f)
    print(f"Model size: {(size / 1e6):.2} MB")


def print_trainable_parameters(model, label):
    parameters, trainable = 0, 0
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    print(
        f"{label} trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)"
    )
    



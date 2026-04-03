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
    
def build_lora_model(num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels = num_labels
    )
    print_trainable_parameters(model, label="Base model")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_lin","v_lin","k_lin","out_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    lora_model = get_peft_model(model,lora_config)
    print_trainable_parameters(lora_model, label="LoRA")
    return lora_model

def preprocess_function(examples, tokenizer):
    # Process Text
    texts = [str(text).lower().strip() for text in examples["text"]]
    
    # Tokenize
    result = tokenizer(
        texts,
        truncation = True,
        padding = True,
        max_length = 128,
        return_tensors = None, # Changed this to return lists
    )
    
    # Add labels
    result["labels"] = examples["labels"]
    return result

if __name__ == "__main__":
    print("Starting LoRA fine-tuning demo...")
    
    model_checkpoint = "distilbert-base-uncased"
    print(f"Using model: {model_checkpoint}")
    
    # Load datasets
    print("\n Loading datasets...")
    dataset1 = load_dataset("imdb", split="train[:1000]")
    dataset2 = load_dataset("ag_news", split="train[:1000]")
    
    print(dataset1[0])
    
    print(f"Dataset 1 size: {len(dataset1)} examples")
    print(f"Dataset 2 size: {len(dataset2)} examples")
    
    # Prepare datasets
    # Hugging Face Trainer expects the target field to be named labels.
    dataset1 = dataset1.rename_column("label", "labels")
    dataset2 = dataset2.rename_column("label", "labels")
    
    train_size = int(0.8 * len(dataset1))
    dataset1_train = dataset1.select(range(train_size))
    dataset1_test = dataset1.select(range(train_size, len(dataset1)))
    dataset2_train = dataset2.select(range(train_size))
    dataset2_test = dataset2.select(range(train_size, len(dataset2)))
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    config = {
        "sentiment":{
            "train_data": dataset1_train,
            "test_data": dataset1_test,
            "num_labels": 2,
            "epochs": 5,
            "path": "./lora-sentiment"
        },
        "topic":{
            "train_data": dataset2_train,
            "test_data": dataset2_test,
            "num_labels": 4,
            "epochs": 5,
            "path": "./lora-topic",
        },
    }
    
    # Preprocess datasets
    print("Preprocessing datasets")
    for cfg in config.values():
        cfg["train_data"] = cfg["train_data"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched = True,
            remove_columns = ["text"] # only remove text column
        )
        
        cfg["test_data"] = cfg["test_data"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched = True,
            remove_columns = ["text"] # only remove text column
        )
        # Set format for pytorch
        # change from this: "input_ids": [101, 2023, 2003, 1037, 2742, 102],
        # into this: "input_ids": tensor([101, 2023, 2003, 1037, 2742, 102]),
        # and also for "attention_mask" and "labels"
        cfg["train_data"].set_format("torch") 
        cfg["test_data"].set_format("torch")
        
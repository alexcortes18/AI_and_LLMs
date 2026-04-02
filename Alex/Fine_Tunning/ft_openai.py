# pip install openai
# pip install openai[datalib]
# pip install urllib3
# pip install python-dotenv
# pip install tiktoken

import io
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import openai
import pandas as pd
import pprint

_ = load_dotenv(find_dotenv()) 

client = OpenAI(
    api_key= os.environ["OPENAI_API_KEY"] # optional since OpenAI client can find the key by itself.
)

# Helper functions:
import json
import tiktoken # for token counting
from collections import defaultdict

encoding = tiktoken.get_encoding("cl100k_base")

def json_to_jsonl(input_file, output_file):
    """ Convert a normal json file into a file used for openai fine-tune training"""
    # Open JSON file
    f = open(input_file)
    
    #.load returns json object as a Python Object
    data = json.load(f)
    # pprint.pprint(data)
    
    # produce JSONL from JSON
    with open(output_file, "w") as outfile:
        for entry in data:
            # print(entry)
            json.dump(entry, outfile)
            outfile.write("\n")
            
# To check whether your fine-tuning training data has the expected chat format before you upload it. 
def check_file_format(dataset):
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call") for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")
        
# To count the estimated toke usage for OpenAI API costs.
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

json_to_jsonl("Alex/Fine_Tunning/teachrafter.json","Alex/Fine_Tunning/output.jsonl")

# Check file format:
data_path = "Alex/Fine_Tunning/output.jsonl"

# Load the dataset from:https://cookbook.openai.com/examples/chat_finetuning_data_prep
with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]
    
# Initial dataset stats:
# print("Num examples:", len(dataset))
# print("First example:")
# for message in dataset[0]["messages"]:
    # print(message)
    
check_file_format(dataset)
# print(f"Length of the dataset is: {len(dataset)}")

# Cost estimations
# Get the length of the conversation
conversation_length = []

for msg in dataset:
    messages = msg["messages"]
    conversation_length.append(num_tokens_from_messages(messages))

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096
TARGET_EPOCHS = 5
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)

if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(
    min(MAX_TOKENS_PER_EXAMPLE, length) for length in conversation_length
)
print(
    f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
)
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(
    f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
)

num_tokens = n_epochs * n_billing_tokens_in_dataset

# Current training estimate using gpt-4o-mini-2024-07-18 pricing: $3.00 / 1M tokens
TRAINING_MODEL = "gpt-4o-mini-2024-07-18"
TRAINING_COST_PER_1M_TOKENS = 3.00

cost = (num_tokens / 1_000_000) * TRAINING_COST_PER_1M_TOKENS
print(f"Estimated training cost for {TRAINING_MODEL}: ${cost:.6f} USD")


# UPLOAD FILES TO OPENAI:
# Upload file once all validations are successful!
# Each time you run this code, it will create another file in your platform.openai.com account, 
# in your Storage section.
# ---------------------------------------------------------------------------------------------------------------------------------
# training_file = ""
# training_file = client.files.create(
#     file=open(data_path, "rb"), purpose="fine-tune"
# )
# print(f"Training file id: {training_file.id}")
# ---------------------------------------------------------------------------------------------------------------------------------

# == Next steps: Create a fine-tuned model ===
# Start the fine-tuning job
# After you've started a fine-tuning job, it may take some time to complete. Your job may be queued
# behind other jobs and training a model can take minutes or hours depending on the
# model and dataset size.
# ---------------------------------------------------------------------------------------------------------------------------------
# response = client.fine_tuning.jobs.create(
#     training_file=training_file.id,
#     model=TRAINING_MODEL,
#     hyperparameters={
#         "n_epochs": 5
#     }
# )
# print(f"Response id from creating the job: {response.id}") 
# ---------------------------------------------------------------------------------------------------------------------------------
# Retrieve the state of a fine-tune
# Status field can contain: running or succeeded or failed, etc.
# state = client.fine_tuning.jobs.retrieve(response.id) #ftjob-0w4vss9dD82XinmEKTpGxrab
state = client.fine_tuning.jobs.retrieve("ftjob-0w4vss9dD82XinmEKTpGxrab")
pprint.pprint(f"Fine-tuning job is running{state}")
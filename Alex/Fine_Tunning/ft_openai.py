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
            json.dump(entry, outfile)
            outfile.write("\n")
            
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
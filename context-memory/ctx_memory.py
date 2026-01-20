import json
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def initialize_client(use_ollama: bool = False) -> OpenAI:
    "Initialize the OpenAI client for either Ollama or OpenAI"
    
    if use_ollama:
        return OpenAI(base_url="http:/localhost:14434/v1",api_key="ollama")
    return OpenAI()

def create_initial_message() -> List[Dict[str,str]]:
    """Creates the initial message from the chatbot"""
    return [
        {"role":"system", "content":"You are a helpful assistant"},
    ]
    
def chat(user_input: str, messages: List[Dict[str,str]],  client: OpenAI, model_name: str) -> str:
    """Handle user input and generate responses"""
    # Append the user's response:
    messages.append({"role":"user", "content":user_input})
    
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        assistant_response = response.choices[0].message.content
        messages.append({"role":"system", "content":assistant_response})
        
        return assistant_response
        
    except Exception as e:
        return f"Error: {str(e)}"
    
def summarize_messages(messages: List[Dict[str,str]]) -> List[Dict[str,str]]:
    """ Summarize older messages to save tokens"""
    summary = "Previous conversation summarized: " + "".join(
        [m["content"] + "..." for m in messages[:-5]]
    )
    return [{"role":"system", "content":summary}] + messages[-5:]

def save_conversation(messages: List[Dict[str,str]], filename: str = "conversation.json"):
    """ Save the conversation as a json file to read it afterwards"""
    with open(filename, mode="w") as f:
        json.dump(messages,f, indent= 2)

def load_conversation(filename: str="conversation.json") -> List[Dict[str,str]]:
    """ Opens or loads the saved conversation so far """
    try:
        with open(filename, "w") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No conversation found at {filename}")
        return create_initial_message()
    
def main():
    # Model selection
    print("Select model type:")
    print("1. OpenAI GPT-4")
    print("2. Ollama (Local)")
    
    choice = input("Enter choice (1 or 2): ")
    use_ollama = choice == 2
    
    # Initialize client, model's name, and messages
    client = initialize_client(use_ollama)
    model_name = "llama3.2" if use_ollama else "gpt-4o-mini"
    messages = create_initial_message()
    
    # Give list of commands
    print(f"\nUsing {'Ollama' if use_ollama else 'OpenAI'} model. Type 'quit' to exit.")
    print("Available commands:")
    print("- 'save': Save conversation")
    print("- 'load': Load conversation")
    print("- 'summary': Summarize conversation")
    print("- 'quit / exit': Exit conversation")
    
    # Go through loop to simulate chat
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit" or user_input.lower == "quit":
            break
        elif user_input.lower() == "save":
            save_conversation(messages)
            print("Conversation saved!")
            continue
        elif user_input.lower() == "load":
            load_conversation()
            print("Conversation loaded!")
            continue
        elif user_input.lower() == "summary":
            messages = summarize_messages(messages)
            # print("Conversation summarized!:", messages)
            print("\n --------------------------------------------------")
            
        response = chat(user_input=user_input, messages= messages, client= client, model_name= model_name)
        print("Assistant's response: ", response)
        
        # Summarize if it gets too long:
        if len(messages) > 10:
            messages = summarize_messages(messages=messages)

# Example usage:~
if __name__ == "__main__":
    main()
            
        
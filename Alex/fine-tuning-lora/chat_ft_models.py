import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import os, sys, time

class ModelInterface:
    def __init__(self):
        self.tokenizer = None
        self.sentiment_model = None
        self.topic_model = None
        self.model_checkpoint = "distilbert-base-uncased"
        
    def load_models(self):
        """Load tokenizer and models with error handling"""
        try:
            print("Loading models")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            
            # Load setiment model
            sentiment_path = "./Alex/fine-tuning-lora/lora-sentiment"
            if os.path.exists(sentiment_path):
                base_model_sentiment = AutoModelForSequenceClassification.from_pretrained(
                    self.model_checkpoint, num_labels = 2)
                
                self.sentiment_model = PeftModel.from_pretrained(base_model_sentiment, sentiment_path)
                
                self.sentiment_model.eval() # Sets the model to evaluation mode.
            else:
                raise FileNotFoundError(
                    "Sentiment model not found. Please train the model first or find the correct path."
                )
            
            topic_path = "./Alex/fine-tuning-lora/lora-topic"
            if os.path.exists(topic_path):
                base_model_topic = AutoModelForSequenceClassification.from_pretrained(
                    self.model_checkpoint, num_labels = 4
                )
                self.topic_model = PeftModel.from_pretrained(
                    base_model_topic, topic_path
                )
                self.topic_model.eval()
            else:
                raise FileNotFoundError(
                    "Topic model not found. Please train the model first or find the correct path."
                )
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def analyze_text(self, text, model_type="both"):
        """Analyze text with specified model type"""
        if not text.strip():
            return None
        
        inputs = self.tokenizer(
            text.lower().strip(),
            return_tensors = "pt",
            truncation = True,
            max_length = 128
        )
        
        results = {}
        
        try:
            if model_type in ["sentiment", "both"]:
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    sentiment_probs = torch.nn.functional.softmax(
                        outputs.logits, dim = -1
                    )
                    sentiment_predictions = torch.argmax(sentiment_probs, dim = -1).item()
                    sentiment_confidence = sentiment_probs[0][sentiment_predictions].item()
                    
                    sentiment_map = {
                        0: "Negative",
                        1: "Postive"
                    }
                    results["sentiment"] = {
                        "prediction": sentiment_map[sentiment_predictions],
                        "confidence": sentiment_confidence
                    }
            
            if model_type in ["topic", "both"]:
                with torch.no_grad():
                    outputs = self.topic_model(**inputs)
                    topic_probs = torch.nn.functional.softmax(
                        outputs.logits, dim = -1
                    )
                    topic_predictions = torch.argmax(topic_probs, dim = -1).item()
                    topic_confidence = topic_probs[0][topic_predictions].item()
                    
                    topic_map = {
                        0: "World",
                        1: "Sports",
                        2: "Business",
                        3: "Science/Technology",
                    }
                    
                    results["topic"] = {
                        "prediction" : topic_map[topic_predictions],
                        "confidence" : topic_confidence
                    }
            return results
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return None

def print_welcome():
    """Print welcome message and instructions"""
    print("\n" + "=" * 50)
    print("Welcome to the LoRA Model Chat Interface!")
    print("=" * 50)
    print("\nThis interface allows you to analyze text for:")
    print("1. Sentiment Analysis (Positive/Negative)")
    print("2. Topic Classification (World/Sports/Business/Science)")
    print("\nCommands:")
    print("- Type 'quit' or 'exit' to end the session")
    print("- Type 'switch' to change analysis mode")
    print("- Type 'help' to see these instructions again")
    print("=" * 50)

def chat_interface():
    """Main chat interface function"""
    interface = ModelInterface()

    print("Initializing models...")
    if not interface.load_models():
        print("Failed to load models. Exiting...")
        return

    print_welcome()
    mode = "both"

    while True:
        try:
            print(f"\nCurrent mode: {mode}")
            text = input("\nEnter text to analyze (or command): ").strip()

            # Handle commands
            if text.lower() in ["quit", "exit"]:
                print("\nThank you for using the LoRA Model Chat Interface!")
                break

            if text.lower() == "help":
                print_welcome()
                continue

            if text.lower() == "switch":
                while True:
                    new_mode = input("Enter new mode (sentiment/topic/both): ").lower()
                    if new_mode in ["sentiment", "topic", "both"]:
                        mode = new_mode
                        print(f"Switched to {mode} mode")
                        break
                    else:
                        print("Invalid mode. Please try again.")
                continue

            if not text:
                continue

            # Analyze text
            print("\nAnalyzing...")
            results = interface.analyze_text(text, mode)

            if results:
                print("\nAnalysis Results:")
                if "sentiment" in results:
                    sent_result = results["sentiment"]
                    print(f"Sentiment: {sent_result['prediction']} ", end="")
                    print(f"(Confidence: {sent_result['confidence']:.1%})")

                if "topic" in results:
                    topic_result = results["topic"]
                    print(f"Topic: {topic_result['prediction']} ", end="")
                    print(f"(Confidence: {topic_result['confidence']:.1%})")
            else:
                print("Analysis failed. Please try again.")

        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    chat_interface()
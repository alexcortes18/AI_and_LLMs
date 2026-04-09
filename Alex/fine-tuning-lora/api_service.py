from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import uvicorn
from typing import List, Dict
import numpy as np
from pathlib import Path
import json

class ReviewRequest(BaseModel):
    text: str
    
class BatchReviewRequest(BaseModel):
    texts: List[str]
    
class SentimentResponse(BaseModel):
    text:str
    sentiment: str
    confidence: float
    processing_time: float
    
class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time: float
    
class ModelService:
    def __init__(self):
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         self.model_checkpoint = "distilbert-base-uncased"
         self.model_path = "./fine-tuning-lora/lora-sentiment"
         self.tokenizer = None #weishenme?
         self.model = None
         self.load_model()
         
    def load_model(self):
        """Load the tokenizer and the model"""
        try:
            print(f"Loading model on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
            
            #Load the base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_checkpoint,
                    num_labels = 2
                )
            
            # Load the LoRA weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval() # Set model to evaluation mode
            self.model.to(self.device)
            print("Model Loaded Successfully!") 
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    async def predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment for a single text."""
        try:
            import time
            start_time = time.time()
            
            # Tokenize the text to be able to predict
            inputs = self.tokenizer(
                text, 
                return_tensors = "pt",
                truncation = True,
                max_length = 128,
                padding = True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs) # inputs has structure like:
                # input_ids, attention_mask, labels
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()
                
            # Ma prediction to sentiment
            sentiment = "Positive" if prediction == 1 else "Negative"
            processing_time = time.time() - start_time
            
            return{
                "text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "processing_time": processing_time
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    async def predict_batch_sentiment(self, texts: List[str]) -> Dict:
        """Predict sentiment for a batch of texts"""
        try:
            import time
            start_time = time.time()
            
            results = []
            for text in texts:
                result = await self.predict_sentiment(text)
                results.append(result)
                
            total_time = time.time() - start_time
            
            return {
                "results": results,
                "total_time": total_time
            }
        except Exception as e:
            print("Error!!!")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app

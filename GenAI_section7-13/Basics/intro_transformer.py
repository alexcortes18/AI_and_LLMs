from transformers import pipeline, AutoTokenizer

def create_simple_llm():
    # Initialize the model and tokenizer
    model_name = "distilgpt2"
    
    # Create the generator pipeline
    generator = pipeline('text-generation',
                         model = model_name,
                         pad_token_id=50256)
    return generator

generator = create_simple_llm()

prompt = "Once upon a time"

generated_text = generator(prompt, max_length=1000, num_return_sequences=1)
print(generated_text[0]['generated_text'])
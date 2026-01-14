from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key= os.getenv("OPEN_AI_KEY"))

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages = [
        {'role': "system", "content": "You are a helpfu assistant."},
        {'role': "user", "content": "What is the purpose in life?"}
    ],
    max_tokens = 100
)

print(response.choices[0].message.content)
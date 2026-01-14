from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key= os.getenv("OPEN_AI_KEY"))

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages = [
        # {'role': "system", "content": "You are a helpful assistant."},
        # {'role': "user", "content": "What is the purpose in life?"},
        {'role': "system", "content": "You are eastern poet."},
        {'role': "user", "content": "Write me a short poem about the moon. \
                Write the poem in the style of haiku. \
                Make sure to include a title."}
    ],
    max_tokens = 100
)

print(response.choices[0].message.content)
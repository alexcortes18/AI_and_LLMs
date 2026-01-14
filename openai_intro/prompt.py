# FEW SHOT LEARNING PROMPT EXAMPLE.

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = (
    OpenAI()
)  # Don't need to specify API-Key since we are loading it here with load_dotevn()
# But if you run into issues you can load it with:
# client = OpenAI(api_key= os.getenv("OPEN_AI_KEY"))

# Few shot prompting
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a translator."},
        {
            "role": "user",
            "content": """ Translate these sentences: 
            'Hello' -> 'Hola',
            'Goodbye' -> 'Adios'.
            Now translate: 'Thank you'.""",
        },
    ],
)
print(completion.choices[0].message.content)

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
# print(completion.choices[0].message.content)

# Zero Shot Prompt  - or Direct Prompt Example

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer the question with direct answers."},
        {"role": "user", "content": "What is the capital of Honduras?"},
    ],
)
# print(completion.choices[0].message.content)

# Chain of Thought Prompt
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a math tutor."},
        {
            "role": "user",
            "content": """ Solve this math problem step by step:
            If John has 5 apples and gives to Mary 2, how many does he have left?
         """,
        },
    ],
)
# print(completion.choices[0].message.content)

# Instructional Prompt
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": """ Write a 300 word summary of the benefits of exercise using bullet points.
         """,
        },
    ],
)
# print(completion.choices[0].message.content)

# Role Playing Prompt
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a character in a fantasy novel."},
        {
            "role": "user",
            "content": """ Describe the setting of the story.
         """,
        },
    ],
)
# print(completion.choices[0].message.content)

# Temperture and Top-P sampling

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a travel blogger writer."},
        {
            "role": "user",
            "content": """ 
                       Write a 500 word blog post about your recent trip to Paris.
                       Make sure to include a step-by step itenerary of your trip.
                    """,
        },
    ],
    # Either use temperture or top_p but not both
    # temperature=0.5,  # controls the randomness of the output
    top_p=0.9,  # controls the diversity of the output
    # max_tokens=50,
    stream=True, # If stream is true, then we need to use delta in choices[0].delta.content
)
# print(completion.choices[0].message.content)
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
    # print("\n")

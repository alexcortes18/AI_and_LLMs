import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

text = """The purpose of life is a deeply personal and subjective question, and it can vary significantly from person to person. Here are some common perspectives on what gives life purpose:

1. Self-Discovery and Growth: Many people find purpose through self-exploration, personal growth, and understanding themselves better.

2. Relationships: Building and nurturing relationships with family, friends, and communities can provide a sense of belonging and fulfillment.

3. Contribution to Others: Helping others and making a"""

tokens = encoding.encode(text)
print(len(tokens))

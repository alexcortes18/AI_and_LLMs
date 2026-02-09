# ---------------------------------------------------------------------------------------------
# USING API ENDPOINT: /chat
# ---------------------------------------------------------------------------------------------
import ollama

# response = ollama.list()
# print(response)

res = ollama.chat(
    model="llama3.2", messages=[{"role": "user", "content": "Why is the sky blue?"}]
)

# print(res["message"]["content"])

res = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
)

# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)

# ---------------------------------------------------------------------------------------------
# USING API ENDPOINT: /generate
# ---------------------------------------------------------------------------------------------
res = ollama.generate(model="llama3.2", prompt="Why is the sky blue?", stream=True)

# print(res["response"])
# for chunk in res:
#     print(chunk["response"], end="", flush=True)

# ---------------------------------------------------------------------------------------------
# Create a new model with modelfile
# ---------------------------------------------------------------------------------------------

modelfile = """
# Set the temperture to 1 [higher is more creative]
Parameter temperature 1
SYSTEM
    You are James, a very smart assistant who knows everything about oceans. You are very succint and informative.
"""
ollama.create(model="knowitall", from_="llama3.2", system=modelfile)
# latest version of ollama does not support modelfile argument, so we use the "from" and "system" arguments instead

res = ollama.generate(
    model="knowitall",
    prompt="Tell me about the Mariana Trench.",
    stream=True,
    # With stream = True we have to iterate over the reponse
)
for chunk in res:
    print(chunk["response"], end="", flush=True)

ollama.delete(model="knowitall")

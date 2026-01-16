import requests
import json

url = "http://localhost:11434/api/generate/"

data = {"model": "llama3.2", "prompt": "Tell me a short story and make it funny."}
response = requests.post(url, json=data, stream=True)

# Check the response status
if response.status_code == 200:
    print("Generated Text:", end=" ", flush=True)
    # Iterate over the streaming response
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            result = json.loads(decoded_line)
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print(
        f"Error: Status code is {response.status_code}, with message: {response.text})"
    )

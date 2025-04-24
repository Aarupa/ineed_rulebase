# import requests
# import json  # Correctly import JSON module

# def query_ollama(prompt):
#     """
#     Sends a prompt to neural_chat running on Ollama and retrieves the response.
#     """
#     url = "http://localhost:11434/api/generate"
#     headers = {"Content-Type": "application/json"}
#     # Modify the prompt to request a response of up to 30 words
#     adjusted_prompt = f"{prompt}\nPlease provide a response of up to 30 words."
#     data = {
#         "model": "neural-chat:latest",
#         "prompt": adjusted_prompt,
#         "stream": True  # Explicitly request streaming
#     }

#     try:
#         response = requests.post(url, json=data, headers=headers, stream=True)
#         response.raise_for_status()

#         full_response = ""
#         for line in response.iter_lines():
#             if line:  # Skip empty lines
#                 json_line = line.decode('utf-8')  # Decode the line
#                 print(f"Received: {json_line}")  # Optional debug
#                 try:
#                     json_data = json.loads(json_line)
#                     full_response += json_data.get("response", "")
#                     if json_data.get("done", False):
#                         break
#                 except ValueError as ve:
#                     print(f"Error parsing JSON line: {ve}")

#         return full_response if full_response else "No response from neural_chat."

#     except requests.exceptions.RequestException as e:
#         print(f"Error details: {e}")
#         return f"Error communicating with neural_chat: {e}"


import requests
import json

# Optional cache (avoids repeating the same response)
from functools import lru_cache

@lru_cache(maxsize=128)
def query_ollama(prompt: str) -> str:
    """
    Optimized function to query the neural-chat model running via Ollama.
    Includes streaming, timeout, caching, and preloading tips.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    # 1. Keep prompt short and to the point (important for speed)
    adjusted_prompt = f"{prompt.strip()} (respond in ~30 words only)"

    data = {
        "model": "neural-chat:latest",  # Change to 'mistral' if you want alternatives
        "prompt": adjusted_prompt,
        "stream": True
    }

    try:
        # 2. Timeout ensures app doesn't hang forever
        response = requests.post(url, json=data, headers=headers, stream=True, timeout=15)
        response.raise_for_status()

        # 3. Streaming results: Token-by-token feel
        full_response = ""
        print("[Response]: ", end="", flush=True)  # Optional: for CLI debugging

        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    token = json_data.get("response", "")
                    full_response += token
                    print(token, end="", flush=True)  # Feels like streaming
                    if json_data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue  # Ignore bad chunks

        print()  # End line in CLI
        return full_response.strip() if full_response else "No response from Neural Chat."

    except requests.exceptions.Timeout:
        return "⏱️ Timeout: Neural Chat took too long to respond."
    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {e}"

import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Corrected API endpoint

def query_ollama(prompt):
    """
    Send a prompt to the Ollama model and return the response.
    """
    payload = {"model": "neural-chat", "prompt": prompt}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "No response from model.")
    except requests.RequestException as e:
        print(f"Error querying Ollama model: {e}")
        return "Error communicating with the model."

# Example usage
if __name__ == "__main__":
    user_input = "Hello, how are you?"
    response = query_ollama(user_input)
    print(f"Model response: {response}")

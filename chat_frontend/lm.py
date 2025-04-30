import requests
import json  # Import the built-in JSON module

def query_neural_chat(prompt):
    """
    Sends a prompt to neural_chat running on Ollama and retrieves the response.
    """
    url = "http://localhost:11434/api/generate"  # Replace with the actual Ollama endpoint
    headers = {"Content-Type": "application/json"}
    data = {"model": "neural-chat:latest", "prompt": prompt}  # Specify the model name here

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)  # Enable streaming
        response.raise_for_status()

        # Process the streamed response
        full_response = ""
        for line in response.iter_lines():
            if line:  # Skip empty lines
                json_line = line.decode('utf-8')  # Decode the line
                print(f"Received: {json_line}")  # Debugging: Print each JSON object
                try:
                    json_data = json.loads(json_line)  # Use the correct json.loads method
                    full_response += json_data.get("response", "")
                    if json_data.get("done", False):  # Check if the response is complete
                        break
                except ValueError as ve:
                    print(f"Error parsing JSON line: {ve}")

        return full_response if full_response else "No response from neural_chat."
    except requests.exceptions.RequestException as e:
        print(f"Error details: {e}")
        return f"Error communicating with neural_chat: {e}"
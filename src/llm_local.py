import requests

def ollama_generate(prompt: str, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> str:
    """
    Works only when Ollama is running on the same machine.
    Streamlit Cloud won't have this, so handle exceptions in app.
    """
    url = f"{base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()  nb
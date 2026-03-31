import requests


def ollama_generate(prompt: str, model: str = "llama3") -> str:
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    except Exception as e:
        return f"LLM error: {str(e)}".strip()
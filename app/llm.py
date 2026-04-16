import requests

from app.config import OLLAMA_URL, OLLAMA_MODEL, REQUEST_TIMEOUT


def ollama_chat(system: str, user: str, temperature: float = 0.2, timeout: int = None) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout or REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error: {r.status_code} {r.text}")

    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()

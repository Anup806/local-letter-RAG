import json
from typing import Iterable, List, Dict, Optional

import requests

from app.config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT,
    OLLAMA_NUM_PREDICT,
    OLLAMA_REPEAT_PENALTY,
    OLLAMA_REPEAT_LAST_N,
    OLLAMA_TOP_P,
    OLLAMA_TOP_K,
)


def _build_messages(
    system: str,
    user: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the Ollama message list with optional history."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def ollama_chat(
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout: int = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Run a non-streaming chat request against Ollama."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_messages(system=system, user=user, history=history),
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": OLLAMA_NUM_PREDICT,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
            "repeat_last_n": OLLAMA_REPEAT_LAST_N,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout or REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error: {r.status_code} {r.text}")

    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def ollama_chat_stream(
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout: int = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Iterable[str]:
    """Stream a chat response token-by-token from Ollama."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_messages(system=system, user=user, history=history),
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": OLLAMA_NUM_PREDICT,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
            "repeat_last_n": OLLAMA_REPEAT_LAST_N,
            "top_p": OLLAMA_TOP_P,
            "top_k": OLLAMA_TOP_K,
        },
    }
    r = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout or REQUEST_TIMEOUT,
        stream=True,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error: {r.status_code} {r.text}")

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        message = data.get("message", {}) or {}
        token = message.get("content", "")
        if token:
            yield token
        if data.get("done") is True:
            break

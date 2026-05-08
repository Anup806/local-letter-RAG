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


# Centralizes Ollama message assembly to keep system/user/history formatting consistent.
def _build_messages(
    system: str,
    user: str,
    history: Optional[List[Dict[str, str]]] = None,  # Prior chat turns as {"role", "content"} dicts.
) -> List[Dict[str, str]]:
    """Build the Ollama message list with optional history."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
    ]
    if history:
        messages.extend(history)  # Preserve prior turns to give the model context.
    messages.append({"role": "user", "content": user})  # Add the current prompt as the latest turn.
    return messages


# Provides a simple non-streaming Ollama call for one-shot responses.
def ollama_chat(
    system: str,
    user: str,
    temperature: float = 0.6,
    timeout: int = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Run a non-streaming chat request against Ollama."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _build_messages(system=system, user=user, history=history),
        "stream": False,    # tells ollama not to stream, API waits until the full response is ready and return in one JSON payload
        "options": {
            "temperature": temperature,  # Randomness of output; higher is more creative.
            "num_predict": OLLAMA_NUM_PREDICT,  # Max tokens to generate.
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,  # Discourage repeated phrases.
            "repeat_last_n": OLLAMA_REPEAT_LAST_N,  # Lookback window for repetition penalty.
            "top_p": OLLAMA_TOP_P,  # Nucleus sampling threshold.
            "top_k": OLLAMA_TOP_K,  # Limit to top-K tokens before sampling. top_p & top_k = lower(safer,deterministic), higher(more variety, risk of odd o/p)
        },
    }
    # send HTTP POST request to the Ollama server
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout or REQUEST_TIMEOUT) 
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error: {r.status_code} {r.text}")

    data = r.json()
    # data.get("message", {}) return message if it exist onther wise {}, 
    # (or {}) make sure you still have dict{} even message is None
    # strip() -> string method, remote whitespaces ie tabs, newlines, spaces
    return (data.get("message", {}) or {}).get("content", "").strip()

# Enables streaming responses for incremental UI updates or long outputs.
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
        # loads convert JSON string (line) into python DICTIONARY
        data = json.loads(line)
        message = data.get("message", {}) or {}
        token = message.get("content", "")
        if token:
            # yield is like return but instead of stopping the function, it pauses and sends back to the caller 
            yield token
        if data.get("done") is True:
            break

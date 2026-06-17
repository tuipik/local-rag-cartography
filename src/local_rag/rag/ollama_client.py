"""Small Ollama chat client for local answer generation."""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_NUM_PREDICT = 1024


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def request_json(url: str, payload: dict[str, object], *, timeout: int = 180) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeError(f"Ollama request timed out after {timeout} seconds") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Ollama request failed: {error}") from error


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*Thinking\.\.\..*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    return text.strip()


def chat(
    *,
    messages: list[ChatMessage],
    model: str,
    ollama_url: str,
    temperature: float = 0.1,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> str:
    base_url = ollama_url.rstrip("/")
    response = request_json(
        f"{base_url}/api/chat",
        {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
    )
    message = response.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama did not return a chat message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama did not return message content")
    return strip_thinking(content)

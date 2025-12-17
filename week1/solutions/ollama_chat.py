
#!/usr/bin/env python3
"""
Simple Ollama chat client using the REST API.

- Sends a user question to the selected model.
- Prints the assistant's reply (optionally streamed).
- Keeps minimal chat history to preserve context.

Run:
    python3 ollama_chat.py --model llama3.1 "What's the capital of Australia?"
    python3 ollama_chat.py --model llama3.1 --stream "Explain Kubernetes briefly."
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Optional

import requests

OLLAMA_HOST = "http://localhost:11434"
CHAT_ENDPOINT = f"{OLLAMA_HOST}/api/chat"


def chat_once(
    model: str,
    user_message: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    stream: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Send a single chat turn to Ollama and return the assistant response.

    :param model: Model name (e.g., 'llama3.1', 'mistral', 'phi3')
    :param user_message: The question or prompt from the user
    :param system_prompt: Optional system message to steer behavior
    :param temperature: Sampling temperature (0.0–1.0+)
    :param stream: If True, print chunks as they arrive and return full text
    :param history: Optional list of prior messages for context
    :return: Full assistant response text
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        # Expect prior messages as [{"role": "...", "content": "..."}]
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }

    if not stream:
        # Non-streaming: single JSON response
        resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns {"message": {"content": "..."} ...}
        content = data.get("message", {}).get("content", "")
        return content

    # Streaming: NDJSON lines with incremental tokens
    full_text = []
    with requests.post(CHAT_ENDPOINT, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                # Defensive: skip malformed lines
                continue

            # Each chunk may have partial content in message.content
            msg = chunk.get("message", {})
            piece = msg.get("content", "")
            if piece:
                full_text.append(piece)
                # Print live to stdout without newline buffering
                print(piece, end="", flush=True)

            # Stop condition when response finishes
            if chunk.get("done"):
                print()  # newline after streaming completes
                break

    return "".join(full_text)


def main():
    parser = argparse.ArgumentParser(description="Ollama Chat via REST API")
    parser.add_argument(
        "question", nargs="?", default=None, help="Question to ask the model."
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Model name (e.g., llama3.1, mistral, phi3, qwen2).",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt to guide the assistant.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0–1.0+). Lower = more deterministic.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response token-by-token.",
    )
    args = parser.parse_args()

    # Ask for input if not provided as an argument
    question = args.question
    if question is None:
        try:
            question = input("Enter your question: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    # Minimal example: single turn without prior history
    answer = chat_once(
        model=args.model,
        user_message=question,
        system_prompt=args.system,
        temperature=args.temperature,
        stream=args.stream,
        history=None,
    )

    if not args.stream:
        print(answer)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import OpenAIError, APIError, RateLimitError, AuthenticationError, BadRequestError, APIConnectionError

"""
Multi-turn chat with robust error handling:
- Validates environment and model
- Retries with exponential backoff on transient failures
- Handles server offline, bad model, auth, and JSON parsing
- Optional streaming with safe consumption
"""

# ---------- Configure logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------- Constants ----------
BASE_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLAMA_API_KEY", "ollama")   # Any string is fine for Ollama
MODEL = os.getenv("LLAMA_MODEL", "gemma3:latest")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
INITIAL_BACKOFF = float(os.getenv("INITIAL_BACKOFF", "0.5"))
BACKOFF_MULTIPLIER = float(os.getenv("BACKOFF_MULTIPLIER", "2.0"))

# ---------- Utilities ----------
def retryable_chat_completion(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> Optional[Any]:
    """
    Retry wrapper for chat completions, with exponential backoff for transient failures.
    Returns:
        - If stream=False: the response object
        - If stream=True: the stream context manager (caller iterates safely)
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            if stream:
                # Return the stream context; caller must use 'with' and iterate safely
                return client.chat.completions.stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
        except (APIConnectionError, APIError) as e:
            # Transient or server-side errors → retry
            logging.warning(f"[Attempt {attempt}/{RETRY_ATTEMPTS}] API/server error: {e}")
        except RateLimitError as e:
            logging.warning(f"[Attempt {attempt}/{RETRY_ATTEMPTS}] Rate limited: {e}")
        except TimeoutError as e:
            logging.warning(f"[Attempt {attempt}/{RETRY_ATTEMPTS}] Timeout: {e}")
        except (AuthenticationError, BadRequestError, OpenAIError) as e:
            # Non-retryable (likely misconfiguration, bad model name, etc.)
            logging.error(f"Non-retryable error: {e}")
            break

        if attempt < RETRY_ATTEMPTS:
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER

    return None

def ensure_env():
    if not BASE_URL.startswith("http"):
        raise ValueError(f"Invalid LLAMA_API_URL: {BASE_URL}")
    if not MODEL:
        raise ValueError("LLAMA_MODEL is empty")
    logging.info(f"Using base_url={BASE_URL}, model={MODEL}")

def run_multi_turn(stream: bool = False):
    ensure_env()

    # Initialize client pointing to Ollama’s OpenAI-compatible API
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # Conversation state
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a concise, helpful assistant for a Senior kafka Engineer."},
        {"role": "user", "content": "Summarize blue/green vs canary deployments."}
    ]

    # --- Turn 1 ---
    resp1 = retryable_chat_completion(
        client, messages, MODEL, TEMPERATURE, MAX_TOKENS, stream=False
    )
    if resp1 is None:
        # Fallback if server unreachable or persistent failure
        logging.error("Failed to get response for Turn 1; falling back to local message.")
        assistant_1 = "Fallback: cp-ansible deploy confluent platform to VMs, CFK deploy Conflyent platform on Kubenetes."
    else:
        try:
            assistant_1 = resp1.choices[0].message.content
        except Exception as e:
            logging.error(f"Response parsing error (Turn 1): {e}")
            assistant_1 = "Parsing error fallback: cons and pros of each deployment."
    print("\nAssistant (Turn 1):\n", assistant_1)
    messages.append({"role": "assistant", "content": assistant_1})

    # --- Turn 2 ---
    messages.append({"role": "user", "content": "Which approach fits a startup company to start from scratch and why?"})
    resp2 = retryable_chat_completion(
        client, messages, MODEL, TEMPERATURE, MAX_TOKENS, stream=False
    )
    if resp2 is None:
        logging.error("Failed to get response for Turn 2; using fallback.")
        assistant_2 = "Fallback: kafka is statefulset, is kubenet teh better choice."
    else:
        try:
            assistant_2 = resp2.choices[0].message.content
        except Exception as e:
            logging.error(f"Response parsing error (Turn 2): {e}")
            assistant_2 = "Parsing error fallback: Prefer cp-ansible for more stable service."
    print("\nAssistant (Turn 2):\n", assistant_2)
    messages.append({"role": "assistant", "content": assistant_2})

    # --- Turn 3 with streaming (optional) ---
    messages.append({"role": "user", "content": "Give a 6-step checklist deploy confluent platform using cp-ansible."})
    if stream:
        try:
            with retryable_chat_completion(
                client, messages, MODEL, TEMPERATURE, MAX_TOKENS, stream=True
            ) as stream_ctx:
                if stream_ctx is None:
                    raise RuntimeError("Streaming not available.")
                print("\nAssistant (Turn 3 - Streaming):\n")
                for event in stream_ctx:
                    # Guard for token events; ignore other events such as 'error' or 'complete'
                    if getattr(event, "type", None) == "token":
                        print(event.token, end="", flush=True)
                print("\n\n[Done]")
        except Exception as e:
            logging.error(f"Streaming error (Turn 3): {e}")
            print("\nFallback checklist:\n1) Version new deployment\n2) Label workloads\n3) Configure VirtualService/DR\n4) Route small traffic\n5) Observe SLOs\n6) Promote or rollback")
    else:
        resp3 = retryable_chat_completion(
            client, messages, MODEL, TEMPERATURE, MAX_TOKENS, stream=False
        )
        if resp3 is None:
            logging.error("Failed to get response for Turn 3; using fallback.")
            assistant_3 = (
                "Fallback checklist:\n"
                "1) Deploy new version alongside stable\n"
                "2) Configure Istio VirtualService/DestinationRule\n"
                "3) Start with 1–5% traffic to canary\n"
                "4) Monitor latency, errors, saturation\n"
                "5) Gradually increase traffic\n"
                "6) Promote canary or rollback"
            )
        else:
            try:
                assistant_3 = resp3.choices[0].message.content
            except Exception as e:
                logging.error(f"Response parsing error (Turn 3): {e}")
                assistant_3 = "Parsing error fallback: progressive traffic, metrics, promotion/rollback."
        print("\nAssistant (Turn 3):\n", assistant_3)

if __name__ == "__main__":
    # CLI: python script.py [--stream]
    use_stream = "--stream" in sys.argv
    run_multi_turn(use_stream)

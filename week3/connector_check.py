# -*- coding: utf-8 -*-
"""
Kafka Connect – Status Assistant (Unified, optimized)
----------------------------------------------------
Single Gradio chat app that can use either:
 - Local Ollama (AI_BACKEND=ollama)
 - GPT via OpenAI SDK (AI_BACKEND=gptai or openai)

Improvements vs previous version:
- Graceful handling when CONNECT_URL is unreachable: returns a friendly
  prompt asking the user to paste another URL; no stack trace.
- Bare `host:port` now defaults to **HTTP** (safer for typical Connect REST).
- Clearer connection/timeout error messages with actionable hints.
- Retry configuration covers connect/read errors, not just HTTP status.
- Optionally supports OpenAI-compatible endpoints via OPENAI_BASE_URL.
- Small sanitization and parsing fixes; early reachability probe.

Env vars:
 AI_BACKEND=ollama|gptai|openai
 OLLAMA_MODEL=llama3.2:latest            # for Ollama
 GPT_MODEL=gpt-5-mini                    # for OpenAI/GPT
 OPENAI_API_KEY=...                      # required when AI_BACKEND=gptai (unless your OPENAI_BASE_URL allows blank)
 OPENAI_BASE_URL=...                     # optional (e.g., http://localhost:11434/v1 for OpenAI-compatible)
 CONNECT_URL=...                         # optional; can be pasted in chat instead
 CONNECT_USERNAME / CONNECT_PASSWORD / CONNECT_BEARER_TOKEN
 VERIFY_TLS=true|false
 CA_CERT_PATH=/path/to/ca-bundle.pem
 CONNECT_TIMEOUT=10
 CONNECT_RETRY_TOTAL=3
 CONNECT_RETRY_BACKOFF=0.5
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

import gradio as gr
from dotenv import load_dotenv

# ------------------ Backend selection ------------------
AI_BACKEND = os.getenv("AI_BACKEND", "ollama").strip().lower()
USE_OPENAI = AI_BACKEND in ("gptai", "openai")

if USE_OPENAI:
    try:
        from openai import OpenAI  # pip install openai
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "AI_BACKEND is 'gptai' but 'openai' package is not installed. pip install openai"
        ) from e
else:
    try:
        import ollama  # pip install ollama
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "AI_BACKEND is 'ollama' but 'ollama' package is not installed. pip install ollama"
        ) from e

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError as ReqConnectionError, Timeout as ReqTimeout
from urllib3.util.retry import Retry
from urllib.parse import quote, urlparse

# ------------------ System prompt ------------------
SYSTEM_MESSAGE = (
    "You are a helpful assistant for Kafka Connect operations. "
    "If the user provides a Connect URL or host:port in the message, extract and use it; otherwise ask for it. "
    "Give short, accurate answers (1–2 sentences). If you don't know, say so."
)

# ------------------ Models / config ------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional for OpenAI-compatible servers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ------------------ Logging & env ------------------
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=level)


def load_env(override: bool = True) -> None:
    load_dotenv(override=override)


# ------------------ CONNECT_URL normalization & extraction ------------------

def _clean_token(tok: str) -> str:
    """Trim spaces, surrounding quotes, and trailing commas; return cleaned token."""
    if not tok:
        return ""
    s = tok.strip().strip(",")
    # remove one set of leading/trailing quotes if present
    if s and s[0] in "'\"":
        s = s[1:]
    if s and s[-1] in "'\"":
        s = s[:-1]
    return s.strip()


def _strip_to_base(url: str) -> str:
    """Reduce input to scheme://host:port (strip any path/query/fragment).
    Accept lists (comma/space separated): picks the first token.
    Accepts bare host:port; defaults to **http**.
    """
    if not url:
        return ""
    first = re.split(r"[\s,]+", url.strip())[0]
    first = _clean_token(first)
    if not first:
        return ""

    # full URL case
    try:
        u = urlparse(first)
        if u.scheme and u.netloc:
            return f"{u.scheme}://{u.netloc}".rstrip("/")
    except Exception:
        pass

    # bare host:port -> default HTTP
    m = re.match(r"^([a-zA-Z0-9_.-]+):(\d{2,5})$", first)
    if m:
        return f"http://{m.group(1)}:{m.group(2)}"
    return ""


def _extract_connect_url(text: str) -> str:
    if not text:
        return ""
    # CONNECT_URL=...
    m = re.search(r"CONNECT_URL\s*=\s*(\S+)", text)
    if m:
        base = _strip_to_base(m.group(1))
        if base:
            return base
    # explicit http(s)
    m2 = re.search(r"(https?://[^\s,]+)", text)
    if m2:
        base = _strip_to_base(m2.group(1))
        if base:
            return base
    # bare host:port
    m3 = re.search(r"([a-zA-Z0-9_.-]+:\d{2,5})", text)
    if m3:
        base = _strip_to_base(m3.group(1))
        if base:
            return base
    return ""


def _maybe_set_connect_url_from_text(text: str) -> str:
    base = _extract_connect_url(text)
    if base:
        os.environ["CONNECT_URL"] = base
    return base


# ------------------ Kafka Connect REST client ------------------
DEFAULT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "10"))


def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=int(os.getenv("CONNECT_RETRY_TOTAL", "3")),
        connect=int(os.getenv("CONNECT_RETRY_TOTAL", "3")),
        read=int(os.getenv("CONNECT_RETRY_TOTAL", "3")),
        backoff_factor=float(os.getenv("CONNECT_RETRY_BACKOFF", "0.5")),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "PUT", "PATCH", "DELETE"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    user = os.getenv("CONNECT_USERNAME")
    pwd = os.getenv("CONNECT_PASSWORD")
    token = os.getenv("CONNECT_BEARER_TOKEN")
    if user and pwd:
        s.auth = (user, pwd)
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})

    verify_tls = os.getenv("VERIFY_TLS", "true").lower()
    if verify_tls in ("false", "0", "no"):
        s.verify = False
    ca_path = os.getenv("CA_CERT_PATH")
    if ca_path:
        s.verify = ca_path
        
    # default headers
    s.headers.setdefault("Accept", "application/json")
    s.headers.setdefault("User-Agent", "connect-status-chat/unified-1.1")
    return s


def _base_url() -> str:
    url = os.getenv("CONNECT_URL")
    if url:
        url = _clean_token(url)
    base = _strip_to_base(url or "")
    if not base:
        raise RuntimeError("CONNECT_URL is required (e.g., http://connect.example.com:8083)")
    return base


def _request_json(method: str, path: str, **kwargs):
    """Core request helper returning parsed JSON or a dict with raw text."""
    session = kwargs.pop("session", None) or _build_session()
    base = _base_url()
    p = path if path.startswith("/") else f"/{path}"
    p = _clean_token(p)
    url = f"{base}{p}"
    timeout = kwargs.pop("timeout", DEFAULT_TIMEOUT)

    try:
        resp = session.request(method, url, timeout=timeout, **kwargs)
    except ReqTimeout:
        raise RuntimeError(
            "Timeout while contacting Kafka Connect REST. "
            "Please verify the URL/port and network path, or provide a different CONNECT_URL."
        )
    except ReqConnectionError as e:
        scheme = urlparse(url).scheme
        hint = (
            " If your Connect REST is non‑TLS, use http://host:port. "
            "If it is TLS, keep https:// and set VERIFY_TLS=true and (optionally) CA_CERT_PATH."
        )
        raise RuntimeError(
            f"Unable to reach {url} ({e}).{hint} "
            "You can paste another URL (e.g., CONNECT_URL=http://host:8083) to continue."
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        msg = f"HTTP {resp.status_code} for {url}: {resp.text}"
        raise requests.HTTPError(msg) from e

    ctype = resp.headers.get("Content-Type", "")
    if "application/json" not in ctype:
        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}
    return resp.json()


def ping_connect() -> None | str:
    """Reachability probe; returns None if OK, else error string."""
    try:
        _request_json("GET", "/connectors")
        return None
    except Exception as e:
        return str(e)


# ------------------ Tools ------------------

def list_connectors() -> Dict[str, Any]:
    data = _request_json("GET", "/connectors")
    if isinstance(data, list):
        connectors = data
    elif isinstance(data, dict) and "connectors" in data:
        connectors = data.get("connectors", [])
    else:
        connectors = []
    return {"connectors": connectors}


def get_connector_status(name: str) -> Dict[str, Any]:
    if not name:
        return {"error": "Connector name is required."}
    name_enc = quote(name, safe="")
    try:
        return _request_json("GET", f"/connectors/{name_enc}/status")
    except Exception as e:
        return {"error": str(e), "name": name}


def get_connector_config(name: str) -> Dict[str, Any]:
    if not name:
        return {"error": "Connector name is required."}
    name_enc = quote(name, safe="")
    try:
        return _request_json("GET", f"/connectors/{name_enc}/config")
    except Exception as e:
        return {"error": str(e), "name": name}


def get_connector_offsets(name: str) -> Dict[str, Any]:
    """Offsets endpoints are available on Kafka 3.6+; older clusters may return 404."""
    if not name:
        return {"error": "Connector name is required."}
    name_enc = quote(name, safe="")
    try:
        return _request_json("GET", f"/connectors/{name_enc}/offsets")
    except Exception as e:
        return {"error": str(e), "name": name}


def stop_connector(name: str) -> Dict[str, Any]:
    if not name:
        return {"error": "Connector name is required."}
    name_enc = quote(name, safe="")
    try:
        return _request_json("PUT", f"/connectors/{name_enc}/stop")
    except Exception as e:
        return {"error": str(e), "name": name}


def start_connector(name: str) -> Dict[str, Any]:
    if not name:
        return {"error": "Connector name is required."}
    name_enc = quote(name, safe="")
    try:
        return _request_json("PUT", f"/connectors/{name_enc}/resume")
    except Exception as e:
        return {"error": str(e), "name": name}


LIST_CONNECTORS_FUNCTION: Dict[str, Any] = {
    "name": "list_connectors",
    "description": "List all Kafka Connect connector names.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

GET_STATUS_FUNCTION: Dict[str, Any] = {
    "name": "get_connector_status",
    "description": "Get the status for a specific Kafka Connect connector.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The connector name to query."}},
        "required": ["name"],
        "additionalProperties": False,
    },
}

GET_CONFIG_FUNCTION: Dict[str, Any] = {
    "name": "get_connector_config",
    "description": "Get the configuration for a specific Kafka Connect connector.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The connector name to query."}},
        "required": ["name"],
        "additionalProperties": False,
    },
}

GET_OFFSETS_FUNCTION: Dict[str, Any] = {
    "name": "get_connector_offsets",
    "description": "Get offsets for a specific Kafka Connect connector (Kafka 3.6+).",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The connector name to query."}},
        "required": ["name"],
        "additionalProperties": False,
    },
}

STOP_CONNECTOR_FUNCTION: Dict[str, Any] = {
    "name": "stop_connector",
    "description": "Stop a specific Kafka Connect connector.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The connector name to stop."}},
        "required": ["name"],
        "additionalProperties": False,
    },
}

START_CONNECTOR_FUNCTION: Dict[str, Any] = {
    "name": "start_connector",
    "description": "Start a specific Kafka Connect connector.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The connector name to resume."}},
        "required": ["name"],
        "additionalProperties": False,
    },
}

TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": LIST_CONNECTORS_FUNCTION},
    {"type": "function", "function": GET_STATUS_FUNCTION},
    {"type": "function", "function": GET_CONFIG_FUNCTION},
    {"type": "function", "function": GET_OFFSETS_FUNCTION},
    {"type": "function", "function": STOP_CONNECTOR_FUNCTION},
    {"type": "function", "function": START_CONNECTOR_FUNCTION},
]


# ------------------ Helpers ------------------

def _ensure_text(content: Any) -> str:
    """Normalize Gradio ChatInterface content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        return " ".join(parts).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        if "value" in content:
            return str(content.get("value", ""))
    return str(content)


# ------------------ Tool-call dispatchers ------------------

def handle_tool_calls_openai(message: Any) -> List[Dict[str, Any]]:
    """Dispatcher for OpenAI SDK messages (tool_calls as list on message)."""
    logger = logging.getLogger(__name__)
    results: List[Dict[str, Any]] = []
    tool_calls = getattr(message, "tool_calls", None) or []

    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None)
        raw_args = getattr(fn, "arguments", {})  # may be dict or str

        if isinstance(raw_args, str):
            try:
                args: Dict[str, Any] = json.loads(raw_args)
            except Exception:
                logger.exception("Invalid JSON arguments for tool %r: %r", name, raw_args)
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        if name == "list_connectors":
            try:
                payload = list_connectors()
            except Exception as e:
                payload = {"error": str(e)}
        elif name == "get_connector_status":
            payload = get_connector_status(args.get("name"))
        elif name == "get_connector_config":
            payload = get_connector_config(args.get("name"))
        elif name == "get_connector_offsets":
            payload = get_connector_offsets(args.get("name"))
        elif name == "stop_connector":
            payload = stop_connector(args.get("name"))
        elif name == "start_connector":
            payload = start_connector(args.get("name"))
        else:
            logger.warning("Unknown tool requested: %s", name)
            payload = {"error": f"Unknown tool: {name}"}

        results.append(
            {
                "role": "tool",
                "content": json.dumps(payload, ensure_ascii=False),
                "tool_call_id": getattr(tc, "id", None),
            }
        )

    return results


def handle_tool_calls_ollama(message: Any) -> List[Dict[str, Any]]:
    """Dispatcher for Ollama chat responses (same shape as OpenAI)."""
    # Ollama Python client mirrors tool_call shape; reuse same logic
    return handle_tool_calls_openai(message)


# ------------------ Chat flow ------------------

def build_messages(user_message: str, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for h in history or []:
        role = h.get("role", "user")
        content = _ensure_text(h.get("content", ""))
        normalized.append({"role": role, "content": content})
    return [{"role": "system", "content": SYSTEM_MESSAGE}] + normalized + [{"role": "user", "content": _ensure_text(user_message)}]


def _openai_client() -> Any:
    # Construct client honoring OPENAI_BASE_URL if provided
    kwargs: Dict[str, Any] = {}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    return OpenAI(**kwargs)


def chat(user_message: str, history: List[Dict[str, Any]]) -> str:
    setup_logging()
    load_env(True)

    # Try to auto-extract/set CONNECT_URL from this turn or recent history
    _maybe_set_connect_url_from_text(user_message)
    if not os.getenv("CONNECT_URL"):
        for h in reversed(history or []):
            content = _ensure_text(h.get("content", ""))
            if _maybe_set_connect_url_from_text(content):
                break
    if not os.getenv("CONNECT_URL"):
        return (
            "CONNECT_URL is not set. Please paste it (e.g., "
            "CONNECT_URL=http://connect.example.com:8083)."
        )

    # Early reachability probe so we can guide the user immediately
    probe_err = ping_connect()
    if probe_err:
        return (
            f"Could not reach the Kafka Connect REST endpoint: {probe_err}\n\n"
            "👉 Please paste a different CONNECT_URL (e.g., `CONNECT_URL=http://host:8083`) or correct the current one.\n"
            "- If your endpoint is non‑TLS, use `http://`.\n"
            "- If it uses TLS, use `https://` and set `VERIFY_TLS=true` and optionally `CA_CERT_PATH`.\n"
        )

    messages = build_messages(user_message, history)

    if USE_OPENAI:
        client = _openai_client()
        response = client.chat.completions.create(model=GPT_MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            tool_results = handle_tool_calls_openai(message)
            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})
            messages.extend(tool_results)
            response = client.chat.completions.create(model=GPT_MODEL, messages=messages)
        return response.choices[0].message.content
    else:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS)
        message = response.message
        if getattr(message, "tool_calls", None):
            tool_results = handle_tool_calls_ollama(message)
            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})
            messages.extend(tool_results)
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response.message.content


# ------------------ Gradio app ------------------
if __name__ == "__main__":
    title = f"Kafka Connect – Status Assistant (backend: {'OpenAI/GPT' if USE_OPENAI else 'Ollama'})"
    desc = (
        "Paste CONNECT_URL in the chat or set it in env. "
        "Examples: 'CONNECT_URL=http://connect.example.com:8083', 'List connectors', 'Offsets of orders-source'.\n"
        "If the URL is unreachable, the app will prompt you to provide another one.\n"
        "Backend selection: AI_BACKEND=ollama|gptai (also accepts 'openai')."
    )
    iface = gr.ChatInterface(
        fn=chat,
        title=title,
        description=desc,
        chatbot=gr.Chatbot(height=400),
    )
    iface.launch(share=False)

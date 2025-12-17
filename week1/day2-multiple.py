
#!/usr/bin/env python3
import os
import json
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode
import requests
from openai import OpenAI

# -----------------------------
# Config (env overrides)
# -----------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

PROM_QUERY_URL = os.getenv("PROM_QUERY_URL", "https://localhost:9090/query")
# Optionally: /api/v1/query or /api/v1/query_range; you provided /query, so we'll use instant query semantics.
PROM_AUTH_HEADER = os.getenv("PROM_AUTH_HEADER")  # e.g. "Bearer <token>"
PROM_BASIC_USER = os.getenv("PROM_BASIC_USER")
PROM_BASIC_PASS = os.getenv("PROM_BASIC_PASS")

CONNECT_STATUS_URL = os.getenv(
    "CONNECT_STATUS_URL",
    "https://dyn-rc-biabkafka02.biab.au.ing.net:9083/connectors/application-debezium-connector/status",
)

VERIFY_TLS = os.getenv("VERIFY_TLS", "true").lower() == "true"
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# -----------------------------
# OpenAI client -> Ollama /v1
# -----------------------------
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# -----------------------------
# Tools (JSON Schema)
# -----------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_prometheus",
            "description": (
                "Run a PromQL instant query against Prometheus and return the latest value(s). "
                "Use when you need metrics such as error rate, lag, throughput."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PromQL query string"},
                    "time": {
                        "type": "string",
                        "description": "Optional RFC3339 or unix ts for evaluation time"
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_connector_status",
            "description": (
                "Fetch Kafka Connect connector status (state and per-task states) "
                "from the Connect REST API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Override status URL. Defaults to configured CONNECT_STATUS_URL.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
]

# -----------------------------
# Real tool handlers (HTTPS)
# -----------------------------
def _prom_headers() -> Dict[str, str]:
    h = {}
    if PROM_AUTH_HEADER:
        h["Authorization"] = PROM_AUTH_HEADER
    return h

def _prom_auth():
    if PROM_BASIC_USER and PROM_BASIC_PASS:
        return (PROM_BASIC_USER, PROM_BASIC_PASS)
    return None

def handle_query_prometheus(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your Prometheus endpoint (instant query).
    You provided: https://localhost:9090/query

    If your Prometheus expects /api/v1/query, set PROM_QUERY_URL accordingly.
    """
    query = args["query"]
    t = args.get("time")  # optional
    params = {"query": query}
    if t:
        params["time"] = t

    url = PROM_QUERY_URL
    # If your endpoint is actually /api/v1/query, you can just set PROM_QUERY_URL to that.
    # Otherwise we compose query string here:
    try:
        r = requests.get(
            url,
            params=params,
            headers=_prom_headers(),
            auth=_prom_auth(),
            timeout=HTTP_TIMEOUT,
            verify=VERIFY_TLS,
        )
        r.raise_for_status()
        data = r.json()
        # Normalize a minimal, predictable shape for the model:
        return {
            "endpoint": url,
            "params": params,
            "status": data.get("status"),
            "resultType": data.get("data", {}).get("resultType"),
            "result": data.get("data", {}).get("result"),
        }
    except Exception as e:
        return {
            "endpoint": url,
            "params": params,
            "error": str(e),
        }

def handle_get_connector_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your Kafka Connect status endpoint.
    Default: the URL you provided (single connector).
    """
    url = args.get("url") or CONNECT_STATUS_URL
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, verify=VERIFY_TLS)
        r.raise_for_status()
        data = r.json()
        # Return compact, model-friendly fields:
        return {
            "endpoint": url,
            "name": data.get("name"),
            "connector": data.get("connector", {}),
            "tasks": data.get("tasks", []),
            # Some Connects include "type", "trace", 'status' fields; pass through:
            "raw": data,
        }
    except Exception as e:
        return {
            "endpoint": url,
            "error": str(e),
        }

tool_handlers = {
    "query_prometheus": handle_query_prometheus,
    "get_connector_status": handle_get_connector_status,
}

# -----------------------------
# System context tailored to your stack
# -----------------------------
SYSTEM_CTX = (
    "You are a concise Kafka/DevOps assistant. Environment: Confluent Platform with Kafka Connect "
    "using Debezium source connectors. Metrics are exposed via JMX, collected by jmx_prometheus_javaagent, "
    "scraped by Prometheus, and visualized with Grafana. Prefer owner-facing recommendations "
    "(connector-level) with PromQL, thresholds, severity, and runbook steps. "
    "Call tools when you need live metrics or connector status."
)

# -----------------------------
# Conversation seed
# -----------------------------
messages = [
    {"role": "system", "content": SYSTEM_CTX},
    {"role": "user", "content": (
        "For connector 'application-debezium-connector', "
        "check its current status, query the 5m error rate and current consumer lag from Prometheus, "
        "then recommend actionable alerts (thresholds, severity) and dashboard panels. "
        "Use the available tools as needed."
    )},
]

# -----------------------------
# 1) First turn - let the model decide tool usage
# -----------------------------
first = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=TEMPERATURE,
)

choice = first.choices[0]
assistant_msg = choice.message
messages.append({"role": "assistant", "content": assistant_msg.content or ""})

# Satisfy tool calls (if any)
if assistant_msg.tool_calls:
    for tc in assistant_msg.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        handler = tool_handlers.get(name)
        if not handler:
            tool_content = json.dumps({"error": f"Unknown tool '{name}'"})
        else:
            result = handler(args)
            tool_content = json.dumps(result)

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_content,
        })

# -----------------------------
# 2) Final turn - streamed synthesis
# -----------------------------
stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=tools,            # keep available in case the model needs one more call
    tool_choice="auto",
    temperature=TEMPERATURE,
    stream=True,
)

buf = []
try:
    for chunk in stream:
        delta = chunk.choices[0].delta
        token = getattr(delta, "content", "")
        if token:
            buf.append(token)
            print(token, end="", flush=True)
finally:
    print("\n")

# Persist assistant reply to history (if you plan another turn)
final_answer = "".join(buf).strip()
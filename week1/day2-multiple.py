
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
    "name": "dns_lookup",
    "description": "Resolve a hostname to IPv4 and IPv6 addresses.",
    "parameters": {"type": "object", "properties": {"hostname": {"type": "string"}}, "required": ["hostname"]}
  },
  {
    "name": "get_last_ip",
    "description": "Get last known IPv4 from state store.",
    "parameters": {"type": "object", "properties": {}, "required": []}
  },
  {
    "name": "set_last_ip",
    "description": "Persist last known IPv4 to state store.",
    "parameters": {"type": "object", "properties": {"ip": {"type": "string"}}, "required": ["ip"]}
  },
  {
    "name": "read_file",
    "description": "Read file content from repository workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
  },
  {
    "name": "yaml_update",
    "description": "Update a scalar value at a key path (e.g., 'kafka.brokerIP') in a YAML file.",
    "parameters": {
      "type": "object",
      "properties": {"path": {"type": "string"}, "keyPath": {"type": "string"}, "newValue": {"type": "string"}},
      "required": ["path", "keyPath", "newValue"]
    }
  },
  {
    "name": "write_file",
    "description": "Write content to file path.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
  },
  {
    "name": "git_commit_push",
    "description": "Commit staged changes and push to branch, triggering CI.",
    "parameters": {"type": "object", "properties": {"branch": {"type": "string"}, "message": {"type": "string"}}, "required": ["branch", "message"]}
  },
  {
    "name": "notify_teams",
    "description": "Send a Teams message with context.",
    "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}}, "required": ["title", "text"]}
  }
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
# SYSTEM_CTX = (
#     "You are a concise Kafka/DevOps assistant. Environment: Confluent Platform with Kafka Connect "
#     "using Debezium source connectors. Metrics are exposed via JMX, collected by jmx_prometheus_javaagent, "
#     "scraped by Prometheus, and visualized with Grafana. Prefer owner-facing recommendations "
#     "(connector-level) with PromQL, thresholds, severity, and runbook steps. "
#     "Call tools when you need live metrics or connector status."
# )

SYSTEM_PROMPT = f"""
You are a DevOps DNS watcher agent. Your job:
1) Call stabilize_dns(HOSTNAME, STABILIZE_QUERIES, STABILIZE_DELAY_SEC).
2) Compare the stabilized IPv4 'primary' with get_last_ip().
3) If changed, update VALUES_FILE either by yaml_update(VALUES_FILE, YAML_KEY, ip) if key exists,
   otherwise fallback to replace_ip_literal(VALUES_FILE, old_ip, new_ip).
4) Commit & push via git_commit_push(GIT_BRANCH, "chore: update <host> <old_ip> -> <new_ip>").
5) Persist state with set_last_ip(new_ip).
6) Call notify_teams with a short operational summary.

Rules:
- Use tools for ALL actions and I/O; never hallucinate file paths.
- Only commit if there is an actual textual change.
- Always quote the IP in YAML values.
- Keep messages concise and operational.
- If hostname has multiple A records, select the 'primary' from stabilize_dns and include all_ipv4 in the summary.
- On failure, still notify_teams with a remediation hint.
- Do not push if new_ip == old_ip.
"""

# -----------------------------
# Conversation seed
# -----------------------------
# messages = [
#     {"role": "system", "content": SYSTEM_CTX},
#     {"role": "user", "content": (
#         "For connector 'application-debezium-connector', "
#         "check its current status, query the 5m error rate and current consumer lag from Prometheus, "
#         "then recommend actionable alerts (thresholds, severity) and dashboard panels. "
#         "Use the available tools as needed."
#     )},
# ]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": json.dumps({
        "hostname": "app-fomdev1-biabkafka02.biab.au.ing.net",
        "values_file": "~/P16424-confluent-kafka-replicator/helm/values.yaml",
        "yaml_key": 'YAML_KEY',
        "branch": 'master',
        "stabilize_queries": 'STABILIZE_QUERIES',
        "stabilize_delay_sec": 'STABILIZE_DELAY_SEC'
    })}
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
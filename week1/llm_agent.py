
#!/usr/bin/env python3
"""
LLM-powered DevOps agent using OpenAI() function calling.

Goal:
- Watch DNS IP for HOSTNAME.
- If it changes (stabilized), update values.yaml (key or literal).
- Commit & push to trigger CI.
- Notify Teams.

Side-effectful ops are implemented as deterministic tools.
The LLM orchestrates which tools to call and in what order.
"""

import os
import re
import json
import time
import socket
import subprocess
from pathlib import Path

from openai import OpenAI

# -------------------------
# Config (env vars)
# -------------------------
HOSTNAME   = os.getenv("HOSTNAME", "app-fomdev1-biabkafka02.biab.au.ing.net")
VALUES_FILE = os.getenv("VALUES_FILE", "deploy/helm/values.yaml")
YAML_KEY   = os.getenv("YAML_KEY", "kafka.brokerIP")  # e.g., "kafka.brokerIP"; leave blank for literal replace.
GIT_BRANCH = os.getenv("GIT_BRANCH", "main")
STATE_FILE = os.getenv("STATE_FILE", ".last_ip.txt")
STABILIZE_QUERIES = int(os.getenv("STABILIZE_QUERIES", "3"))
STABILIZE_DELAY_SEC = float(os.getenv("STABILIZE_DELAY_SEC", "5"))
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")

# -------------------------
# Deterministic tool impls
# -------------------------

def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout

def dns_once(hostname: str):
    """Try nslookup then socket; return ipv4 list."""
    try:
        res = subprocess.run(["nslookup", hostname], capture_output=True, text=True, timeout=10)
        ipv4 = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", res.stdout) if res.returncode == 0 else []
    except Exception:
        ipv4 = []
    if not ipv4:
        try:
            infos = socket.getaddrinfo(hostname, None)
            ipv4 = list(dict.fromkeys([a[4][0] for a in infos if a[0] == socket.AF_INET]))
        except Exception:
            ipv4 = []
    return ipv4

def stabilize_dns(hostname: str, queries: int, delay_sec: float):
    """Require N consecutive identical IPv4[0] results to treat as stable."""
    stable = ""
    count = 0
    all_last = []
    for _ in range(queries * 5):  # cap attempts
        ipv4 = dns_once(hostname)
        primary = ipv4[0] if ipv4 else ""
        if not primary:
            time.sleep(delay_sec); continue
        if primary == stable:
            count += 1
        else:
            stable = primary
            count = 1
        all_last = ipv4
        if count >= queries:
            return {"primary": stable, "all_ipv4": all_last}
        time.sleep(delay_sec)
    return {"primary": stable, "all_ipv4": all_last}

def get_last_ip():
    p = Path(STATE_FILE)
    return {"ip": p.read_text().strip()} if p.exists() else {"ip": ""}

def set_last_ip(ip: str):
    Path(STATE_FILE).write_text(ip)
    return {"ok": True}

def read_file(path: str):
    p = Path(path)
    if not p.exists():
        return {"exists": False, "content": ""}
    return {"exists": True, "content": p.read_text()}

def yaml_update(path: str, keyPath: str, newValue: str):
    """
    Simple line-based replacement for a scalar key:
      myKey: "1.2.3.4"
    Will quote the value.
    """
    text = Path(path).read_text()
    key = keyPath.split(".")[-1]
    pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*.*?$", re.MULTILINE)

    def _repl(m):
        indent = m.group(1)
        return f'{indent}{key}: "{newValue}"'

    new_text, n = pattern.subn(_repl, text, count=1)
    if n == 0:
        return {"changed": False, "content": text}
    Path(path).write_text(new_text)
    return {"changed": True, "content": new_text}

def replace_ip_literal(path: str, old_ip: str, new_ip: str):
    text = Path(path).read_text()
    if old_ip and old_ip in text:
        new_text = text.replace(old_ip, new_ip)
        Path(path).write_text(new_text)
        return {"changed": new_text != text}
    else:
        # Fallback: replace first IPv4 literal
        found = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
        if found:
            new_text = text.replace(found[0], new_ip)
            Path(path).write_text(new_text)
            return {"changed": new_text != text}
    return {"changed": False}

def git_commit_push(branch: str, message: str):
    _run(["git", "add", "-A"])
    status = _run(["git", "status", "--porcelain"]).strip()
    if not status:
        return {"pushed": False, "reason": "no changes"}
    _run(["git", "commit", "-m", message])
    _run(["git", "push", "origin", branch])
    return {"pushed": True}

def notify_teams(title: str, text: str):
    if not TEAMS_WEBHOOK_URL:
        return {"sent": False, "reason": "no webhook"}
    import urllib.request
    body = json.dumps({
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "summary": title,
        "themeColor": "0076D7",
        "title": title,
        "text": text
    }).encode("utf-8")
    req = urllib.request.Request(TEAMS_WEBHOOK_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"sent": True, "status": resp.status}
    except Exception as e:
        return {"sent": False, "error": str(e)}

# Map tool names to callables
TOOLS_IMPL = {
    "stabilize_dns": stabilize_dns,
    "get_last_ip": get_last_ip,
    "set_last_ip": set_last_ip,
    "read_file": read_file,
    "yaml_update": yaml_update,
    "replace_ip_literal": replace_ip_literal,
    "git_commit_push": git_commit_push,
    "notify_teams": notify_teams,
}

# -------------------------
# LLM wiring (OpenAI)
# -------------------------

client = OpenAI()  # uses OPENAI_API_KEY from env
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # adjust if you prefer gpt-4.1 or o3-mini for tool use

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

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "stabilize_dns",
            "description": "Resolve and stabilize the IP for a hostname across N queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {"type": "string"},
                    "queries": {"type": "integer"},
                    "delay_sec": {"type": "number"}
                },
                "required": ["hostname", "queries", "delay_sec"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_last_ip",
            "description": "Get last known IPv4 (from state).",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_last_ip",
            "description": "Persist last known IPv4 to state.",
            "parameters": {"type": "object", "properties": {"ip": {"type": "string"}}, "required": ["ip"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file content from repository workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "yaml_update",
            "description": "Update a scalar value at 'keyPath' in YAML file 'path'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "keyPath": {"type": "string"},
                    "newValue": {"type": "string"}
                },
                "required": ["path", "keyPath", "newValue"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_ip_literal",
            "description": "Replace an existing IP literal in file with new IP (fallback).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_ip": {"type": "string"},
                    "new_ip": {"type": "string"}
                },
                "required": ["path", "old_ip", "new_ip"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit_push",
            "description": "Commit and push changes to branch to trigger CI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["branch", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "notify_teams",
            "description": "Send a Teams message with context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["title", "text"]
            }
        }
    }
]

def call_tool(name: str, args: dict):
    if name not in TOOLS_IMPL:
        return {"error": f"Unknown tool {name}"}
    try:
        return TOOLS_IMPLname
    except TypeError as e:
        return {"error": f"Bad args: {e}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    # Prepare conversation with config context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "hostname": HOSTNAME,
            "values_file": VALUES_FILE,
            "yaml_key": YAML_KEY,
            "branch": GIT_BRANCH,
            "stabilize_queries": STABILIZE_QUERIES,
            "stabilize_delay_sec": STABILIZE_DELAY_SEC
        })}
    ]

    while True:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message

        # If model wants to call tools, execute them and continue the loop
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                result = call_tool(fn, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn,
                    "content": json.dumps(result)
                })
            continue

        # No tool calls => model returned final content (summary / no-op)
        print(msg.content)
        break

if __name__ == "__main__":
    main()

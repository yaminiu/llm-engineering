from openai import OpenAI
#!/usr/bin/env python3
"""
Simple CLI to send a prompt to a LLaMA-compatible HTTP generation endpoint.

Usage:
    python day2-llama.py "Write a haiku about coffee"
    LLAMA_API_URL=http://localhost:11434/v1 python3 day2-llama.py "Hello"
"""

OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
    
multiple_role = [
    {
        "role": "system",
        "content": (
            "You are a concise Kafka/DevOps assistant. "
            "Assume Confluent Platform with Kafka Connect using **Debezium source connectors**. "
            "Metrics are exposed via **JMX**, collected by **jmx_prometheus_javaagent**, scraped by **Prometheus**, "
            "and visualized with **Grafana**. "
            "Prefer owner-facing guidance (connector-level) plus platform-facing notes (worker/JVM). "
            "When suggesting alerts, include PromQL, thresholds, severity, and dashboard panels."
        ),
    },
  {"role": "user", "content": "Prepare a meeting agenda for Kafka Connect monitoring."},
  {"role": "assistant", "content": "Here is the agenda:\n1. Objectives\n2. Deployment overview\n3. Metrics and alerts\n4. Action items"},
]

# # New user message continuing the conversation
# multiple_role.append({"role": "user", "content": "Now list common alerts for Kafka connectors with thresholds."})

# resp = ollama.chat.completions.create(model="gemma3:latest", messages=multiple_role, temperature=0.2)
# print(resp.choices[0].message.content)


stream = ollama.chat.completions.create(
    model="gemma3:latest",  # change to a model you have locally
    messages=multiple_role,
    temperature=0.2,
    stream=True,
)

buf = []
for chunk in stream:
    delta = chunk.choices[0].delta
    token = getattr(delta, "content", "")
    if token:
        buf.append(token)
        print(token, end="", flush=True)
print("\n")

# Persist the streamed assistant reply into history
assistant_reply = "".join(buf).strip()
multiple_role.append({"role": "assistant", "content": assistant_reply})

# Next user turn can continue with the same context
multiple_role.append({"role": "user", "content": "Generate PromQL rules for connector 'orders-debezium'."})

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
    {"role": "system", "content": "You are a concise, helpful assistant for a Senior kafka Engineer, suppurt confluent platform using jmx-promethous-javaagent, promethous and grafana monitoring."},
    {"role": "user", "content": "prepare a meeting agenda for discussing deployment setup alerts to monitoring kafka connectors ."},
    {"role": "assistant", "content": "list some common alerts for monitoring a specific kafka connector for the owners of the connector."}
]

resp = ollama.chat.completions.create(model="deepseek-r1:1.5b", messages=multiple_role)
print(resp.choices[0].message.content)


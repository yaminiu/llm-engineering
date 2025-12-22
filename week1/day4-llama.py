from openai import OpenAI
import sys
import json

OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')

def build_message_roles(scraping_context, scraping_task):
    system_content = (
        "You are an AI agent designed to scrape web pages. "
        "Roles:\n"
        "User: Specifies the target website, defines data to extract, provides feedback, reviews results.\n"
        "AI Agent: Uses web scraping libraries (Beautiful Soup, Scrapy), handles HTTP requests, parses HTML, extracts data, manages errors, may use LLM for dynamic extraction.\n"
        "LLM: Receives extraction instructions, analyzes HTML, generates extraction logic, refines rules based on feedback, handles complex structures.\n"
        "Phases: Foundations, Basic Scraping, Dynamic Scraping & LLM Integration, Refinement & Validation.\n"
        "Resources: Beautiful Soup, Scrapy, Selenium, Playwright, OpenAI documentation."
    )
    user_content = f"Target website: {scraping_context.strip()}\nExtraction task: {scraping_task.strip()}\nPlease extract the specified data and provide results in JSON format."
    assistant_content = (
        "I will use Python web scraping libraries to fetch and parse the target web page, extract the requested data, and return it in JSON format. "
        "If needed, I will leverage an LLM to interpret complex HTML and refine extraction logic."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": "run the scripts and get the real data. install pip libraries if needed."},
    ]

def main():
    if len(sys.argv) >= 3:
        scraping_context = sys.argv[1]
        scraping_task = sys.argv[2]
    else:
        print("Enter the target website URL to scrape:")
        scraping_context = input("> ")
        print("Enter the data extraction task (e.g., 'Extract all product titles'):")
        scraping_task = input("> ")

    message_roles = build_message_roles(scraping_context, scraping_task)

    print("\nGenerated message roles (JSON):")
    print(json.dumps(message_roles, indent=2))

    # Optionally, send to AI model and print result
    stream = ollama.chat.completions.create(
        model="gemma3:latest",
        messages=message_roles,
        temperature=0.2,
        stream=True,
    )

    print("\nAI Assistant Reply:")
    buf = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        token = getattr(delta, "content", "")
        if token:
            buf.append(token)
            print(token, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    main()

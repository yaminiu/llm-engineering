"""
A full business solution: Company Brochure Generator

This script demonstrates how to build a product that generates a brochure for a company
using LLMs and web scraping. Given a company name and its website, it:
- Scrapes the website for links and content
- Uses an LLM to select relevant links for a brochure (e.g., About, Careers)
- Fetches the content of those links
- Uses an LLM to generate a brochure in markdown

Dependencies:
- openai
- requests
- beautifulsoup4
- IPython (for Markdown display, optional)

Usage:
    python day5.py <company_name> <company_url>
    # Or run interactively and call create_brochure() or stream_brochure()
"""

from openai import OpenAI
import sys
import json
from IPython.display import Markdown, display, update_display
from scraper import fetch_website_links, fetch_website_contents

# Optional: for pretty markdown display in Jupyter/IPython
try:
    from IPython.display import Markdown, display, update_display
except ImportError:
    Markdown = display = update_display = None

# --- Scraper utilities (assume these are implemented in scraper.py or inline here) ---

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- LLM setup ---

OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
MODEL = "gemma3:latest"

def fetch_website_links(url):
    """Fetch all links from the given website URL."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        resp.encoding = 'utf-8'  # Ensure UTF-8 decoding
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Skip mailto, javascript, etc.
            if href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            # Convert relative links to absolute
            full_url = urljoin(url, href)
            links.append(full_url)
        return links
    except Exception as e:
        print(f"Error fetching links from {url}: {e}")
        return []

def fetch_website_contents(url):
    """Fetch and return the text content of a web page."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        resp.encoding = 'utf-8'  # Ensure UTF-8 decoding
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"[Error fetching {url}: {e}]"



# --- Prompts ---

link_system_prompt = """You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(url):
    user_prompt = f"""Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company, 
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt

def select_relevant_links(url):
    print(f"Selecting relevant links for {url} by calling LLM model")
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    print(f"Found {len(links['links'])} relevant links")
    return links

def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result

brochure_system_prompt = """You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
the length of the brochure should be about 500 words.
# """

# brochure_system_prompt = """
# You are an assistant that analyzes the contents of several relevant pages from a company website
# and creates a short, humorous, entertaining, witty brochure about the company for prospective customers, investors and recruits.
# Respond in markdown without code blocks.
# Include details of company culture, customers and careers/jobs if you have the information.
# the length of the brochure should be about 1500 words.
# """

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

def create_brochure(company_name, url):
    response = ollama.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
    )
    result = response.choices[0].message.content
    display(Markdown(result))

def stream_brochure(company_name, url):
    stream = ollama.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
        stream=True
    )    
    response = ""
    if display is not None and update_display is not None:
        display_handle = display(Markdown(""), display_id=True)
    else:
        display_handle = None
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        if display_handle is not None:
            update_display(Markdown(response), display_id=display_handle.display_id)
        else:
            # Fallback: print to console
            print(response, end="\r", flush=True)
    if display_handle is None:
        print("\n" + response)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        company_name = sys.argv[1]
        url = sys.argv[2]
        print(f"Generating brochure for {company_name} ({url}) ...")
        stream_brochure(company_name, url) # ← causes the TypeError
        # print(get_brochure_user_prompt(company_name, url))
    else:
        print("Usage: python day5.py <company_name> <company_url>")
        print("Or call create_brochure(company_name, url) or stream_brochure(company_name, url) in code.")

# week2/gradio_app.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from scraper import fetch_website_contents

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging
# You can choose whichever providers you like - or all Ollama

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")
    
# Connect to OpenAI, Anthropic and Google; comment out the Claude or Google lines if you're not using them

openai = OpenAI()

gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)


# Let's create a call that streams back results
# If you'd like a refresher on Generators (the "yield" keyword),
# Please take a look at the Intermediate Python guide in the guides folder

def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
        
def stream_gemini(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = gemini.chat.completions.create(
        model='gemini-2.5-pro',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result
    
def stream_brochure(company_name, url, model):
    yield ""
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += fetch_website_contents(url)
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result
    

# Again this is typical Experimental mindset - I'm changing the global variable we used above:

system_message = """
You are an assistant that analyzes the contents of a company website landing page
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
"""
        
name_input = gr.Textbox(label="Company name:")
url_input = gr.Textbox(label="Landing page URL including http:// or https://")
model_selector = gr.Dropdown(["GPT", "Gemini"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_brochure,
    title="Brochure Generator", 
    inputs=[name_input, url_input, model_selector], 
    outputs=[message_output], 
    examples=[
            ["ING", "https://www.ing.com.au/", "GPT"],
            ["Edward Donner", "https://edwarddonner.com", "Gemini"]
        ], 
    flagging_mode="never"
    )
view.launch()
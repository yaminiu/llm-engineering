
from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from torch import no_grad
import torch


load_dotenv(override=True)
# openai_api_key = os.getenv('OPENAI_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')
hf_token = os.getenv('HF_TOKEN')

print("Logging into Hugging Face Hub... with token starting:", hf_token[:8])
login(hf_token, add_to_git_credential=True)


os.environ["HF_HOME"] = "/opt/huggingface"          # central cache dir
os.environ["TRANSFORMERS_CACHE"] = "/opt/huggingface/transformers"

# Model ID for chat/instruction variant
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto"             # Auto place on GPU if available
)

# Define chat messages
messages = [
    {"role": "system", "content": "You are a helpful assistant that writes professional email replies."},
    {"role": "user", "content": "Draft a courteous email confirming the rollback and requesting logs."}
]

# Render the chat template into a single prompt string
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,               # Get raw string first
    add_generation_prompt=True    # Add assistant turn marker for generation
)

print("Rendered prompt:\n", prompt_text)

# Tokenize the rendered prompt
inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

# Generate a reply
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode the generated text
reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\nGenerated reply:\n", reply)

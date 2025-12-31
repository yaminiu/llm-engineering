from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import pipeline, AutoTokenizer


load_dotenv(override=True)
# openai_api_key = os.getenv('OPENAI_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')
hf_token = os.getenv('HF_TOKEN')

print("Logging into Hugging Face Hub... with token starting:", hf_token[:8])
login(hf_token, add_to_git_credential=True)


os.environ["HF_HOME"] = "/opt/huggingface"          # central cache dir
os.environ["TRANSFORMERS_CACHE"] = "/opt/huggingface/transformers"


# Choose the model id (base or instruct)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"    


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,        # fast tokenizer backend (recommended)
    trust_remote_code=True  # safe for official HF repos
)


text = "Please summarize the latest deployment status and highlight any blockers."
tokens = tokenizer.encode(text)
print("Input text:", text)
print("Token IDs:", tokens)
print("Number of tokens:", len(tokens))

tokenizer.decode(tokens)  # convert back to text
print("Decoded text:", tokenizer.decode(tokens))

tokenizer.batch_decode(tokens)  # decode multiple sequences
print("Batch decoded texts:", tokenizer.batch_decode(tokens))

# tokenizer.get_vocab_size()  # size of the vocabulary
# print("Vocabulary size:", tokenizer.get_vocab_size())
tokenizer.model_max_length  # max length the model can handle
print("Model max length:", tokenizer.model_max_length) 
# tokenizer.get_added_vocab()
# print("Added vocabulary:", tokenizer.get_added_vocab())
# special tokens added to the model
print(len(tokenizer.all_special_tokens), "special tokens:", tokenizer.all_special_tokens)


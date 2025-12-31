from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import pipeline


load_dotenv(override=True)
# openai_api_key = os.getenv('OPENAI_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')
hf_token = os.getenv('HF_TOKEN')

print("Logging into Hugging Face Hub... with token starting:", hf_token[:8])
login(hf_token, add_to_git_credential=True)


os.environ["HF_HOME"] = "/opt/huggingface"          # central cache dir
os.environ["TRANSFORMERS_CACHE"] = "/opt/huggingface/transformers"


nlp = pipeline("sentiment-analysis")  # choose a task
result = nlp("I love M365 Copilot—it's super helpful!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999...}]


sentiment = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    # device="gpu"  # use GPU if available
)

print(sentiment([
    "Downtime was minimal.",
    "This broke our production cluster."
]))
# Batch in → list out


generator = pipeline(
    "text-generation",
    model="gpt2", # small demo model
    device_map="auto"  # use GPU if available
)

print(generator("Write a short release note: ", max_new_tokens=40))



summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
We deployed v2.3 of the service with blue/green. Canary at 5% traffic passed SLOs.
Roll-forward completed within 12 minutes. No incidents reported.
"""
print(summarizer(text, max_length=60, min_length=20, do_sample=False))



ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Alice moved services from Sydney to Singapore with AWS SSO."))
# [{'entity_group': 'PER', 'word': 'Alice', ...}, {'entity_group': 'LOC', 'word': 'Sydney', ...}]


# 4) Zero-shot classification
zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zsc("Rotate keys and update IAM policies.", ["security", "devops", "finance"]))

# use case: classify support tickets into teams
labels = ["security", "devops", "finops", "network", "compliance"]
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli")

tickets = [
    "Investigate the suspicious login from Russia.",
    "Our AWS costs have increased by 30% this month.",
    "Set up a VPN for the new office in Berlin."
]   
for ticket in tickets:
    result = classifier(ticket, labels)
    print(f"Ticket: {ticket}\nClassification: {result['labels'][0]}\n")
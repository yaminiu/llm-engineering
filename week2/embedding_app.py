
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Sentences
sentences = [
    "I love eating pizza.",
    "running is my favorite sport.",
    "The weather is sunny today."
]

# 3. Generate embeddings
embeddings = model.encode(sentences)

# 4. Query
query = "What is your favorite sprort?"
query_embedding = model.encode([query])

# 5. Compute similarity
similarities = cosine_similarity(query_embedding, embeddings)
best_match_index = np.argmax(similarities)
print(f"Best match: {sentences[best_match_index]}")

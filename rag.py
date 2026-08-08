import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import ollama

# -------------------------
# Load Environment Variables
# -------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# -------------------------
# Connect to Pinecone
# -------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------
# Load Embedding Model
# -------------------------
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -------------------------
# User Question
# -------------------------
question = input("Ask me: ")

# -------------------------
# Create Query Embedding
# -------------------------
query_embedding = embedding_model.encode(question).tolist()

# -------------------------
# Retrieve Top Chunks
# -------------------------
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

context = ""

for match in results["matches"]:
    context += match["metadata"]["text"] + "\n\n"

# -------------------------
# Prompt
# -------------------------
prompt = f"""
You are Meer Abdullah's AI portfolio assistant.

Answer ONLY using the provided context.
If the answer is not in the context, reply:
"I don't have that information in my knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

# -------------------------
# Generate Response
# -------------------------
response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAnswer:\n")
print(response["message"]["content"])
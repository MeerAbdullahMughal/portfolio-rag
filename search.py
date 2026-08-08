import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Ask a question
query = "What programming languages does Meer know?"

# Convert query into embedding
query_embedding = model.encode(query).tolist()

# Search Pinecone
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

# Print results
print("\nRetrieved Chunks:\n")

for match in results["matches"]:
    print("=" * 60)
    print(match["metadata"]["text"])
    print()
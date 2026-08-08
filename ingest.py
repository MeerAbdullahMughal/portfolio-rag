import os
import fitz
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# -----------------------------
# Connect to Pinecone
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -----------------------------
# Load Embedding Model
# -----------------------------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# Read PDF
# -----------------------------
doc = fitz.open("portfolio_knowledge.pdf")

text = ""

for page in doc:
    text += page.get_text()

doc.close()

# -----------------------------
# Split into Chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")

# -----------------------------
# Generate embeddings and upload
# -----------------------------
vectors = []

for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()

    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {
            "text": chunk
        }
    })

index.upsert(vectors=vectors)

print(f"Successfully uploaded {len(vectors)} chunks to Pinecone!")
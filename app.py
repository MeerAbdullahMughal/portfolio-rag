import os
import ollama

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="Meer Abdullah Mughal Portfolio Chatbot",
    description="AI-powered RAG chatbot for Meer Abdullah Mughal's portfolio.",
    version="1.0.0"
)

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)\
# ----------------------------
# Pinecone  
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ----------------------------
# Embedding Model
# ----------------------------
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# ----------------------------
# Request Model
# ----------------------------
class ChatRequest(BaseModel):
    question: str


# ----------------------------
# Chat Endpoint
# ----------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question

    # Convert question to embedding
    query_embedding = embedding_model.encode(question).tolist()

    # Retrieve relevant chunks
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    # Build context
    context = ""

    for match in results["matches"]:
        context += match["metadata"]["text"] + "\n\n"

    # ----------------------------
    # System Prompt
    # ----------------------------
    system_prompt = """
You are Meer Abdullah Mughal's official AI Assistant.

Your purpose is to answer questions about Meer Abdullah Mughal using ONLY the information provided in the retrieved context.

You can answer questions about:
- Education
- Skills
- Technical expertise
- Projects
- Internship
- Experience
- Programming languages
- Frameworks
- Artificial Intelligence and Machine Learning
- Career goals
- Portfolio

Rules:

1. Never invent or assume information.

2. If the answer is not present in the context, reply exactly:
"I don't have that information in my knowledge base."

3. Keep answers professional, concise and friendly.

4. Use bullet points whenever appropriate.

5. If someone greets you, greet them naturally.

6. If someone asks "Who is Meer?", introduce him professionally.

7. Never mention Pinecone, embeddings, retrieved chunks, vector databases, or internal implementation unless explicitly asked.

8. Answer as Meer's AI Assistant, never as ChatGPT.
"""

    # ----------------------------
    # User Prompt
    # ----------------------------
    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    # ----------------------------
    # Generate Response
    # ----------------------------
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    # ----------------------------
    # Return Response
    # ----------------------------
    return {
        "answer": response["message"]["content"]
    }


# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def home():
    return {
        "message": "Meer Abdullah Mughal Portfolio Chatbot API is running..."
    }
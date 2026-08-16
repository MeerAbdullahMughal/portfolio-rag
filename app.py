import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Meer Abdullah Mughal Portfolio Chatbot",
    description="AI-powered RAG chatbot for Meer Abdullah Mughal's portfolio.",
    version="2.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Pinecone
# =========================================================

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)


# =========================================================
# Embedding Model
# =========================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# =========================================================
# LangChain Ollama
# =========================================================

llm = ChatOllama(
    model="qwen2.5:3b"
)


# =========================================================
# Chat State
# =========================================================

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    answer: str

# =========================================================
# Node 1: Retrieve
# =========================================================
def retrieve(state: ChatState):

    question = state["messages"][-1].content

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    context = ""

    for match in results["matches"]:
        context += (
            match["metadata"]["text"]
            + "\n\n"
        )

    return {
        "context": context
    }

# =========================================================
# Node 2: Generate
# =========================================================
def generate(state: ChatState):

    context = state["context"]

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

    messages = [
        SystemMessage(
            content=system_prompt
        )
    ]

    messages.extend(state["messages"])

    messages.append(
        HumanMessage(
            content=f"""
Retrieved context:

{context}

Use the retrieved context to answer the user's latest question.
"""
        )
    )

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "messages": [
            AIMessage(content=response.content)
        ]
    }


# =========================================================
# Build LangGraph
# =========================================================

graph_builder = StateGraph(ChatState)

graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)


graph_builder.add_edge(
    START,
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "generate"
)

graph_builder.add_edge(
    "generate",
    END
)


memory = MemorySaver()

graph = graph_builder.compile(
    checkpointer=memory
)


# =========================================================
# Request Model
# =========================================================

class ChatRequest(BaseModel):
    question: str
    thread_id: str

# =========================================================
# Chat Endpoint
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=request.question)
            ],
            "context": "",
            "answer": ""
        },
        config={
            "configurable": {
                "thread_id": request.thread_id
            }
        }
    )

    return {
        "answer": result["answer"]
    }

# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Meer Abdullah Mughal Portfolio Chatbot API is running..."
    }
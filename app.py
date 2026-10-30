from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict

import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

load_dotenv(Path(__file__).with_name(".env"))


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Meer Abdullah Mughal Portfolio Chatbot",
    description="AI-powered RAG chatbot for Meer Abdullah Mughal's portfolio.",
    version="2.0.0"
)

@app.get("/health")
def health():
    knowledge_file = Path(__file__).with_name("portfolio_knowledge.pdf")
    return {
        "status": "ok",
        "chat_ready": knowledge_file.is_file() and knowledge_file.stat().st_size > 0,
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_chat_services():
    knowledge_file = Path(__file__).with_name("portfolio_knowledge.pdf")
    if not knowledge_file.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Portfolio knowledge file not found: {knowledge_file.name}",
        )

    with fitz.open(knowledge_file) as document:
        text = "\n".join(page.get_text() for page in document)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    ).split_text(text)
    if not chunks:
        raise HTTPException(
            status_code=503,
            detail=f"No searchable text found in {knowledge_file.name}",
        )

    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    llm = ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0,
        num_predict=120,
    )
    return chunks, vectorizer, chunk_vectors, llm


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

    chunks, vectorizer, chunk_vectors, _ = get_chat_services()
    question = state["messages"][-1].content

    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_vectors).ravel()
    best_matches = scores.argsort()[-3:][::-1]
    context = "\n\n".join(
        chunks[index]
        for index in best_matches
        if scores[index] > 0
    )

    return {
        "context": context
    }


def normalize_faq_line(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip("?.!")


# =========================================================
# Node 2: Generate
# =========================================================
def generate(state: ChatState):

    _, _, _, llm = get_chat_services()
    context = state["context"]
    question = state["messages"][-1].content
    if not context:
        return {
            "answer": "I don't have that information in my knowledge base.",
            "messages": [
                AIMessage(
                    content="I don't have that information in my knowledge base."
                )
            ],
        }

    context_lines = [line.strip() for line in context.splitlines()]
    normalized_question = normalize_faq_line(question)
    for line_index, line in enumerate(context_lines):
        if normalize_faq_line(line) != normalized_question:
            continue
        answer = next(
            (
                answer_line
                for answer_line in context_lines[line_index + 1:]
                if answer_line and not answer_line.endswith("?")
            ),
            None,
        )
        if answer:
            return {
                "answer": answer,
                "messages": [AIMessage(content=answer)],
            }

    system_prompt = """
You are Meer Abdullah Mughal's official AI Assistant.

Your purpose is to answer questions about Meer Abdullah Mughal using ONLY the information provided in the retrieved context.

Answer the user's question directly in at most two short sentences. Do not add
definitions, general advice, or facts that are not explicitly in the context.
If the context contains a direct answer, repeat only the relevant details.

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
def home() -> FileResponse:
    return FileResponse(Path(__file__).with_name("index.html"))
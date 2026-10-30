# Portfolio RAG

Meer Abdullah Mughal's portfolio chatbot. FastAPI retrieves relevant passages
from the included PDF and uses a local Ollama model to answer questions.

## Requirements

- Python 3.14 (or a Python version supported by the pinned packages)
- Ollama for Windows

## Windows setup

Run these commands from the project directory in PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The embedding stack includes PyTorch and can require several gigabytes during
installation. If disk space is limited and the required ML packages are already
installed for your system Python, create a venv that can reuse them and install
the app adapter:

```powershell
py -3.14 -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install langchain-ollama==1.1.0
```

This fallback is less isolated than installing the complete requirements.

The chatbot uses `portfolio_knowledge.pdf` directly and does not need Pinecone
credentials or a `.env` file.

## Start the local model

Ensure the Ollama application/service is running, then download the smaller
model used by the app:

```powershell
ollama pull qwen2.5:0.5b
ollama list
```

The smaller model avoids the memory-allocation failure observed with the 3B
model on machines with limited free RAM.

## Optional: upload the portfolio knowledge base to Pinecone

Only needed if you want to use the standalone Pinecone ingestion/search scripts.
Configure `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` in `.env`, and create a
384-dimensional cosine index before running:

```powershell
python ingest.py
```

This reads `portfolio_knowledge.pdf` and upserts its chunks into the configured
Pinecone index. The chatbot does not use this index.

## Run the API

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/` for the portfolio frontend. Its chat widget sends
questions to the same server's `/chat` API. Open
`http://127.0.0.1:8000/docs` for the interactive API documentation and
`http://127.0.0.1:8000/health` to check API status.

The frontend can also be opened directly as `index.html`; in that case, the
chat widget connects to `http://127.0.0.1:8000/chat`. The API and Ollama must
be running.

When opening the frontend through VS Code Live Server (for example,
`http://127.0.0.1:5500`), the chat widget automatically sends API requests to
the backend at `http://127.0.0.1:8000`.

POST a chat question to `/chat` with JSON shaped like:

```json
{
  "question": "What projects has Meer worked on?",
  "thread_id": "local-session-1"
}
```

The `thread_id` groups conversation messages in the process-local memory
checkpoint; restarting the API clears that memory.

## Troubleshooting

- **Ollama connection/model error:** start Ollama and verify
  `ollama list` includes `qwen2.5:0.5b`.
- **No knowledge retrieved:** check that `portfolio_knowledge.pdf` exists and
  contains selectable text.
- **Chat UI cannot connect:** confirm the API is running at
  `http://127.0.0.1:8000` and use the UI served from that API or VS Code Live
  Server on port 5500.
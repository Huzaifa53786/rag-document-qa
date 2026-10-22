# 📄 AI-Based Document Question Answering System (RAG)

Ask questions about your PDFs and text files in natural language. The system retrieves the most relevant passages first, then has an LLM answer using only that context — reducing hallucinations and returning the sources.

**Tech stack:** Python · LangChain · OpenAI or Groq (free tier) LLM · ChromaDB · HuggingFace embeddings · FastAPI · Streamlit · Docker

## Architecture

```
Upload → Load → Chunk → Embed → ChromaDB
                                   │
Question → Embed → Top-k semantic search
                                   │
                  Retrieved context + question → LLM → Answer + Sources
```

## Features

- PDF and TXT ingestion with recursive chunking (1000 chars, 200 overlap)
- Local embeddings (`all-MiniLM-L6-v2`) — no embedding API cost
- LLM via **OpenAI** or **Groq free tier** (set either key in `.env`)
- Persistent vector store (ChromaDB)
- Grounded answers with source file and page references
- FastAPI REST API with auto-generated Swagger docs
- Streamlit chat UI with graceful offline-API handling
- Docker / docker-compose support
- Keyless smoke test (`tests/test_api.py`)

## Project structure

```
app/
  rag.py            # ingestion, retrieval, generation
  main.py           # FastAPI endpoints
tests/
  test_api.py       # keyless smoke test (no API key needed)
streamlit_app.py    # chat UI
data/sample.txt     # sample document for trying it out
Dockerfile
docker-compose.yml
requirements.txt
```

## Setup

```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the keyless smoke test (no API key needed)

```bash
python tests/test_api.py
```

### Run the API

```bash
uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

### Run the chat UI (second terminal)

```bash
streamlit run streamlit_app.py
```

### Run with Docker

```bash
docker compose up --build
```

API on :8000, UI on :8501.

## API

| Method | Endpoint  | Description                  |
|--------|-----------|------------------------------|
| GET    | `/health` | Health check                 |
| POST   | `/upload` | Upload and index a PDF/TXT   |
| POST   | `/ask`    | Ask a question               |

```bash
curl -X POST http://localhost:8000/upload -F "file=@data/sample.txt"

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do employees get?", "top_k": 4}'
```

Example response:

```json
{
  "answer": "Full-time employees accrue 20 days of paid annual leave per year.",
  "sources": [{"file": "sample.txt", "page": null}]
}
```

## Configuration

Set in `.env`: `OPENAI_API_KEY` **or** `GROQ_API_KEY` (Groq has a free tier), plus optional `LLM_MODEL` (default `gpt-4o-mini`; use a Groq model name like `llama-3.1-8b-instant` when using Groq), `CHUNK_SIZE`, `CHUNK_OVERLAP`.

> **Note:** the model name in `.env` must match your provider. Groq model IDs (e.g. `llama-3.1-8b-instant`) differ from OpenAI's.

## Demo

_Add a GIF or screenshot here._

## Future improvements

- Reranking (cross-encoder) for better retrieval precision
- Hybrid search (BM25 + vectors)
- Per-user collections and authentication
- Streaming responses
- Evaluation with RAGAS

## License

MIT

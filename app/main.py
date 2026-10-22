import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag import ingest, answer

app = FastAPI(title="Document QA (RAG)")
os.makedirs("data", exist_ok=True)


class Question(BaseModel):
    question: str
    top_k: int = 8


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported")
    path = os.path.join("data", file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    n = ingest(path)
    return {"filename": file.filename, "chunks_indexed": n}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    try:
        return answer(q.question, q.top_k)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
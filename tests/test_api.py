"""Keyless smoke test for the FastAPI layer.

Runs the full request/response cycle with the RAG pipeline stubbed out,
so it works without an OPENAI_API_KEY, GROQ_API_KEY, or embedding model download.

Usage (after `pip install -r requirements.txt`):
    python tests/test_api.py
"""
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.main as main  # noqa: E402  (importing triggers app.rag -> needs deps installed)


# ---- Stub the RAG pipeline (no LLM call, no embeddings) ----
def fake_ingest(path: str) -> int:
    return 7


def fake_answer(question: str, k: int = 4) -> dict:
    if "zzzz-not-in-docs" in question:
        return {"answer": "I couldn't find that in the documents.", "sources": []}
    return {
        "answer": "Employees accrue 20 days of paid annual leave per year.",
        "sources": [{"file": "sample.txt", "page": None}],
    }


main.ingest = fake_ingest
main.answer = fake_answer

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("Running smoke tests...")

r = client.get("/health")
check("GET /health -> 200", r.status_code == 200, r.text)
check("GET /health body", r.json() == {"status": "ok"}, r.text)

r = client.post("/upload", files={"file": ("sample.txt", io.BytesIO(b"hello"), "text/plain")})
check("POST /upload .txt -> 200", r.status_code == 200, r.text)
check("POST /upload reports chunks", r.json().get("chunks_indexed") == 7, r.text)

r = client.post("/upload", files={"file": ("evil.exe", io.BytesIO(b"x"), "application/octet-stream")})
check("POST /upload .exe -> 400", r.status_code == 400, r.text)

r = client.post("/ask", json={"question": "How much annual leave do I get?", "top_k": 4})
check("POST /ask valid -> 200", r.status_code == 200, r.text)
body = r.json()
check("POST /ask has answer", "annual leave" in body.get("answer", ""), r.text)
check("POST /ask has sources", body.get("sources") == [{"file": "sample.txt", "page": None}], r.text)

r = client.post("/ask", json={"question": "zzzz-not-in-docs"})
check("POST /ask out-of-scope -> grounded refusal",
      r.status_code == 200 and "couldn't find" in r.json().get("answer", ""), r.text)

r = client.post("/ask", json={"question": "   "})
check("POST /ask blank -> 400", r.status_code == 400, r.text)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)

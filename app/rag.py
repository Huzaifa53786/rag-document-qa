from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        timeout=30,
        max_retries=1,
    )
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=30,
        max_retries=1,
    )

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}

Answer:""")


def ingest(path: str) -> int:
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")

    chunks = splitter.split_documents(loader.load())
    for c in chunks:
        c.metadata["source"] = os.path.basename(path)
    vectorstore.add_documents(chunks)
    return len(chunks)


def answer(question: str, k: int = 8) -> dict:
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)

    response = (prompt | llm).invoke({"context": context, "question": question})

    seen, sources = set(), []
    for d in docs:
        page = d.metadata.get("page")
        key = (d.metadata.get("source"), page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": d.metadata.get("source"),
                        "page": page + 1 if isinstance(page, int) else page})
    return {"answer": response.content, "sources": sources}
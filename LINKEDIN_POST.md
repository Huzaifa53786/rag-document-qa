🚀 Just built an AI-powered Document Question Answering system, and here's what I learned.

Instead of scrolling through 50-page PDFs, you can now ask questions in plain English and get answers with sources.

How it works (RAG):
📄 Documents are split into chunks
🔢 Each chunk is converted to embeddings
🗄️ Stored in a vector database (ChromaDB)
🔍 Your question retrieves the most relevant chunks via semantic search
🤖 The LLM answers ONLY from that context

Key takeaway: retrieving context before generation noticeably reduces hallucinations. The model says "I don't know" instead of making things up.

Stack: Python • LangChain • Groq (open-source LLM) • ChromaDB • FastAPI • Streamlit • Docker

Next up: reranking, multi-user collections, and cloud deployment.

🔗 GitHub:(https://github.com/Huzaifa53786/rag-document-qa)

What would you build with RAG? 👇

#AI #MachineLearning #RAG #LangChain #Python #FastAPI #LLM #GenerativeAI

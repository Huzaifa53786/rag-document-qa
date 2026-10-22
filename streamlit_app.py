import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document QA (RAG)", page_icon="📄")
st.title("📄 Chat with your documents")

with st.sidebar:
    st.header("Upload")
    f = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if f and st.button("Index document"):
        with st.spinner("Indexing..."):
            try:
                r = requests.post(f"{API_URL}/upload", files={"file": (f.name, f.getvalue())}, timeout=120)
                if r.ok:
                    st.success(f"Indexed {r.json()['chunks_indexed']} chunks from {f.name}")
                else:
                    st.error(r.text)
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to API. Is the backend running on port 8000?")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 8)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(f"{API_URL}/ask", json={"question": q, "top_k": top_k}, timeout=60)
                if r.ok:
                    data = r.json()
                    text = data["answer"]
                    srcs = {(s["file"], s["page"]) for s in data["sources"]}
                    if srcs:
                        text += "\n\n**Sources:** " + ", ".join(
                            f"{fn}" + (f" (p.{pg})" if isinstance(pg, int) else "")
                            for fn, pg in sorted(srcs, key=str)
                        )
                else:
                    text = f"Error {r.status_code}: {r.text}"
            except requests.exceptions.ConnectionError:
                text = "⚠️ Cannot connect to API. Is the backend running?"
        st.markdown(text)
    st.session_state.messages.append({"role": "assistant", "content": text})

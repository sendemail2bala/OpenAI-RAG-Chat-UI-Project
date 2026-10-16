import os
from pathlib import Path
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document

load_dotenv()
st.set_page_config(page_title="OpenAI RAG Chat", page_icon="💬", layout="wide")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CHUNK_WORDS, OVERLAP, TOP_K = 180, 30, 4

def get_client():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return OpenAI(api_key=key) if key and key != "put_your_api_key_here" else None

def extract_file(f):
    ext = Path(f.name).suffix.lower()
    if ext == ".txt":
        return [(None, f.getvalue().decode("utf-8", errors="ignore"))]
    if ext == ".pdf":
        return [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(f).pages, 1)]
    if ext == ".docx":
        doc = Document(f)
        return [(None, "\n".join(p.text for p in doc.paragraphs))]
    return []

def split_text(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    return [" ".join(words[i:i+CHUNK_WORDS]) for i in range(0, len(words), step) if words[i:i+CHUNK_WORDS]]

def build_chunks(files):
    chunks = []
    for f in files:
        for page, text in extract_file(f):
            for part in split_text(text):
                chunks.append({"source": f.name, "page": page, "text": part})
    return chunks

def embeddings(client, texts, batch_size=64):
    vectors = []
    for i in range(0, len(texts), batch_size):
        res = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch_size])
        vectors.extend(x.embedding for x in res.data)
    return np.asarray(vectors, dtype=np.float32)

def build_index(client, chunks):
    vectors = embeddings(client, [c["text"] for c in chunks])
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx

def retrieve(client, question, chunks, index):
    q = embeddings(client, [question])
    faiss.normalize_L2(q)
    scores, ids = index.search(q, min(TOP_K, len(chunks)))
    return [{**chunks[int(i)], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]

def respond(client, question, history, contexts):
    context = "\n\n".join(
        f"[Source: {c['source']}" + (f", page {c['page']}" if c["page"] else "") + f"]\n{c['text']}"
        for c in contexts
    )
    recent = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]
    )
    prompt = f"""Answer using only the document context. Do not invent facts. If the answer is absent, say you could not find it in the uploaded documents. Cite filenames and PDF page numbers when possible.

DOCUMENT CONTEXT:
{context}

RECENT CHAT:
{recent}

QUESTION: {question}"""
    result = client.responses.create(model=CHAT_MODEL, input=prompt)
    return result.output_text

st.title("💬 OpenAI RAG Chat")
st.caption("Upload documents and chat with their contents.")
client = get_client()
if not client:
    st.error("OpenAI API key is not configured.")
    st.write("Copy `.env.example` to `.env`, add your API key, save it, and restart the app.")
    st.stop()

with st.sidebar:
    st.subheader("Models")
    st.write(f"Chat: `{CHAT_MODEL}`")
    st.write(f"Embeddings: `{EMBED_MODEL}`")
    st.warning("API calls may incur charges. Keep your API key private.")
    if st.button("Clear chat/index"):
        for k in ("chunks", "index", "signature", "messages"):
            st.session_state.pop(k, None)
        st.rerun()

files = st.file_uploader("Upload PDF, DOCX, or TXT documents", type=["pdf","docx","txt"], accept_multiple_files=True)
if files:
    signature = tuple((f.name, len(f.getvalue()), hash(f.getvalue())) for f in files)
    if st.session_state.get("signature") != signature:
        try:
            with st.spinner("Reading documents and creating embeddings with OpenAI..."):
                chunks = build_chunks(files)
                if not chunks:
                    st.error("No readable text found in these files.")
                    st.stop()
                index = build_index(client, chunks)
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["signature"] = signature
                st.session_state["messages"] = []
            st.success(f"Indexed {len(chunks)} text chunks.")
        except Exception as e:
            st.error(f"Document indexing failed: {e}")
            st.stop()
else:
    for k in ("chunks", "index", "signature"):
        st.session_state.pop(k, None)
    st.session_state.setdefault("messages", [])

if "chunks" not in st.session_state:
    st.info("Upload one or more documents to begin.")
    st.stop()

for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Retrieved sources"):
                for s in m["sources"]:
                    page = f", page {s['page']}" if s["page"] else ""
                    st.write(f"- {s['source']}{page} (similarity {s['score']:.3f})")

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state["messages"].append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating answer..."):
                contexts = retrieve(client, question, st.session_state["chunks"], st.session_state["index"])
                answer = respond(client, question, st.session_state["messages"][:-1], contexts)
            st.markdown(answer)
            with st.expander("Retrieved sources"):
                for s in contexts:
                    page = f", page {s['page']}" if s["page"] else ""
                    st.write(f"- {s['source']}{page} (similarity {s['score']:.3f})")
            st.session_state["messages"].append({"role":"assistant","content":answer,"sources":contexts})
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")
            st.caption("Check API key, API billing/credits, network, and model access.")
            st.session_state["messages"].pop()

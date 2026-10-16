# OpenAI RAG Chat UI

Streamlit chat app for uploaded PDF, DOCX, and TXT files. Uses OpenAI `text-embedding-3-small` embeddings, FAISS retrieval, and the OpenAI Responses API (`gpt-4o-mini` by default). Retrieved sources are displayed with PDF page numbers where available.

## Setup (Windows PowerShell)

1. Copy `.env.example` to `.env`.
2. Open `.env` and replace `put_your_api_key_here` with your OpenAI API key. Never share or commit this key.
3. Run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Open the local URL shown, usually `http://localhost:8501`.
5. Upload files and ask questions in the chat box.

You can also run `run_windows.bat`; it installs dependencies, creates `.env` if missing, opens it for editing, then you can run the batch file again.

## Notes
- OpenAI API usage is billed separately from ChatGPT subscription; ensure API billing/credits are active.
- The FAISS index and chat history live in session memory and are rebuilt after restart.
- Scanned/image-only PDFs need OCR first.
- This is a local development starter; add authentication and access controls before hosting publicly.

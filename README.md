# 📚 PDF RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about their contents — no external APIs, no cloud dependencies, no API keys required.

---

## How It Works

```
Upload PDFs → Extract Text → Chunk → Embed → FAISS Index → Question → Retrieve → LLM → Answer
```

1. PDFs are parsed page-by-page and split into overlapping word chunks
2. Each chunk is converted into a semantic vector using `all-MiniLM-L6-v2`
3. Normalized embeddings are stored in a local FAISS index (cosine similarity via inner product)
4. When you ask a question, the top-5 most relevant chunks are retrieved
5. FLAN-T5 generates a document-grounded answer using those chunks + chat history as context

---

## Features

- 📄 Upload and process multiple PDFs in one go
- 🔍 Semantic similarity search via FAISS (`IndexFlatIP`)
- 🧠 Fully local LLM inference with FLAN-T5 — no API keys required
- 💬 Persistent chat history with multi-turn context
- 📚 "View Retrieved Context" expander showing source file, page number, and similarity score
- 💾 Index persists to disk — no need to re-upload documents on restart
- 🗑️ One-click index and document clearing
- ⚡ Model caching with `@st.cache_resource` for fast reloads

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| Text Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (`faiss-cpu`) |
| Language Model | `google/flan-t5-base` (HuggingFace Transformers) |
| Numerical Processing | NumPy |

---

## Project Structure

```
├── app.py                   # All logic in one file — PDF processing, chunking,
│                            # embedding, FAISS index, LLM, and Streamlit UI
├── requirements.txt
├── faiss_store/             # Auto-created after first index build
│   ├── index.faiss          # Persisted FAISS vector index
│   └── meta.pkl             # Persisted chunk metadata
└── uploaded_documents/      # Auto-created when PDFs are uploaded
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- ~1–2 GB disk space for model downloads on first run

### Installation

**1. Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

---

## Usage

**Step 1 — Upload PDFs**
Use the sidebar file uploader to add one or more PDF documents.

**Step 2 — Build Index**
Click **🔨 Build Index**. The app will:
- Save uploaded files to `uploaded_documents/`
- Extract text page-by-page
- Split into overlapping word chunks
- Generate and normalize embeddings
- Store everything in a FAISS index on disk

**Step 3 — Ask Questions**
Type any question in the chat input at the bottom. The system retrieves the most relevant passages and generates a grounded answer.

**Step 4 — Inspect Sources**
Expand **🔎 View Retrieved Context** below any answer to see which file and page each chunk came from, along with its similarity score.

**Clearing the Index**
Click **🧹 Clear Index** in the sidebar to wipe the FAISS index, metadata, and all uploaded documents.

---

## Configuration

All parameters are defined at the top of `app.py` and can be changed there:

| Parameter | Variable | Default |
|---|---|---|
| Embedding model | `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| Language model | `LLM_MODEL` | `google/flan-t5-base` |
| Chunk size | `CHUNK_WORDS` | `500` words |
| Chunk overlap | `CHUNK_OVERLAP` | `80` words |
| Top-K retrieval | `TOP_K_DEFAULT` | `5` chunks |
| Max context length | `MAX_CONTEXT_CHARS` | `6000` characters |
| Max generated tokens | `MAX_NEW_TOKENS` | `600` |

---

## How the Index Works

Embeddings are L2-normalized before being added to a `faiss.IndexFlatIP` index. Because the vectors are unit-normalized, inner product search is mathematically equivalent to cosine similarity — giving semantically meaningful rankings without a dedicated cosine index.

The index and its metadata are saved to `faiss_store/` after every build, so the app reloads them automatically on restart without reprocessing your documents.

---

## Notes

- All processing and inference runs **locally** — your documents never leave your machine.
- FLAN-T5 is best for factual, extractive answers. For richer responses, swap `LLM_MODEL` for a larger model such as `google/flan-t5-large`.
- If a question cannot be answered from the retrieved context, the model is explicitly instructed to say so rather than hallucinate.
- Scanned PDFs (image-only) will not yield extractable text — use OCR-preprocessed PDFs for those.

# RAG-Based PDF Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and have a conversation with their contents — no external APIs, no cloud dependencies.

---

## How It Works

```
Upload PDFs → Extract Text → Chunk → Embed → FAISS Index → Question → Retrieve → LLM → Answer
```

1. PDFs are parsed and split into overlapping text chunks
2. Each chunk is converted into a semantic vector embedding
3. Embeddings are stored in a local FAISS vector database
4. When you ask a question, the most relevant chunks are retrieved
5. A language model generates a document-grounded answer using those chunks as context

---

## Features

- 📄 Upload and process multiple PDFs at once
- 🔍 Semantic similarity search via FAISS
- 🧠 Local LLM inference (FLAN-T5) — no API keys required
- 💬 Persistent chat history across the session
- 📚 "View Retrieved Context" expander showing source file, page, and relevance score
- 💾 Index persists to disk — survives app restarts
- 🗑️ One-click index clearing

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| Text Embeddings | Sentence Transformers |
| Vector Store | FAISS (faiss-cpu) |
| Language Model | FLAN-T5 (via HuggingFace Transformers) |
| Numerical Processing | NumPy |

---

## Project Structure

```
├── app.py                  # Streamlit app entry point
├── pdf_processor.py        # PDF reading and text extraction
├── chunker.py              # Text splitting with overlap
├── embedder.py             # Sentence Transformer embedding model
├── vector_store.py         # FAISS index wrapper (save/load/search/clear)
├── retriever.py            # Similarity search pipeline
├── llm.py                  # FLAN-T5 prompt construction and generation
└── faiss_index/            # Persisted index and metadata (auto-created)
```

> File names may vary depending on your implementation. Adjust accordingly.

---

## Getting Started

### Prerequisites
- Python 3.8+
- ~2–4 GB disk space for model downloads (first run only)

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
pip install streamlit sentence-transformers transformers faiss-cpu pypdf2 numpy
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Usage

**Step 1 — Upload PDFs**
Use the sidebar file uploader to add one or more PDF documents.

**Step 2 — Process Documents**
Click the process button. The app extracts text, chunks it, generates embeddings, and indexes everything into FAISS. A progress indicator shows status.

**Step 3 — Ask Questions**
Type any question in the chat input. The system retrieves the most relevant passages and generates a grounded answer.

**Step 4 — Inspect Sources**
Expand the "View Retrieved Context" section below any answer to see exactly which pages and documents informed the response, along with similarity scores.

---

## Configuration

Key parameters you can tune in the source code:

| Parameter | Description | Default |
|---|---|---|
| `chunk_size` | Number of characters per chunk | ~500 |
| `chunk_overlap` | Overlap between adjacent chunks | ~50 |
| `top_k` | Number of chunks retrieved per query | 3–5 |
| Embedding model | Sentence Transformers model name | `all-MiniLM-L6-v2` |
| LLM | HuggingFace model name | `google/flan-t5-base` |

---

## System Architecture

### PDF Processing
Pages are read individually. Only pages containing valid text are retained. Each page produces a structured record with `filename`, `page number`, and `text`.

### Chunking
Pages are split into smaller overlapping segments to improve retrieval precision. Overlapping ensures that context spanning chunk boundaries is not lost.

### Embeddings
Sentence Transformers converts each chunk into a dense vector. Vectors are L2-normalized so that FAISS inner product search is equivalent to cosine similarity.

### FAISS Index
A custom wrapper around FAISS handles storing embeddings alongside metadata, persisting the index to disk, reloading it on startup, and clearing it on demand.

### Prompt Construction
Each LLM call receives a structured prompt containing:
1. Recent chat history
2. Retrieved context chunks (with source attribution)
3. The user's current question

If no relevant context is found, the model is instructed to say so rather than hallucinate.

---

## Notes

- All processing and inference runs **locally** — your documents never leave your machine.
- The FAISS index is saved to disk after processing, so you don't need to re-upload documents on every restart.
- FLAN-T5 is best suited for factual, extractive answers. For more conversational output, you can swap in a larger model (e.g., `flan-t5-large` or an Ollama-served LLM).

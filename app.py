import os
import pickle
import faiss
import numpy as np
import streamlit as st
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import shutil

# ===============================
# Configuration
# ===============================
INDEX_DIR = "faiss_store"
UPLOAD_DIR = "uploaded_documents"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

CHUNK_WORDS = 500
CHUNK_OVERLAP = 80
TOP_K_DEFAULT = 5
MAX_CONTEXT_CHARS = 6000
MAX_NEW_TOKENS = 600

# ===============================
# Helper Functions
# ===============================
def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_uploaded_file(uploaded_file, directory):
    """Save uploaded file to the specified directory."""
    ensure_dir(directory)
    file_path = os.path.join(directory, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract page-wise text records from a PDF file."""
    try:
        reader = PdfReader(file_path)
        records = []
        filename = os.path.basename(file_path)
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append({
                    "filename": filename,
                    "page": page_num,
                    "text": text
                })
        return records
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return []

def chunk_records(records: List[Dict[str, Any]], chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split each page record into overlapping word chunks."""
    chunks = []
    step = max(1, chunk_words - overlap)
    
    for rec in records:
        words = rec["text"].split()
        for i in range(0, len(words), step):
            part = " ".join(words[i:i + chunk_words]).strip()
            if part:
                chunks.append({
                    "filename": rec["filename"],
                    "page": rec["page"],
                    "text": part,
                    "chunk_id": f"{rec['filename']}_p{rec['page']}_c{len(chunks)+1}"
                })
    return chunks

def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize vectors so inner-product = cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

class FaissCosineIndex:
    """FAISS index with cosine similarity and metadata persistence."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product for cosine similarity
        self.metadata: List[Dict[str, Any]] = []
        self._load()

    def _save(self):
        """Save index and metadata to disk."""
        ensure_dir(INDEX_DIR)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self):
        """Load existing index and metadata from disk."""
        try:
            if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
                st.sidebar.success(f"Loaded existing index with {self.index.ntotal} vectors")
        except Exception as e:
            st.sidebar.warning(f"Could not load existing index: {str(e)}")

    def add(self, vecs: np.ndarray, metas: List[Dict[str, Any]]):
        """Add vectors and metadata to index."""
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"Embedding shape mismatch, expected (*,{self.dim}) got {vecs.shape}")
        
        self.index.add(vecs)
        self.metadata.extend(metas)
        self._save()

    def search(self, qvec: np.ndarray, top_k: int = 5):
        """Search for similar vectors in the index."""
        if self.index.ntotal == 0:
            return []
        
        D, I = self.index.search(qvec, top_k)
        # Return list of (index, similarity_score) tuples
        return [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i != -1 and 0 <= i < len(self.metadata)]

    def clear(self):
        """Clear the index and remove persisted files."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        
        # Remove persisted files to avoid mismatch
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        
        # Remove uploaded documents
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        
        ensure_dir(UPLOAD_DIR)  # Recreate empty directory

# ===============================
# App Setup
# ===============================
st.set_page_config(
    page_title="PDF RAG (FAISS + FLAN-T5)", 
    layout="wide",
    page_icon="📚"
)

st.title("📚 PDF Chatbot")
st.caption("Upload PDFs → Build index → Ask questions grounded in your documents.")

# Initialize models with caching
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def get_llm():
    return pipeline("text2text-generation", model=LLM_MODEL)

# Load models
with st.spinner("Loading embedding model..."):
    embedder = get_embedder()
    
with st.spinner("Loading language model..."):
    llm = get_llm()

dim = embedder.get_sentence_embedding_dimension()
index = FaissCosineIndex(dim=dim)

# Ensure upload directory exists
ensure_dir(UPLOAD_DIR)

# ===============================
# Sidebar Controls
# ===============================
with st.sidebar:
    st.header("🧰 Controls")
    
    # File upload
    files = st.file_uploader(
        "Upload PDF(s)", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PDF documents to build the knowledge base"
    )
    
    # Action buttons
    col_a, col_b = st.columns(2)
    with col_a:
        build_btn = st.button("🔨 Build Index", use_container_width=True)
    with col_b:
        clear_btn = st.button("🧹 Clear Index", use_container_width=True)
    
    st.divider()
    
    # Index statistics
    st.write("### Index Statistics")
    st.write(f"Indexed vectors: **{index.index.ntotal}**")
    st.write(f"Metadata entries: **{len(index.metadata)}**")
    
    # Show uploaded documents
    if os.path.exists(UPLOAD_DIR) and os.listdir(UPLOAD_DIR):
        st.write("### Uploaded Documents")
        for doc in os.listdir(UPLOAD_DIR):
            st.write(f"📄 {doc}")
    
    if index.index.ntotal > 0:
        st.success("Index is ready for queries!")
    else:
        st.warning("No documents indexed yet.")

# Handle clear index button
if clear_btn:
    index.clear()
    st.sidebar.success("Index and documents cleared successfully!")
    st.rerun()

# Handle build index button
if build_btn and files:
    all_chunks = []
    saved_files = []
    
    with st.status("Processing PDFs...", expanded=True) as status:
        # Save uploaded files
        st.write("Saving uploaded files...")
        for f in files:
            file_path = save_uploaded_file(f, UPLOAD_DIR)
            saved_files.append(file_path)
            st.write(f"✓ Saved: {f.name}")
        
        # Process each saved file
        st.write("Extracting text from PDFs...")
        for file_path in saved_files:
            records = extract_text_from_pdf(file_path)
            if records:
                chunks = chunk_records(records)
                all_chunks.extend(chunks)
                st.write(f"✓ {os.path.basename(file_path)}: {len(records)} pages → {len(chunks)} chunks")
            else:
                st.write(f"✗ {os.path.basename(file_path)}: No text extracted")
        
        if not all_chunks:
            st.error("No extractable text found in the uploaded PDFs.")
        else:
            st.write("Generating embeddings...")
            texts = [c["text"] for c in all_chunks]
            
            # Create embeddings
            vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            vecs = normalize(vecs)  # Critical for cosine search
            
            st.write("Adding to vector database...")
            index.add(vecs, all_chunks)
            
            status.update(label="Indexing complete!", state="complete")
            st.success(f"Indexed {len(all_chunks)} text chunks from {len(files)} PDF(s)")
            st.toast("Index saved to disk.", icon="💾")

st.markdown("---")

# ===============================
# Chat Interface (NEW DESIGN)
# ===============================
st.subheader("💬 Chat with your documents")

# We need to change the session state key to 'messages' for the chat UI
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input using the dedicated chat input field
if user_question := st.chat_input("Ask a question about your documents..."):
    
    # Add user's message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Check if the index is ready before proceeding
    if index.index.ntotal == 0:
        with st.chat_message("assistant"):
            response = "I can't answer questions yet. Please upload one or more PDFs and click '🔨 Build Index' in the sidebar first."
            st.warning(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Start processing the query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Embed the user's question
                qvec = embedder.encode([user_question], convert_to_numpy=True)
                qvec = normalize(qvec)

                # Retrieve context
                hits = index.search(qvec, top_k=TOP_K_DEFAULT)
                retrieved = [index.metadata[i] | {"score": score} for i, score in hits]

                if not retrieved:
                    response = "I couldn't find any relevant information in your documents to answer that question."
                    st.warning(response)
                else:
                    # Build the conversational prompt with history
                    context_text = "\n\n".join(r["text"] for r in retrieved)
                    if len(context_text) > MAX_CONTEXT_CHARS:
                        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[Truncated...]"

                    # Format previous messages for the prompt's chat history
                    chat_history_str = ""
                    # We use the new 'messages' session state, taking the last 3 pairs (user+assistant)
                    for msg in st.session_state.messages[-7:-1]: # up to the last user message
                        if msg["role"] == "user":
                           chat_history_str += f"User: {msg['content']}\n"
                        else:
                           chat_history_str += f"Assistant: {msg['content']}\n"


                    prompt = (
                        "You are a helpful assistant. Answer the user's 'Current Question' using the 'Chat History' for context and the 'Retrieved Context' for information.\n"
                        "If the answer is not in the 'Retrieved Context', say you cannot find it in the documents.\n\n"
                        "--- CHAT HISTORY ---\n"
                        f"{chat_history_str}"
                        "--- RETRIEVED CONTEXT ---\n"
                        f"{context_text}\n\n"
                        "--- CURRENT QUESTION ---\n"
                        f"{user_question}\n"
                        "Answer:"
                    )

                    # Generate the answer from the LLM
                    try:
                        out = llm(prompt, max_new_tokens=MAX_NEW_TOKENS)
                        response = out[0]["generated_text"].strip()
                    except Exception as e:
                        response = f"Sorry, I ran into an error: {str(e)}"
                        st.error(response)

                # Display the response and the retrieved context in an expander
                st.markdown(response)
                with st.expander("🔎 View Retrieved Context"):
                    for i, r in enumerate(retrieved):
                        st.markdown(f"**{i+1}. {r['filename']}** (p.{r['page']}) · similarity: `{r['score']:.3f}`")
                        st.caption(r["text"])
                        st.markdown("---")

        # Add the assistant's final response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


from pathlib import Path
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Setup directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR = "brahvi_faiss_store"

# Load multilingual embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased")

# Function to process PDF and store as FAISS index
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        texts = [Document(page_content=page.page_content) for page in pages]

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(texts)

        # Store in FAISS vector DB
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(VECTOR_STORE_DIR)
        return len(chunks)
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return 0

# Function to query the FAISS knowledge base
def query_model(question):
    try:
        db = FAISS.load_local(VECTOR_STORE_DIR, embedding_model)
        results = db.similarity_search(question, k=3)
        return "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        st.error(f"❌ Error querying model: {e}")
        return ""

# Streamlit UI
st.set_page_config(page_title="📚 Brahui Learning App", layout="centered")
st.title("📘 Brahui Book Learner")
st.write("Upload your Brahui language books (PDFs) and ask questions in any language.")

# Upload Section
uploaded_file = st.file_uploader("📤 Upload a Brahui Book (PDF)", type=["pdf"])
if uploaded_file:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("🔄 Processing uploaded book..."):
        chunk_count = process_pdf(file_path)
    if chunk_count > 0:
        st.success(f"✅ Successfully stored {chunk_count} chunks in the knowledge base.")

# Question Answer Section
question = st.text_input("💬 Ask a question:")
if st.button("Ask") and question:
    with st.spinner("🧠 Thinking..."):
        answer = query_model(question)
    st.markdown("### 📖 Answer:")
    st.write(answer if answer else "❌ No relevant information found.")

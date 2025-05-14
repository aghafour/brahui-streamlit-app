
from pathlib import Path
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st

# Setup directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR = "brahvi_chroma"

# Load embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased")

# Process PDF and store vectors
def process_pdf(file_path):
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(pages)
    db = Chroma.from_documents(texts, embedding_model, persist_directory=VECTOR_STORE_DIR)
    db.persist()
    return len(texts)

# Query the vector store
def query_model(question):
    db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
    results = db.similarity_search(question, k=3)
    return "\n\n---\n\n".join([doc.page_content for doc in results])

# Streamlit UI
st.set_page_config(page_title="Brahui Book Learner", layout="centered")
st.title("📚 Brahui Learning Platform")
st.write("Upload your Brahui translation or grammar books (PDFs) and ask questions.")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a Brahui Book (PDF)", type=["pdf"])
if uploaded_file:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
    with st.spinner("🔄 Processing..."):
        chunks = process_pdf(file_path)
    st.success(f"✅ Added {chunks} chunks to the knowledge base.")

# Ask question
question = st.text_input("💬 Ask a question:")
if st.button("Ask") and question:
    with st.spinner("🧠 Thinking..."):
        answer = query_model(question)
    st.markdown("### 📖 Answer:")
    st.write(answer if answer else "❌ No relevant information found.")

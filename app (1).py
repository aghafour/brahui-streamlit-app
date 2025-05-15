from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR = Path("brahvi_faiss_store")

# Embedding model initialization
embedding_model = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Function to process PDF and store vectors
def process_pdf(file_path: Path) -> int:
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        # Convert pages into Documents
        documents = [Document(page_content=page.page_content) for page in pages]

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Create or overwrite FAISS vector store
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(str(VECTOR_STORE_DIR))

        return len(chunks)
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return 0

# Function to answer questions using vector store
def query_model(question: str) -> str:
    try:
        if not VECTOR_STORE_DIR.exists():
            st.warning("⚠️ No vector store found. Please upload and process a PDF first.")
            return ""

        db = FAISS.load_local(str(VECTOR_STORE_DIR), embedding_model)
        results = db.similarity_search(question, k=3)

        if not results:
            return "❌ No relevant information found."

        return "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        st.error(f"❌ Error querying model: {e}")
        return ""

# Streamlit UI setup
st.set_page_config(page_title="📘 Brahui Book Learner", layout="centered")
st.title("📚 Brahui Language Learning App")
st.write("Upload a Brahui PDF book and ask questions in any language.")

# Upload Section
uploaded_file = st.file_uploader("📤 Upload a Brahui Book (PDF)", type=["pdf"])
if uploaded_file:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("🔄 Processing PDF..."):
        chunk_count = process_pdf(file_path)
    if chunk_count > 0:
        st.success(f"✅ Stored {chunk_count} chunks successfully!")

# QA Section
question = st.text_input("💬 Ask a question:")
if st.button("Ask") and question.strip():
    with st.spinner("🧠 Searching..."):
        answer = query_model(question.strip())
    st.markdown("### 📖 Answer:")
    st.write(answer)

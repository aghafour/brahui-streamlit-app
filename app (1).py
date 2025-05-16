from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR = "brahvi_faiss_store"

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Function to process PDF and store embeddings
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        texts = [Document(page_content=page.page_content) for page in pages]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(texts)

        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(VECTOR_STORE_DIR)

        return len(chunks)
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return 0

# Function to query the model
def query_model(question):
    try:
        if not Path(VECTOR_STORE_DIR).exists():
            st.warning("⚠️ Please upload and process a PDF first.")
            return ""

        db = FAISS.load_local(VECTOR_STORE_DIR, embedding_model)
        results = db.similarity_search(question, k=3)

        if not results:
            return "❌ No relevant answer found."
        return "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        st.error(f"❌ Error querying model: {e}")
        return ""

# Streamlit UI setup
st.set_page_config(page_title="📘 Brahui Book Learner", layout="centered")
st.title("📚 Brahui Language Learning App")
st.write("Upload a Brahui PDF book and ask questions in any language.")

# Upload PDF section
uploaded_file = st.file_uploader("📤 Upload a Brahui Book (PDF)", type=["pdf"])
if uploaded_file:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("🔄 Processing PDF..."):
        chunks = process_pdf(file_path)
    if chunks > 0:
        st.success(f"✅ Stored {chunks} chunks successfully!")

# Ask a question section
st.markdown("---")
st.subheader("💬 Ask a Question From the Uploaded Book")
question = st.text_input("Type your question here:")
if st.button("Ask") and question:
    with st.spinner("🧠 Thinking..."):
        answer = query_model(question)
    st.markdown("### 📖 Answer:")
    st.write(answer)

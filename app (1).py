from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR = Path("brahvi_faiss_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Function to process PDF and append to vector store
def process_pdf(file_path):
    try:
        loader = UnstructuredPDFLoader(str(file_path))
        pages = loader.load()
        texts = [Document(page_content=page.page_content) for page in pages if "Scanned by CamScanner" not in page.page_content]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(texts)

        # Load or create vector store
        if (VECTOR_STORE_DIR / "index.faiss").exists():
            db = FAISS.load_local(VECTOR_STORE_DIR, embedding_model, allow_dangerous_deserialization=True)
            db.add_documents(chunks)
        else:
            db = FAISS.from_documents(chunks, embedding_model)

        db.save_local(VECTOR_STORE_DIR)

        return len(chunks)
    except Exception as e:
        st.error(f"\u274c Error processing PDF: {e}")
        return 0

# Function to query the vector store
def query_model(question):
    try:
        if not (VECTOR_STORE_DIR / "index.faiss").exists():
            st.warning("\u26a0\ufe0f Please upload and process at least one PDF first.")
            return ""

        db = FAISS.load_local(
            VECTOR_STORE_DIR,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        results = db.similarity_search(question, k=3)

        if not results:
            return "\u274c No relevant answer found."
        return "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        st.error(f"\u274c Error querying model: {e}")
        return ""

# Streamlit UI
st.set_page_config(page_title="\ud83d\udcd8 Brahui Knowledge QA", layout="centered")
st.title("\ud83d\udcda Brahui Language Learning App")
st.write("Upload multiple Brahui books (e.g., grammar, translation) and ask questions in Arabic or Roman script.")

# Upload PDF Section
uploaded_file = st.file_uploader("\ud83d\udcc4 Upload Brahui Book (PDF)", type=["pdf"])
if uploaded_file:
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner("\ud83d\udd04 Processing PDF and updating knowledge base..."):
        chunks = process_pdf(file_path)
    if chunks > 0:
        st.success(f"\u2705 Added {chunks} new chunks to the knowledge base!")

# Ask a Question
st.subheader("\u2753 Ask a Question")
user_question = st.text_input("Type your question (in Arabic or Roman Brahui):")
if st.button("Ask"):
    if user_question.strip():
        with st.spinner("\ud83d\udd0d Retrieving answer..."):
            answer = query_model(user_question)
        st.markdown(f"### \ud83d\udcdd Answer:\n{answer}")
    else:
        st.warning("Please type a question to ask.")

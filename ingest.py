import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def ingest_pdf(pdf_path, db_path="./chroma_db"):
    print(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks.")

    # Embeddings & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"Vector database created at {db_path}")
    return vector_db

if __name__ == "__main__":
    # Use the path provided by the user
    PDF_PATH = "/Users/shashikalasingh/Downloads/Shashikala Singh - ATS Resume.pdf"
    ingest_pdf(PDF_PATH)

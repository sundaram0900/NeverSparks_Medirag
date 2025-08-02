

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # <- this stays in langchain



def ingest_documents(pdf_folder: str = "docs", persist_directory: str = "db"):
    """
    Ingests PDF documents from the specified folder, splits them into chunks,
    and stores their vector embeddings using Chroma.

    Args:
        pdf_folder (str): Folder containing PDF files.
        persist_directory (str): Directory where Chroma DB will be persisted.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_docs = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    for file in pdf_files:
        file_path = os.path.join(pdf_folder, file)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()

    print(f"[✔] Ingested {len(all_docs)} chunks from {len(pdf_files)} PDFs.")


if __name__ == "__main__":
    ingest_documents()

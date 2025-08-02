# ragpipe.py

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_FILE = "data.pdf"  # Make sure this is in your repo
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def load_retriever():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter

    loader = PyPDFLoader("data.pdf")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ❗ Completely in-memory Chroma setup (no persist_directory)
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="medirag",      # any name is fine
        client_settings={
            "chroma_db_impl": "duckdb+parquet",
            "persist_directory": None,   # don't persist
            "anonymized_telemetry": False,
        }
    )

    return vectordb.as_retriever()


def load_llm():
    """
    Load an LLM from Hugging Face Hub via API (cloud-compatible).
    """
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",  # You can try other models like mistralai/Mixtral
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

def build_qa_chain():
    """
    Build the RetrievalQA chain from LLM and retriever.
    """
    retriever = load_retriever()
    llm = load_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa

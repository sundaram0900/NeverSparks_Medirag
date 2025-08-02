# ragpipe.py

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS  # ✅ Replaces Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Configuration
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_FILE = "data.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def load_retriever():
    """
    Load the vector database retriever using HuggingFace embeddings and FAISS (in-memory).
    """
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # ✅ FAISS is fully in-memory and cloud-safe
    vectordb = FAISS.from_documents(split_docs, embedding=embeddings)
    return vectordb.as_retriever()

from langchain_community.llms import HuggingFaceHub

def load_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",       # ✅ Make sure this model is hosted and supports text generation
        model_kwargs={"temperature": 0.3, "max_length": 512},
        task="text2text-generation"          # ✅ REQUIRED: Hugging Face inference task
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

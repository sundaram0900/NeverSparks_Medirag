# ragpipe.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import os

# Configuration
DB_DIR = "db"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_FILE = os.path.join("models", "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf")  # 💡 Replace with actual downloaded filename

def load_retriever():
    """
    Load the vector database retriever using HuggingFace embeddings and Chroma.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    return vectordb.as_retriever()

def load_llm():
    """
    Load a local GGUF quantized LLM using CTransformers (TinyLLaMA model).
    """
    return CTransformers(
        model=MODEL_FILE,
        model_type="llama",  # ✅ TinyLLaMA uses "llama" type
        config={
            'max_new_tokens': 512,
            'temperature': 0.3,
            'context_length': 2048
        },
        local_files_only=True
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

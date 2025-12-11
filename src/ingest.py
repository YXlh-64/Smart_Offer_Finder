import sys
from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.ollama import OllamaEmbeddings
from pinecone import Pinecone

from .config import get_settings

load_dotenv()


def choose_embeddings(settings):
    """Use Ollama embeddings (qllama/multilingual-e5-base, local, no API needed)."""
    return OllamaEmbeddings(
        model=settings.embedding_model.replace("ollama/", ""),
        base_url=settings.ollama_base_url
    )


def load_documents(data_dir: Path) -> List:
    pdf_files = sorted(data_dir.glob("*.pdf")) # to be updated to match json
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}. Add data then rerun.")

    docs = []
    for path in pdf_files:
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())
    return docs


def build_vectorstore():
    settings = get_settings()
    data_dir = Path("data/raw")

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required. Set it in .env file.")

    docs = load_documents(data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = choose_embeddings(settings)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    
    # Create vector store and add documents
    vectorstore = PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=settings.pinecone_index_name
    )
    
    print(f"Successfully indexed {len(chunks)} chunks to Pinecone index '{settings.pinecone_index_name}'.")


if __name__ == "__main__":
    try:
        build_vectorstore()
    except Exception as exc:  # pragma: no cover
        print(f"[ingest] failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import sys
import argparse
from pathlib import Path
from typing import List
import os
import json

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma
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


def load_documents_from_json(json_path: Path) -> List[Document]:
    """Load documents from JSON file with metadata and content."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    for item in data:
        # Create a Document with content and metadata
        doc = Document(
            page_content=item.get('content', ''),
            metadata=item.get('metadata', {})
        )
        docs.append(doc)
    
    print(f"Loaded {len(docs)} documents from {json_path.name}")
    return docs


def build_vectorstore_pinecone(chunks: List[Document], embeddings, settings):
    """Build and index documents to Pinecone."""
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required. Set it in .env file.")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    
    # Create vector store and add documents
    vectorstore = PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=settings.pinecone_index_name
    )
    
    print(f"✅ Successfully indexed {len(chunks)} chunks to Pinecone index '{settings.pinecone_index_name}'.")
    return vectorstore


def build_vectorstore_chromadb(chunks: List[Document], embeddings, settings):
    """Build and index documents to ChromaDB (local)."""
    persist_directory = settings.vectorstore_path
    
    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="smart-offer-finder"
    )
    
    print(f"✅ Successfully indexed {len(chunks)} chunks to ChromaDB at '{persist_directory}'.")
    return vectorstore


def build_vectorstore(db_type: str = "pinecone"):
    """Build vector store and index documents.
    
    Args:
        db_type: Database type - "pinecone" or "chromadb"
    """
    settings = get_settings()
    data_dir = Path("data/raw")
    json_path = data_dir / "metadata_version.json"

    # Load data_sample.json
    print(f"Loading {json_path.name}...")
    all_docs = load_documents_from_json(json_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = choose_embeddings(settings)
    
    # Build vector store based on db_type
    db_type = db_type.lower()
    if db_type == "pinecone":
        print("Using Pinecone vector database...")
        vectorstore = build_vectorstore_pinecone(chunks, embeddings, settings)
    elif db_type == "chromadb":
        print("Using ChromaDB vector database...")
        vectorstore = build_vectorstore_chromadb(chunks, embeddings, settings)
    else:
        raise ValueError(f"Unknown database type: {db_type}. Choose 'pinecone' or 'chromadb'")
    
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector database"
    )
    parser.add_argument(
        "--db",
        choices=["pinecone", "chromadb"],
        default="pinecone",
        help="Vector database to use (default: pinecone)"
    )
    
    args = parser.parse_args()
    
    try:
        build_vectorstore(db_type=args.db)
        print(f"\nIngestion completed successfully using {args.db}!")
    except Exception as exc:  # pragma: no cover
        print(f"[ingest] failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

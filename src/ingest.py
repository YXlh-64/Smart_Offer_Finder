import os
# Disable ChromaDB telemetry BEFORE any chromadb imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
import chromadb
from chromadb.config import Settings as ChromaSettings

# Assuming your config file is in the same package structure
from .config import get_settings

load_dotenv()

# --- CONFIGURATION ---
# Path to the JSON file you generated in Phase 1
JSON_DATA_PATH = Path("data/raw/data.json") 


def choose_embeddings(settings):
    """
    Use Ollama embeddings (e.g., intfloat/multilingual-e5-large).
    Ensures the model name matches your config.
    """
    model_name = settings.embedding_model.replace("ollama/", "")
    return OllamaEmbeddings(
        model=model_name,
        base_url=settings.ollama_base_url
    )


def load_knowledge_base_json(file_path: Path) -> List[Document]:
    """
    Loads the pre-processed JSON file containing Markdown tables and metadata.
    Converts them into LangChain Document objects.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"❌ JSON file not found at: {file_path}. Please run your processing script first.")

    print(f"📂 Loading data from {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        # 1. Extract Content
        content = entry.get("content", "")
        
        # 2. Clean Metadata
        # Filter out None/Null values in metadata
        raw_meta = entry.get("metadata", {})
        clean_meta = {k: v for k, v in raw_meta.items() if v is not None}
        
        # 3. Create Document
        # We store the 'source' explicitly for citations later
        if "source_filename" in clean_meta:
            clean_meta["source"] = clean_meta["source_filename"]

        docs.append(Document(
            page_content=content,
            metadata=clean_meta
        ))
        
    print(f"   Converted {len(docs)} JSON entries into Documents.")
    return docs


def build_vectorstore():
    settings = get_settings()

    # 1. Load the JSON Data
    # Ensure the path matches where you saved your file
    docs = load_knowledge_base_json(JSON_DATA_PATH)

    # 2. Split Text
    # We use a larger chunk size because we want to keep Markdown tables intact.
    # Splitting by "\n\n" is prioritized to keep paragraphs/tables together.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Large enough for a pricing table
        chunk_overlap=150,        # Context overlap
        separators=["\n\n", "\n", " ", ""], # Try to split by double newline first
        keep_separator=False
    )
    
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} text chunks.")

    # 3. Initialize Embeddings
    print(f"⏳ Initializing Embeddings ({settings.embedding_model})...")
    embeddings = choose_embeddings(settings)
    
    # 4. Create ChromaDB Vectorstore
    print(f"🚀 Creating ChromaDB collection: '{settings.chroma_collection_name}'...")
    print(f"📁 Persist directory: {settings.chroma_persist_directory}")
    
    # Create ChromaDB client with telemetry disabled
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # from_documents automatically embeds and stores in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        client=chroma_client
    )
    
    print(f"✅ Success! Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    try:
        build_vectorstore()
    except Exception as exc:
        print(f"❌ [ingest] Failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
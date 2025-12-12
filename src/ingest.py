import os
import json
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import get_settings

load_dotenv()

# Path to your data
JSON_DATA_PATH = Path("data/raw/data.json")

def build_vectorstore():
    settings = get_settings()
    
    # 1. Load Data
    print(f"📂 Loading data from {JSON_DATA_PATH}...")
    if not JSON_DATA_PATH.exists():
        print("❌ File not found.")
        return

    with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        content = entry.get("content", "")
        
        # --- CRITICAL FIX 1: Remove None values from Metadata ---
        # ChromaDB crashes if metadata contains None/Null. We filter them out here.
        raw_metadata = entry.get("metadata", {})
        metadata = {k: v for k, v in raw_metadata.items() if v is not None}
        
        # Store original text in metadata for display in the UI
        metadata["original_text"] = content
        
        # Ensure 'source' exists for citations
        if "source_filename" in metadata:
            metadata["source"] = metadata["source_filename"]
        
        # --- CRITICAL FIX 2: Add metadata prefix for better retrieval ---
        # Include key metadata in the content that gets embedded
        # This helps with retrieval of specific establishments, document types, etc.
        metadata_prefix = []
        
        # Add document title if available
        if "document_title" in metadata and metadata["document_title"]:
            metadata_prefix.append(f"Titre: {metadata['document_title']}")
        
        # Add source filename if available
        if "source_filename" in metadata and metadata["source_filename"]:
            metadata_prefix.append(f"Source: {metadata['source_filename']}")
        
        # Add document type if available
        if "document_type" in metadata and metadata["document_type"]:
            metadata_prefix.append(f"Type: {metadata['document_type']}")
        
        # Combine metadata prefix with content
        if metadata_prefix:
            metadata_text = " | ".join(metadata_prefix)
            enriched_content = f"{metadata_text}\n\n{content}"
        else:
            enriched_content = content
        
        # --- CRITICAL FIX 3: Add "passage: " prefix for E5 model ---
        # The E5 model requires this prefix to understand the text is a document to be searched
        model_content = f"passage: {enriched_content}"

        docs.append(Document(page_content=model_content, metadata=metadata))

    print(f"   Processed {len(docs)} documents.")

    # 2. Split Text
    # Use chunk size from settings (configurable via .env)
    print(f"📝 Using chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, 
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} text chunks.")

    # 3. Initialize Fast GPU Embeddings
    print("⏳ Initializing GPU Embeddings (multilingual-e5-base)...")
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = SentenceTransformerEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Create ChromaDB
    print(f"🚀 Indexing to {settings.chroma_persist_directory}...")
    
    # Ensure fresh start: Clear old DB if needed (Optional, but safer if schema changed)
    # import shutil
    # if os.path.exists(settings.chroma_persist_directory):
    #     shutil.rmtree(settings.chroma_persist_directory)

    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.chroma_collection_name,
        client=chroma_client
    )
    print("✅ Indexing Complete! You can now run chat.py")

if __name__ == "__main__":
    build_vectorstore()
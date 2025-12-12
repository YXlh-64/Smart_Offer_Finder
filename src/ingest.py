import os
import json
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
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
        
        # --- CRITICAL FIX 1: Add "passage: " prefix ---
        # The E5 model requires this prefix to understand the text is a document to be searched
        clean_content = content 
        model_content = f"passage: {content}" 
        
        # --- CRITICAL FIX 2: Remove None values from Metadata ---
        # ChromaDB crashes if metadata contains None/Null. We filter them out here.
        raw_metadata = entry.get("metadata", {})
        metadata = {k: v for k, v in raw_metadata.items() if v is not None}
        
        # Store original text in metadata for display in the UI
        metadata["original_text"] = clean_content
        
        # Ensure 'source' exists for citations
        if "source_filename" in metadata:
            metadata["source"] = metadata["source_filename"]

        docs.append(Document(page_content=model_content, metadata=metadata))

    print(f"   Processed {len(docs)} documents.")

    # 2. Split Text
    # We use a large chunk size (1200) to keep pricing tables intact
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Split into {len(chunks)} text chunks.")

    # 3. Initialize Fast GPU Embeddings
    print("⏳ Initializing GPU Embeddings (multilingual-e5-base)...")
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
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
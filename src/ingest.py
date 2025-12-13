import os
import json
import sys
import re
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

# Define Paths for both data sources
DATA_PATH_1 = Path("data/raw/data.json")
DATA_PATH_2 = Path("data/raw/data2.json")


def normalize_file_path(raw_path: str) -> str:
    """
    Normalize file paths for cross-platform compatibility.
    
    - Converts backslashes to forward slashes
    - Removes leading ./ or ./
    - Removes absolute drive letters (C:/, D:/)
    - Returns a clean relative path from project root
    """
    if not raw_path:
        return ""
    
    # Convert backslashes to forward slashes
    path = raw_path.replace("\\", "/")
    
    # Remove leading ./ or ./
    path = re.sub(r'^\.\/+', '', path)
    
    # Remove Windows drive letters (e.g., C:/, D:/)
    path = re.sub(r'^[A-Za-z]:\/+', '', path)
    
    # Remove any double slashes
    path = re.sub(r'\/+', '/', path)
    
    return path

def process_standard_data(file_path: Path) -> List[Document]:
    """
    Process data.json which has a standard 'content' and 'metadata' structure.
    """
    print(f"📂 Loading standard data from {file_path}...")
    if not file_path.exists():
        print(f"⚠️  File {file_path} not found. Skipping.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        content = entry.get("content", "")
        
        # Filter None values from metadata
        raw_metadata = entry.get("metadata", {})
        metadata = {k: v for k, v in raw_metadata.items() if v is not None}
        
        # Store original text
        metadata["original_text"] = content
        
        # Ensure 'source' exists
        if "source_filename" in metadata:
            metadata["source"] = metadata["source_filename"]
        else:
            metadata["source"] = file_path.name
        
        # Normalize and store file_path for download links
        raw_file_path = raw_metadata.get("file_path", "")
        if raw_file_path:
            metadata["file_path"] = normalize_file_path(raw_file_path)
        else:
            metadata["file_path"] = ""

        # Create metadata prefix for better retrieval context
        metadata_prefix = []
        if "document_title" in metadata and metadata["document_title"]:
            metadata_prefix.append(f"Titre: {metadata['document_title']}")
        if "source_filename" in metadata and metadata["source_filename"]:
            metadata_prefix.append(f"Source: {metadata['source_filename']}")
        
        if metadata_prefix:
            metadata_text = " | ".join(metadata_prefix)
            enriched_content = f"{metadata_text}\n\n{content}"
        else:
            enriched_content = content

        # Add "passage: " prefix for E5 model
        model_content = f"passage: {enriched_content}"
        
        docs.append(Document(page_content=model_content, metadata=metadata))
    
    print(f"   ✓ Processed {len(docs)} documents from {file_path.name}")
    return docs

def process_procedural_data(file_path: Path) -> List[Document]:
    """
    Process data2.json which has 'Process_Title' and 'Steps'.
    We format this into a readable guide for the LLM.
    """
    print(f"📂 Loading procedural data from {file_path}...")
    if not file_path.exists():
        print(f"⚠️  File {file_path} not found. Skipping.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        process_title = entry.get("Process_Title", "Untitled Process")
        steps_list = entry.get("Steps", [])
        
        # Format the steps into a readable string
        formatted_steps = []
        for idx, step_obj in enumerate(steps_list, 1):
            # Extract the value from the dictionary (e.g., {"step_1": "text"})
            step_text = list(step_obj.values())[0] if step_obj else ""
            formatted_steps.append(f"{idx}. {step_text}")
        
        steps_content = "\n".join(formatted_steps)
        
        # Create a rich content block
        # We explicitly label it as a Guide/Process for the LLM
        full_content = (
            f"Processus: {process_title}\n"
            f"Type: Guide Opérationnel\n"
            f"Étapes:\n{steps_content}"
        )
        
        # Get and normalize file path from entry
        raw_file_path = entry.get("path", "")
        normalized_path = normalize_file_path(raw_file_path) if raw_file_path else ""
        
        # Create Metadata
        metadata = {
            "source": file_path.name,
            "title": process_title,
            "type": "procedure",
            "original_text": full_content,
            "file_path": normalized_path
        }

        # Add E5 prefix
        model_content = f"passage: {full_content}"
        
        docs.append(Document(page_content=model_content, metadata=metadata))

    print(f"   ✓ Processed {len(docs)} documents from {file_path.name}")
    return docs

def build_vectorstore():
    settings = get_settings()
    
    # 1. Load and Combine Data
    all_docs = []
    
    # Load data.json
    all_docs.extend(process_standard_data(DATA_PATH_1))
    
    # Load data2.json
    all_docs.extend(process_procedural_data(DATA_PATH_2))

    if not all_docs:
        print("❌ No documents found to ingest.")
        return

    print(f"\n📦 Total documents to ingest: {len(all_docs)}")

    # 2. Split Text
    print(f"📝 Using chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, 
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"✂️  Split into {len(chunks)} text chunks.")

    # 3. Initialize Embeddings (use CUDA if available, otherwise CPU)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⏳ Initializing Embeddings (multilingual-e5-base) on {device}...")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = SentenceTransformerEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Create/Update ChromaDB
    print(f"🚀 Indexing to {settings.chroma_persist_directory}...")
    
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
    print("✅ Indexing Complete! Both datasets are now in the same database.")

if __name__ == "__main__":
    build_vectorstore()
"""
NGBSS Guide Ingestion Pipeline
Extracts procedural steps from NGBSS PDF guides using Gemini Flash Vision
and stores them in ChromaDB for RAG retrieval.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import io

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("⚠️ pdf2image not installed. Run: pip install pdf2image")

# Alternative: PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ChromaDB
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ngbss.gemini_vision import GeminiVision
from src.config import get_settings


class NGBSSIngestionPipeline:
    """Pipeline to ingest NGBSS PDF guides into ChromaDB."""
    
    def __init__(self, 
                 pdf_folder: str,
                 chroma_persist_dir: str = "data/chroma_db",
                 collection_name: str = "ngbss-guides"):
        """
        Initialize the ingestion pipeline.
        
        Args:
            pdf_folder: Path to folder containing NGBSS PDF files
            chroma_persist_dir: Path to ChromaDB persistence directory
            collection_name: Name of ChromaDB collection for NGBSS guides
        """
        self.pdf_folder = Path(pdf_folder)
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        
        # Initialize Gemini Vision
        print("🚀 Initializing Gemini Flash Vision...")
        self.vision = GeminiVision(model_name="gemini-2.5-flash")
        
        # Initialize embeddings
        print("🚀 Loading embeddings on CUDA...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        
    def _extract_procedure_name(self, filename: str) -> str:
        """Extract procedure name from PDF filename."""
        # Remove extension and common prefixes
        name = Path(filename).stem
        name = re.sub(r'^Guide\s*NGBSS\s*[-_]?\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*[-_]\s*(complément|complement|v\d+).*$', '', name, flags=re.IGNORECASE)
        return name.strip()
    
    def _pdf_to_images_pymupdf(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images using PyMuPDF (faster, no poppler needed)."""
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render at 2x resolution for better OCR
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        
        doc.close()
        return images
    
    def _pdf_to_images_pdf2image(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to images using pdf2image (requires poppler)."""
        return convert_from_path(pdf_path, dpi=200)
    
    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF to list of PIL Images."""
        if PYMUPDF_AVAILABLE:
            print(f"  📄 Converting with PyMuPDF: {pdf_path.name}")
            return self._pdf_to_images_pymupdf(pdf_path)
        elif PDF2IMAGE_AVAILABLE:
            print(f"  📄 Converting with pdf2image: {pdf_path.name}")
            return self._pdf_to_images_pdf2image(pdf_path)
        else:
            raise RuntimeError("No PDF library available. Install: pip install PyMuPDF or pip install pdf2image")
    
    def process_single_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Process a single NGBSS PDF file.
        
        Returns:
            List of structured documents ready for ChromaDB
        """
        procedure_name = self._extract_procedure_name(pdf_path.name)
        print(f"\n📁 Processing: {pdf_path.name}")
        print(f"   Procedure: {procedure_name}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        print(f"   Pages: {len(images)}")
        
        documents = []
        
        for step_num, image in enumerate(images, start=1):
            print(f"   🔍 Analyzing page {step_num}/{len(images)}...", end=" ", flush=True)
            
            # Extract information using Gemini Vision
            result = self.vision.extract_from_image(
                image=image,
                procedure_name=procedure_name,
                step_number=step_num
            )
            
            if result and "error" not in result:
                # Build rich text content for embedding
                content_parts = [
                    f"Procédure: {procedure_name}",
                    f"Étape {step_num}: {result.get('summary', '')}",
                    f"Action: {result.get('action', '')}",
                ]
                
                if result.get('navigation'):
                    content_parts.append(f"Navigation: {result['navigation']}")
                
                if result.get('visual_location'):
                    content_parts.append(f"Position: {result['visual_location']}")
                
                if result.get('fields'):
                    content_parts.append(f"Champs: {', '.join(result['fields'])}")
                
                if result.get('buttons'):
                    content_parts.append(f"Boutons: {', '.join(result['buttons'])}")
                
                if result.get('warnings'):
                    content_parts.append(f"⚠️ Notes: {', '.join(result['warnings'])}")
                
                page_content = "\n".join(content_parts)
                
                doc = {
                    "page_content": page_content,
                    "metadata": {
                        "source": pdf_path.name,
                        "procedure_name": procedure_name,
                        "step_order": step_num,
                        "total_steps": len(images),
                        "type": "ngbss_guide",  # Important for filtering!
                        "action": result.get('action', ''),
                        "navigation": result.get('navigation', ''),
                        "raw_extraction": json.dumps(result, ensure_ascii=False)
                    }
                }
                documents.append(doc)
                print("✅")
            else:
                print(f"⚠️ Skipped (extraction failed)")
        
        return documents
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDF files in the input folder."""
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        print(f"\n📚 Found {len(pdf_files)} PDF files in {self.pdf_folder}")
        
        all_documents = []
        
        for pdf_path in pdf_files:
            docs = self.process_single_pdf(pdf_path)
            all_documents.extend(docs)
        
        print(f"\n✅ Total documents extracted: {len(all_documents)}")
        return all_documents
    
    def save_to_json(self, documents: List[Dict], output_path: str = "data/ngbss_extracted.json"):
        """Save extracted documents to JSON for inspection."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved to {output_file}")
    
    def ingest_to_chromadb(self, documents: List[Dict]):
        """
        Ingest documents into ChromaDB.
        
        Uses a SEPARATE collection for NGBSS guides to allow filtered retrieval.
        """
        if not documents:
            print("⚠️ No documents to ingest!")
            return
        
        print(f"\n📦 Ingesting {len(documents)} documents to ChromaDB...")
        print(f"   Collection: {self.collection_name}")
        
        # Get or create collection
        collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "NGBSS procedural guides for AT agents"}
        )
        
        # Prepare data for ingestion
        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [f"ngbss_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        print("   🧠 Generating embeddings...")
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Upsert to ChromaDB
        print("   💾 Upserting to ChromaDB...")
        collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ Successfully ingested {len(documents)} NGBSS guide steps!")
        print(f"   Collection now has {collection.count()} documents")
    
    def run(self, save_json: bool = True):
        """
        Run the complete ingestion pipeline.
        
        Args:
            save_json: Whether to save extracted data to JSON file
        """
        print("=" * 60)
        print("🚀 NGBSS Guide Ingestion Pipeline")
        print("=" * 60)
        
        # Test Gemini connection
        print("\n🔌 Testing Gemini API connection...")
        if not self.vision.test_connection():
            print("❌ Failed to connect to Gemini. Check your API keys.")
            return
        print("✅ Gemini API connected!")
        
        # Process all PDFs
        documents = self.process_all_pdfs()
        
        # Save to JSON for inspection
        if save_json:
            self.save_to_json(documents)
        
        # Ingest to ChromaDB
        self.ingest_to_chromadb(documents)
        
        print("\n" + "=" * 60)
        print("✅ NGBSS Ingestion Complete!")
        print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest NGBSS PDF guides into ChromaDB")
    parser.add_argument(
        "--pdf-folder", 
        type=str, 
        default="data/ngbss",
        help="Path to folder containing NGBSS PDF files"
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="data/chroma_db",
        help="Path to ChromaDB persistence directory"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="ngbss-guides",
        help="ChromaDB collection name for NGBSS guides"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving extracted data to JSON"
    )
    
    args = parser.parse_args()
    
    # Check if PDF folder exists
    pdf_folder = Path(args.pdf_folder)
    if not pdf_folder.exists():
        print(f"❌ PDF folder not found: {pdf_folder}")
        print(f"   Please create it and add your NGBSS PDF files.")
        pdf_folder.mkdir(parents=True, exist_ok=True)
        print(f"   Created empty folder: {pdf_folder}")
        return
    
    # Run pipeline
    pipeline = NGBSSIngestionPipeline(
        pdf_folder=args.pdf_folder,
        chroma_persist_dir=args.chroma_dir,
        collection_name=args.collection
    )
    pipeline.run(save_json=not args.no_json)


if __name__ == "__main__":
    main()

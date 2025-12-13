"""
Hybrid Retriever Implementation

Combines BM25 (sparse/keyword) and Dense (semantic/embedding) retrieval
using Reciprocal Rank Fusion (RRF) for improved search quality.

Includes language detection to filter documents by language.
"""

import time
import re
from typing import List, Optional, Dict, Any

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank-bm25 is required for hybrid retrieval. Install with: pip install rank-bm25"
    )


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.
    
    Returns:
        'ar' for Arabic, 'fr' for French/Latin scripts
    """
    # Arabic Unicode range: \u0600-\u06FF (Arabic), \u0750-\u077F (Arabic Supplement)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
    
    # Count Arabic characters
    arabic_chars = len(arabic_pattern.findall(text))
    total_chars = len(text.replace(' ', ''))
    
    if total_chars == 0:
        return 'fr'  # Default to French
    
    arabic_ratio = arabic_chars / total_chars
    
    # If more than 20% Arabic characters, consider it Arabic
    if arabic_ratio > 0.2:
        return 'ar'
    return 'fr'


def filter_documents_by_language(documents: List[Document], target_language: str) -> List[Document]:
    """
    Filter documents to match the target language.
    
    Args:
        documents: List of documents to filter
        target_language: 'ar' for Arabic, 'fr' for French
        
    Returns:
        Filtered list of documents matching the target language
    """
    filtered = []
    for doc in documents:
        doc_language = detect_language(doc.page_content)
        if doc_language == target_language:
            filtered.append(doc)
    
    # If no documents match, return original (fallback)
    if not filtered:
        print(f"  ⚠️ No documents found for language '{target_language}', using all documents")
        return documents
    
    return filtered

# Import reranker if available
try:
    from src.reranker import BGEReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


class HybridRetriever(BaseRetriever):
    """
    Custom hybrid retriever that combines BM25 and dense retrieval
    with timing tracking and optional reranking.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.
    """
    
    bm25_retriever: BM25Retriever
    dense_retriever: Any  # VectorStore retriever
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    ensemble_retriever: Optional[EnsembleRetriever] = None
    timing_data: Dict[str, float] = {}
    reranker: Optional[Any] = None
    reranker_top_k: int = 5
    filter_by_language: bool = True  # Enable language filtering by default
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create ensemble retriever with specified weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[self.bm25_weight, self.dense_weight]
        )
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search with timing.
        
        Args:
            query: Search query string
            run_manager: Optional callback manager
            
        Returns:
            List of relevant documents, ranked by combined score
        """
        # Reset timing data
        self.timing_data = {}
        
        # Step 1: BM25 retrieval
        bm25_start = time.time()
        bm25_docs = self.bm25_retriever.invoke(query)
        self.timing_data["bm25_search"] = (time.time() - bm25_start) * 1000
        
        # Step 2: Dense retrieval
        dense_start = time.time()
        dense_docs = self.dense_retriever.invoke(query)
        self.timing_data["dense_search"] = (time.time() - dense_start) * 1000
        
        # Step 3: Ensemble/Fusion (RRF)
        fusion_start = time.time()
        combined_docs = self.ensemble_retriever.invoke(query)
        self.timing_data["rrf_fusion"] = (time.time() - fusion_start) * 1000
        
        # Step 4: Language filtering (filter documents to match query language)
        if self.filter_by_language:
            filter_start = time.time()
            query_language = detect_language(query)
            pre_filter_count = len(combined_docs)
            combined_docs = filter_documents_by_language(combined_docs, query_language)
            self.timing_data["language_filter"] = (time.time() - filter_start) * 1000
            print(f"  🌐 Language filter: {query_language.upper()} ({pre_filter_count} → {len(combined_docs)} docs)")
        
        # Step 5: Optional reranking
        if self.reranker is not None:
            rerank_start = time.time()
            combined_docs = self.reranker.rerank(query, combined_docs)[:self.reranker_top_k]
            self.timing_data["reranking"] = (time.time() - rerank_start) * 1000
        
        # Calculate total hybrid search time
        self.timing_data["hybrid_search"] = sum([
            self.timing_data.get("bm25_search", 0),
            self.timing_data.get("dense_search", 0),
            self.timing_data.get("rrf_fusion", 0)
        ])
        
        return combined_docs


def get_hybrid_retriever(
    documents: List[Document],
    vectorstore: Chroma,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    bm25_k: int = 10,
    dense_k: int = 10,
    use_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_top_k: int = 5
) -> HybridRetriever:
    """
    Create a hybrid retriever from documents and an existing vectorstore.
    
    Args:
        documents: List of documents for BM25 indexing
        vectorstore: Existing ChromaDB vectorstore for dense retrieval
        bm25_weight: Weight for BM25 retriever (0.0-1.0)
        dense_weight: Weight for dense retriever (0.0-1.0)
        bm25_k: Number of documents to retrieve from BM25
        dense_k: Number of documents to retrieve from dense search
        use_reranker: Whether to apply reranking after fusion
        reranker_model: Model name for reranker
        reranker_top_k: Number of top documents after reranking
        
    Returns:
        HybridRetriever instance
    """
    # Validate weights
    if bm25_weight + dense_weight != 1.0:
        # Normalize weights
        total = bm25_weight + dense_weight
        bm25_weight = bm25_weight / total
        dense_weight = dense_weight / total
    
    # Create BM25 retriever from documents
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = bm25_k
    
    # Create dense retriever from vectorstore
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": dense_k}
    )
    
    # Initialize reranker if requested
    reranker = None
    if use_reranker and RERANKER_AVAILABLE:
        reranker = BGEReranker(
            model_name=reranker_model,
            top_k=reranker_top_k
        )
    
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        reranker=reranker,
        reranker_top_k=reranker_top_k
    )


def get_hybrid_retriever_from_vectorstore(
    vectorstore: Chroma,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    bm25_k: int = 10,
    dense_k: int = 10,
    use_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_top_k: int = 5
) -> HybridRetriever:
    """
    Create a hybrid retriever directly from a vectorstore.
    Extracts documents from the vectorstore for BM25 indexing.
    
    Args:
        vectorstore: ChromaDB vectorstore
        bm25_weight: Weight for BM25 retriever (0.0-1.0)
        dense_weight: Weight for dense retriever (0.0-1.0)
        bm25_k: Number of documents to retrieve from BM25
        dense_k: Number of documents to retrieve from dense search
        use_reranker: Whether to apply reranking after fusion
        reranker_model: Model name for reranker
        reranker_top_k: Number of top documents after reranking
        
    Returns:
        HybridRetriever instance
    """
    # Extract documents from vectorstore
    print("  [Hybrid] Extracting documents from vectorstore for BM25 indexing...")
    extraction_start = time.time()
    
    collection = vectorstore._collection
    results = collection.get(include=["documents", "metadatas"])
    
    documents = []
    for doc_text, metadata in zip(results["documents"], results["metadatas"]):
        documents.append(Document(
            page_content=doc_text,
            metadata=metadata or {}
        ))
    
    extraction_time = (time.time() - extraction_start) * 1000
    print(f"       ✓ Extracted {len(documents)} documents in {extraction_time:.2f}ms")
    
    return get_hybrid_retriever(
        documents=documents,
        vectorstore=vectorstore,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        bm25_k=bm25_k,
        dense_k=dense_k,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        reranker_top_k=reranker_top_k
    )


# Preset configurations for common use cases

def get_balanced_retriever(
    documents: List[Document],
    vectorstore: Chroma,
    use_reranker: bool = False
) -> HybridRetriever:
    """Balanced 50/50 hybrid retriever."""
    return get_hybrid_retriever(
        documents=documents,
        vectorstore=vectorstore,
        bm25_weight=0.5,
        dense_weight=0.5,
        use_reranker=use_reranker
    )


def get_keyword_focused_retriever(
    documents: List[Document],
    vectorstore: Chroma,
    use_reranker: bool = False
) -> HybridRetriever:
    """Keyword-focused retriever (70% BM25, 30% Dense)."""
    return get_hybrid_retriever(
        documents=documents,
        vectorstore=vectorstore,
        bm25_weight=0.7,
        dense_weight=0.3,
        use_reranker=use_reranker
    )


def get_semantic_focused_retriever(
    documents: List[Document],
    vectorstore: Chroma,
    use_reranker: bool = False
) -> HybridRetriever:
    """Semantic-focused retriever (30% BM25, 70% Dense)."""
    return get_hybrid_retriever(
        documents=documents,
        vectorstore=vectorstore,
        bm25_weight=0.3,
        dense_weight=0.7,
        use_reranker=use_reranker
    )

"""
Reranker module for improving retrieval quality using BGE reranker models.
"""
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain.schema import Document


class BGEReranker:
    """
    BGE Reranker for reranking retrieved documents based on relevance to query.
    Uses cross-encoder models from BAAI for better semantic matching.
    """
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-v2-m3", 
        top_k: int = 4,
        batch_size: int = 32
    ):
        """
        Initialize the BGE reranker.
        
        Args:
            model_name: HuggingFace model name for the reranker
            top_k: Number of top documents to return after reranking
            batch_size: Batch size for inference (higher = faster but more memory)
        """
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.model = None
        
    def _load_model(self):
        """Lazy load the reranker model with GPU support if available."""
        if self.model is None:
            print(f"[reranker] Loading reranker model: {self.model_name}")
            
            # CrossEncoder will automatically use GPU if available
            self.model = CrossEncoder(self.model_name, max_length=512)
            
            # Check if GPU is being used
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[reranker] Model loaded successfully on {device}")
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The user's query
            documents: List of retrieved documents to rerank
            
        Returns:
            List of reranked documents (top_k most relevant)
        """
        if not documents:
            return documents
        
        # Load model 
        self._load_model()
        
        # Prepare query-document pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores (batched for efficiency)
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scored_docs = list(zip(scores, documents))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k documents
        top_docs = [doc for _, doc in scored_docs[:self.top_k]]
        
        print(f"[reranker] Reranked {len(documents)} documents, returning top {len(top_docs)}")
        
        return top_docs
    
    def rerank_with_scores(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return them with their relevance scores.
        
        Args:
            query: The user's query
            documents: List of retrieved documents to rerank
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []
        
        # Load model 
        self._load_model()
        
        # Prepare query-document pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores (batched for efficiency)
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        # Create list of (score, document) tuples
        scored_docs = list(zip(scores, documents))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k documents with scores
        top_docs_with_scores = [(doc, float(score)) for score, doc in scored_docs[:self.top_k]]
        
        print(f"[reranker] Reranked {len(documents)} documents with scores, returning top {len(top_docs_with_scores)}")
        
        return top_docs_with_scores

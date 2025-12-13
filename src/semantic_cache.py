"""
Semantic Cache Layer for RAG System
Uses Redis with vector similarity search to cache query responses
and serve instant results for repeated or similar questions.
"""

import json
import time
import hashlib
from typing import Optional, Dict, Any, List
import numpy as np

REDIS_AVAILABLE = False

try:
    import redis
    from redis.commands.search.field import VectorField, TextField, NumericField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except ImportError:
    print("⚠️ Redis search module not available. Semantic caching will be disabled.")
    print("   To enable, install with: pip install redis[hiredis]")


class SemanticCache:
    """
    Semantic cache that stores query embeddings and responses in Redis.
    Uses cosine similarity to match similar queries and return cached responses.
    """
    
    # Similarity threshold for cache hits (adjustable)
    # 0.95 = very similar (recommended for production)
    # 0.90 = moderately similar (more cache hits, less precise)
    # 0.98 = extremely similar (fewer cache hits, very precise)
    DEFAULT_SIMILARITY_THRESHOLD = 0.95
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        embedding_dim: int = 768,  # Default for multilingual-e5-base
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl_seconds: int = 86400,  # 24 hours default TTL
        index_name: str = "semantic_cache_idx",
    ):
        """
        Initialize semantic cache with Redis connection.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            embedding_dim: Dimension of embedding vectors
            similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0)
            ttl_seconds: Time-to-live for cached entries in seconds
            index_name: Name of the Redis search index
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,  # We'll handle encoding ourselves
        )
        
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.index_name = index_name
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0,
        }
        
        # Initialize the search index
        self._create_index()
    
    def _create_index(self):
        """
        Create Redis search index for vector similarity search.
        This is idempotent - won't fail if index already exists.
        """
        try:
            # Check if index exists
            self.redis_client.ft(self.index_name).info()
            print(f"[Semantic Cache] Index '{self.index_name}' already exists")
        except:
            # Create new index
            schema = (
                VectorField(
                    "embedding",
                    "FLAT",  # Can use HNSW for larger datasets
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dim,
                        "DISTANCE_METRIC": "COSINE",
                    }
                ),
                TextField("query"),
                TextField("response"),
                TextField("sources"),
                NumericField("timestamp"),
            )
            
            definition = IndexDefinition(
                prefix=["cache:"],
                index_type=IndexType.HASH
            )
            
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition
            )
            print(f"[Semantic Cache] Created index '{self.index_name}'")
    
    def _generate_cache_key(self, query: str) -> str:
        """
        Generate a unique cache key for a query.
        
        Args:
            query: User query string
            
        Returns:
            Cache key with 'cache:' prefix
        """
        # Use hash for consistent key generation
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"cache:{query_hash}"
    
    def _embedding_to_bytes(self, embedding: List[float]) -> bytes:
        """Convert embedding list to bytes for Redis storage."""
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _bytes_to_embedding(self, embedding_bytes: bytes) -> List[float]:
        """Convert bytes from Redis to embedding list."""
        return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
    
    def get(
        self, 
        query: str, 
        query_embedding: List[float],
        similarity_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a query if similarity is above threshold.
        
        Args:
            query: User query string
            query_embedding: Embedding vector of the query
            similarity_threshold: Override default similarity threshold
            
        Returns:
            Cached response dict with 'answer', 'sources', 'similarity', 'cache_hit'
            or None if no cache hit
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        threshold = similarity_threshold or self.similarity_threshold
        
        try:
            # Convert embedding to bytes for search
            query_vector = self._embedding_to_bytes(query_embedding)
            
            # KNN search for most similar cached query
            # The (1 - similarity) because Redis returns distance, not similarity
            query_obj = (
                Query(f"*=>[KNN 1 @embedding $vec AS similarity]")
                .sort_by("similarity")
                .return_fields("query", "response", "sources", "similarity")
                .dialect(2)
            )
            
            result = self.redis_client.ft(self.index_name).search(
                query_obj,
                query_params={"vec": query_vector}
            )
            
            if result.total > 0:
                doc = result.docs[0]
                
                # Redis returns distance, convert to similarity
                # Cosine distance = 1 - cosine_similarity
                similarity_score = 1 - float(doc.similarity)
                
                if similarity_score >= threshold:
                    # Cache hit!
                    elapsed_ms = (time.time() - start_time) * 1000
                    self.stats["hits"] += 1
                    
                    print(f"✅ [Semantic Cache] HIT (similarity: {similarity_score:.4f}, "
                          f"latency: {elapsed_ms:.2f}ms)")
                    print(f"   Original query: {doc.query}")
                    print(f"   Current query:  {query}")
                    
                    return {
                        "answer": doc.response,
                        "sources": json.loads(doc.sources) if doc.sources != "[]" else [],
                        "similarity": similarity_score,
                        "cache_hit": True,
                        "cached_query": doc.query,
                        "latency_ms": elapsed_ms,
                    }
                else:
                    # Similarity below threshold
                    print(f"⚠️  [Semantic Cache] MISS (similarity: {similarity_score:.4f} "
                          f"< threshold: {threshold:.4f})")
            else:
                # No results found
                print(f"❌ [Semantic Cache] MISS (no cached queries)")
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            print(f"❌ [Semantic Cache] Error during retrieval: {e}")
            self.stats["misses"] += 1
            return None
    
    def set(
        self,
        query: str,
        query_embedding: List[float],
        response: str,
        sources: List[str] = None,
    ):
        """
        Store a query and its response in the cache.
        
        Args:
            query: User query string
            query_embedding: Embedding vector of the query
            response: Generated response text
            sources: List of source documents (optional)
        """
        try:
            cache_key = self._generate_cache_key(query)
            
            # Prepare data for storage
            data = {
                "query": query,
                "embedding": self._embedding_to_bytes(query_embedding),
                "response": response,
                "sources": json.dumps(sources or []),
                "timestamp": int(time.time()),
            }
            
            # Store in Redis with TTL
            self.redis_client.hset(cache_key, mapping=data)
            
            if self.ttl_seconds > 0:
                self.redis_client.expire(cache_key, self.ttl_seconds)
            
            print(f"💾 [Semantic Cache] Stored: {query[:60]}...")
            
        except Exception as e:
            print(f"❌ [Semantic Cache] Error during storage: {e}")
    
    def clear(self):
        """Clear all cached entries."""
        try:
            # Delete all keys with 'cache:' prefix
            keys = self.redis_client.keys("cache:*")
            if keys:
                self.redis_client.delete(*keys)
                print(f"🗑️  [Semantic Cache] Cleared {len(keys)} entries")
            else:
                print(f"🗑️  [Semantic Cache] No entries to clear")
        except Exception as e:
            print(f"❌ [Semantic Cache] Error during clear: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, hit rate, and total queries
        """
        total = self.stats["total_queries"]
        hits = self.stats["hits"]
        
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        
        return {
            "hits": hits,
            "misses": self.stats["misses"],
            "total_queries": total,
            "hit_rate_percent": round(hit_rate, 2),
        }
    
    def print_stats(self):
        """Print cache statistics to console."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("📊 SEMANTIC CACHE STATISTICS")
        print("=" * 60)
        print(f"Total Queries:  {stats['total_queries']}")
        print(f"Cache Hits:     {stats['hits']} ✅")
        print(f"Cache Misses:   {stats['misses']} ❌")
        print(f"Hit Rate:       {stats['hit_rate_percent']:.2f}%")
        print("=" * 60 + "\n")
    
    def update_similarity_threshold(self, new_threshold: float):
        """
        Update the similarity threshold for cache hits.
        
        Args:
            new_threshold: New similarity threshold (0.0-1.0)
                - 0.95: Very similar (recommended for production)
                - 0.90: Moderately similar (more cache hits)
                - 0.98: Extremely similar (fewer cache hits)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        
        old_threshold = self.similarity_threshold
        self.similarity_threshold = new_threshold
        print(f"🔧 [Semantic Cache] Threshold updated: {old_threshold:.2f} → {new_threshold:.2f}")


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Semantic Cache\n")
    
    # Initialize cache
    cache = SemanticCache(
        redis_host="localhost",
        redis_port=6379,
        embedding_dim=768,
        similarity_threshold=0.95,
        ttl_seconds=3600,  # 1 hour
    )
    
    # Test with dummy embeddings
    print("1. Testing cache miss...")
    test_embedding = [0.1] * 768  # Dummy embedding
    result = cache.get("What is the WEEKEND BOOST offer?", test_embedding)
    print(f"Result: {result}\n")
    
    print("2. Storing in cache...")
    cache.set(
        query="What is the WEEKEND BOOST offer?",
        query_embedding=test_embedding,
        response="The WEEKEND BOOST offer provides...",
        sources=["doc1.pdf", "doc2.pdf"]
    )
    
    print("\n3. Testing cache hit with same query...")
    result = cache.get("What is the WEEKEND BOOST offer?", test_embedding)
    print(f"Cache hit: {result is not None}\n")
    
    print("4. Testing cache hit with similar query...")
    similar_embedding = [0.101] * 768  # Very similar
    result = cache.get("Tell me about WEEKEND BOOST", similar_embedding)
    print(f"Cache hit: {result is not None}\n")
    
    # Print statistics
    cache.print_stats()

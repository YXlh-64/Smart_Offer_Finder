#!/usr/bin/env python3
"""
Test script for Semantic Cache functionality
Demonstrates cache hits/misses and performance improvements
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("🧪 SEMANTIC CACHE TEST")
print("="*70)
print()

# Test 1: Check Redis connection
print("Test 1: Redis Connection")
print("-" * 70)
try:
    from src.semantic_cache import SemanticCache
    
    cache = SemanticCache(
        redis_host="localhost",
        redis_port=6379,
        similarity_threshold=0.95,
    )
    print("✅ Redis connection successful")
    print(f"   Index: {cache.index_name}")
    print(f"   Threshold: {cache.similarity_threshold}")
    print()
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    print("   Please ensure Redis is running: sudo systemctl start redis-server")
    sys.exit(1)

# Test 2: Clear cache for clean test
print("Test 2: Clear Cache")
print("-" * 70)
cache.clear()
print()

# Test 3: Test cache miss (first query)
print("Test 3: Cache Miss (New Query)")
print("-" * 70)

# Create a dummy embedding
dummy_embedding = [0.1 + i*0.001 for i in range(768)]

query1 = "Qu'est-ce que l'offre WEEKEND BOOST?"

start = time.time()
result = cache.get(query1, dummy_embedding)
elapsed_ms = (time.time() - start) * 1000

print(f"Query: {query1}")
print(f"Result: {result}")
print(f"Time: {elapsed_ms:.2f}ms")
print()

# Test 4: Store in cache
print("Test 4: Store Response in Cache")
print("-" * 70)

cache.set(
    query=query1,
    query_embedding=dummy_embedding,
    response="L'offre WEEKEND BOOST permet aux clients d'augmenter leur débit...",
    sources=["offer_guide.pdf", "weekend_boost.pdf"]
)
print()

# Test 5: Test cache hit (exact same query)
print("Test 5: Cache Hit (Exact Same Query)")
print("-" * 70)

start = time.time()
result = cache.get(query1, dummy_embedding)
elapsed_ms = (time.time() - start) * 1000

print(f"Query: {query1}")
print(f"Cache Hit: {result is not None}")
if result:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Answer: {result['answer'][:50]}...")
    print(f"Time: {elapsed_ms:.2f}ms")
print()

# Test 6: Test cache hit with similar query
print("Test 6: Cache Hit (Similar Query)")
print("-" * 70)

# Create a very similar embedding (small perturbation)
similar_embedding = [0.1 + i*0.001 + 0.0001 for i in range(768)]
query2 = "Explique-moi l'offre WEEKEND BOOST"

start = time.time()
result = cache.get(query2, similar_embedding)
elapsed_ms = (time.time() - start) * 1000

print(f"Query: {query2}")
print(f"Cache Hit: {result is not None}")
if result:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Cached Query: {result['cached_query']}")
    print(f"Time: {elapsed_ms:.2f}ms")
else:
    print(f"Time: {elapsed_ms:.2f}ms (Miss)")
print()

# Test 7: Test cache miss with very different query
print("Test 7: Cache Miss (Different Query)")
print("-" * 70)

different_embedding = [0.5 + i*0.002 for i in range(768)]
query3 = "Quels sont les services d'interconnexion?"

start = time.time()
result = cache.get(query3, different_embedding)
elapsed_ms = (time.time() - start) * 1000

print(f"Query: {query3}")
print(f"Cache Hit: {result is not None}")
print(f"Time: {elapsed_ms:.2f}ms")
print()

# Test 8: Test threshold adjustment
print("Test 8: Similarity Threshold Adjustment")
print("-" * 70)

print("Original threshold: 0.95")
cache.update_similarity_threshold(0.90)
print("Updated threshold: 0.90")
print()

# Test with new threshold
result = cache.get(query2, similar_embedding, similarity_threshold=0.90)
print(f"Cache Hit with 0.90 threshold: {result is not None}")

cache.update_similarity_threshold(0.98)
result = cache.get(query2, similar_embedding, similarity_threshold=0.98)
print(f"Cache Hit with 0.98 threshold: {result is not None}")
print()

# Reset threshold
cache.update_similarity_threshold(0.95)

# Test 9: Test TTL expiration (if short TTL)
print("Test 9: Cache Statistics")
print("-" * 70)
cache.print_stats()

# Test 10: Multiple queries performance comparison
print("Test 10: Performance Comparison")
print("-" * 70)

# Simulate cache miss (different query each time)
print("Simulating 5 cache misses...")
miss_times = []
for i in range(5):
    embedding = [0.1 + i*0.1 + j*0.001 for j in range(768)]
    start = time.time()
    cache.get(f"Query {i}", embedding)
    miss_times.append((time.time() - start) * 1000)

avg_miss = sum(miss_times) / len(miss_times)
print(f"Average cache miss time: {avg_miss:.2f}ms")

# Simulate cache hits (same query)
print("\nSimulating 5 cache hits...")
hit_times = []
for i in range(5):
    start = time.time()
    cache.get(query1, dummy_embedding)
    hit_times.append((time.time() - start) * 1000)

avg_hit = sum(hit_times) / len(hit_times)
print(f"Average cache hit time: {avg_hit:.2f}ms")

speedup = avg_miss / avg_hit if avg_hit > 0 else 0
print(f"\n🚀 Speedup: {speedup:.1f}x faster")
print()

# Final statistics
print("="*70)
print("📊 FINAL STATISTICS")
print("="*70)
cache.print_stats()

print("✅ All tests completed!")
print()
print("Next steps:")
print("1. Ensure Redis is running")
print("2. Run: python main.py")
print("3. Test with real queries via the API")

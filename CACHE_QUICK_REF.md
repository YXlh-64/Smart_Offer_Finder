# 🎯 Semantic Cache - Quick Reference

## Installation

```bash
# 1. Install Redis
sudo apt install redis-server
sudo systemctl start redis-server

# 2. Install Python dependencies
pip install redis[hiredis]>=5.0.0 numpy>=1.24.0

# 3. Configure (optional - defaults work)
echo "USE_SEMANTIC_CACHE=true" >> .env
echo "CACHE_SIMILARITY_THRESHOLD=0.95" >> .env

# 4. Start backend
python main.py
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_SEMANTIC_CACHE` | `true` | Enable/disable caching |
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity threshold (0.0-1.0) |
| `CACHE_TTL_SECONDS` | `86400` | Time-to-live in seconds (24h) |

## Similarity Threshold Guide

```
0.98 ━━━━━━━ Very Strict   (fewer hits, very precise)
0.95 ━━━━━━━ Recommended   (good balance) ⭐
0.90 ━━━━━━━ Lenient      (more hits, less precise)
0.85 ━━━━━━━ Very Lenient (many hits, some false positives)
```

**Adjusting the threshold:**

In `.env`:
```bash
CACHE_SIMILARITY_THRESHOLD=0.92
```

Or dynamically:
```python
from src.chat import semantic_cache
semantic_cache.update_similarity_threshold(0.92)
```

## Performance Metrics

```
┌──────────────────────┬─────────────┐
│ Cache Hit            │ 50-100ms    │ ⚡ FAST
│ Cache Miss (No RR)   │ 2000-3000ms │
│ Cache Miss (With RR) │ 3000-5000ms │
└──────────────────────┴─────────────┘

Expected Speedup: 30-60x faster for cache hits
```

## Console Output

```bash
✅ [Semantic Cache] HIT (similarity: 0.9721, latency: 52ms)
❌ [Semantic Cache] MISS (no cached queries)
💾 [Semantic Cache] Stored: Query about WEEKEND BOOST...
```

## API - No Changes Required!

The caching is **completely transparent**:

```python
# Existing code works as-is
result = cached_chain_invoke("Your question here")

# Automatically checks cache first
# Returns cached response if similarity > threshold
# Otherwise invokes RAG and caches result
```

## Monitoring

### Get Statistics
```python
from src.chat import semantic_cache

stats = semantic_cache.get_stats()
# {'hits': 45, 'misses': 23, 'total_queries': 68, 'hit_rate_percent': 66.18}

semantic_cache.print_stats()
```

### Redis Commands
```bash
# Check connection
redis-cli ping

# View cached keys
redis-cli KEYS "cache:*"

# Get cache size
redis-cli DBSIZE

# Monitor real-time
redis-cli MONITOR

# Clear cache
redis-cli FLUSHDB
```

## Common Operations

### Clear Cache
```python
semantic_cache.clear()
```

### Update Threshold
```python
semantic_cache.update_similarity_threshold(0.92)
```

### Disable Caching
```bash
# In .env
USE_SEMANTIC_CACHE=false
```

### Check if Enabled
```python
from src.chat import settings
if settings.use_semantic_cache:
    print("✅ Enabled")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Redis connection refused | `sudo systemctl start redis-server` |
| Module not found | `pip install redis[hiredis]` |
| Too few cache hits | Lower threshold to 0.90-0.92 |
| Too much memory | Reduce TTL or clear cache |

## Testing

```bash
# Test cache functionality
python test_semantic_cache.py

# Test with real queries
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Qu\'est-ce que WEEKEND BOOST?", "session_id": "test"}'

# Run same query again - should be much faster!
```

## Architecture Flow

```
Query → Embed → Redis KNN Search
              ↓
        Similarity ≥ 0.95?
        ↓           ↓
       YES          NO
        ↓           ↓
    Cache Hit   Invoke RAG
    (~50ms)     (~3000ms)
        ↓           ↓
        ↓       Store in Cache
        ↓           ↓
        └───────────┘
              ↓
          Return Response
```

## Example Queries

### First time (Cache Miss)
```
Q: "Qu'est-ce que l'offre WEEKEND BOOST?"
→ Miss → Full RAG → 3200ms
```

### Second time (Cache Hit)
```
Q: "Qu'est-ce que l'offre WEEKEND BOOST?"
→ HIT → 52ms (61x faster!)
```

### Similar query (Cache Hit)
```
Q: "Explique-moi WEEKEND BOOST"
→ HIT (similarity: 0.97) → 48ms
```

### Different query (Cache Miss)
```
Q: "Quels sont les services d'interconnexion?"
→ Miss → Full RAG → 2900ms
```

## Key Files

- `src/semantic_cache.py` - Cache implementation
- `src/config.py` - Configuration settings
- `src/chat.py` - Integration with RAG chain
- `main.py` - FastAPI endpoints with caching
- `test_semantic_cache.py` - Test script
- `SEMANTIC_CACHE_GUIDE.md` - Full documentation

## Quick Start Checklist

- [ ] Redis installed and running
- [ ] Dependencies installed (`redis[hiredis]`, `numpy`)
- [ ] Backend started (`python main.py`)
- [ ] Semantic cache initialization logged
- [ ] Test with repeated query
- [ ] Verify cache hit in console
- [ ] Check statistics

---

**Ready to use!** Just restart your backend and start asking questions. 🚀

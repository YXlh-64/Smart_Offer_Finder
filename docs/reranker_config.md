# Reranker Configuration Guide

## Enabling/Disabling the Reranker

You can now easily toggle the reranker on or off using the `USE_RERANKER` environment variable.

### Quick Toggle

Add to your `.env` file:

```env
# Enable reranker (default)
USE_RERANKER=true

# Disable reranker (faster, but less accurate)
USE_RERANKER=false
```

### When to Enable vs Disable

#### ✅ Enable Reranker (USE_RERANKER=true)

**Best for:**
- High accuracy requirements
- Complex/technical documents
- When you have GPU available
- Response time < 2 seconds is acceptable

**Performance:**
- Latency: ~1-2 seconds per query
- Accuracy: +10-20% improvement
- Memory: ~2GB

#### ⚡ Disable Reranker (USE_RERANKER=false)

**Best for:**
- Speed is critical (< 500ms response needed)
- Simple documents
- CPU-only environments
- Limited memory/resources

**Performance:**
- Latency: ~500ms per query
- Accuracy: Baseline (still good with quality embeddings)
- Memory: Minimal

### Configuration Options

```env
# Toggle reranker on/off
USE_RERANKER=true

# Reranker settings (only used if USE_RERANKER=true)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_TOP_K=4
INITIAL_RETRIEVAL_K=20
```

### Behavior

**With Reranker Enabled:**
```
Query → Retrieve 20 docs → Rerank → Top 4 docs → LLM
```

**With Reranker Disabled:**
```
Query → Retrieve 4 docs → LLM
```

### Testing Both Modes

Try both modes to see which works best for your use case:

```bash
# Test with reranker
echo "USE_RERANKER=true" >> .env
python -m chatbot.src.chat

# Test without reranker
echo "USE_RERANKER=false" >> .env
python -m chatbot.src.chat
```

Compare response time and accuracy for your specific documents and queries.

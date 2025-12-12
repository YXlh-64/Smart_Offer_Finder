# BGE Reranker Example

This example demonstrates how the BGE reranker improves retrieval quality.

## How It Works

### Without Reranker (Previous Approach)
```
Query → Vector Search → Top 4 Documents → LLM
```

### With Reranker (New Approach)
```
Query → Vector Search (20 docs) → BGE Reranker → Top 4 Documents → LLM
```

## Why Reranking Helps

1. **Better Semantic Understanding**: Cross-encoders (used by BGE reranker) process query and document together, capturing subtle semantic relationships that bi-encoders (used for initial retrieval) might miss.

2. **Two-Stage Retrieval**: 
   - Stage 1: Fast vector search retrieves 20 candidates
   - Stage 2: Slower but more accurate reranker selects best 4

3. **Improved Relevance**: Documents are ranked by actual semantic similarity to the query, not just vector distance.

## Example Scenario

**Query**: "What are the payment terms for international contracts?"

**Without Reranker** (vector similarity only):
- Doc 1: General payment information (score: 0.82)
- Doc 2: International shipping terms (score: 0.81)
- Doc 3: Contract templates (score: 0.79)
- Doc 4: Payment deadlines (score: 0.78)

**With Reranker** (semantic relevance):
- Doc 1: International contract payment terms (rerank score: 0.95)
- Doc 2: Payment schedules for cross-border deals (rerank score: 0.89)
- Doc 3: Currency and payment methods (rerank score: 0.85)
- Doc 4: International payment regulations (rerank score: 0.82)

The reranker identifies documents that are semantically more relevant to the specific question about "payment terms" in "international contracts".

## Configuration

Adjust reranker behavior in `.env`:

```env
# Retrieve more candidates for better reranking
INITIAL_RETRIEVAL_K=20

# Number of final documents to use
RERANKER_TOP_K=4

# Change reranker model if needed
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

## Performance Notes

- **First run**: Model downloads automatically (~1.5GB for bge-reranker-v2-m3)
- **Latency**: Adds ~200-500ms per query (depending on hardware)
- **Accuracy**: Typically improves retrieval quality by 10-20%
- **Memory**: Requires ~2GB GPU memory (or CPU if no GPU available)

## Alternative Models

You can use other BGE reranker models:

- `BAAI/bge-reranker-base`: Smaller, faster (278M params)
- `BAAI/bge-reranker-large`: More accurate (560M params)
- `BAAI/bge-reranker-v2-m3`: Multilingual, balanced (default)

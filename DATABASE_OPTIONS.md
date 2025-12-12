# Vector Database Options

This Smart Offer Finder RAG system supports two vector database backends: **Pinecone** (cloud) and **ChromaDB** (local).

## Quick Comparison

| Feature | Pinecone | ChromaDB |
|---------|----------|----------|
| **Deployment** | Cloud-hosted | Local storage |
| **Cost** | Free tier available | Free (open-source) |
| **Scalability** | ~100M vectors (free) | Depends on disk space |
| **Speed** | Fast (managed) | Very fast (local) |
| **Setup** | Requires API key | Zero setup |
| **Internet** | Required | Not required |
| **Persistence** | Automatic | File-based |

---

## Using Pinecone

### Setup

1. **Create a Pinecone account** at [pinecone.io](https://pinecone.io)
2. **Create an index** with:
   - Name: `smart-offer-finder-new`
   - Dimensions: `384`
   - Metric: `cosine`
3. **Get your credentials:**
   - API Key
   - Environment (e.g., `us-east-1-abc1d23`)

4. **Fill `.env` file:**
   ```bash
   PINECONE_API_KEY=your_api_key_here
   PINECONE_INDEX_NAME=smart-offer-finder-new
   PINECONE_ENVIRONMENT=your_environment_here
   ```

### Ingestion

Ingest documents to Pinecone (default):

```fish
source .venv/bin/activate.fish
.venv/bin/python -m src.ingest
# or explicitly:
.venv/bin/python -m src.ingest --db pinecone
```

### Chat

The chat interface uses Pinecone by default:

```fish
source .venv/bin/activate.fish
.venv/bin/python -m src.chat
```

**Note:** If you switch databases, you need to update `src/chat.py` to load the correct vectorstore.

---

## Using ChromaDB

### Setup

ChromaDB requires **zero setup**! It stores vectors locally.

1. **Optional:** Configure storage location in `.env`:
   ```bash
   VECTORSTORE_PATH=data/vectorstore  # Default location
   ```

### Ingestion

Ingest documents to ChromaDB:

```fish
source .venv/bin/activate.fish
.venv/bin/python -m src.ingest --db chromadb
```

This creates a local vector database at `data/vectorstore/`.

### Chat

To use ChromaDB in chat, you need to modify `src/chat.py`:

#### Option A: Update `load_vectorstore()` function

Open `src/chat.py` and replace the `load_vectorstore()` function:

```python
def load_vectorstore(settings) -> Chroma:
    """Load ChromaDB vector store."""
    embeddings = choose_embeddings(settings)
    
    vectorstore = Chroma(
        persist_directory=settings.vectorstore_path,
        embedding_function=embeddings,
        collection_name="smart-offer-finder"
    )
    
    return vectorstore
```

Then add this import at the top:

```python
from langchain_community.vectorstores import Chroma
```

#### Option B: Add command argument to chat.py

For flexibility, you could add a `--db` argument to `chat.py` as well (similar to `ingest.py`).

---

## File Locations

### Pinecone
- **Data location:** Cloud (managed by Pinecone)
- **No local files**

### ChromaDB
- **Data location:** `data/vectorstore/` (default)
- **Structure:**
  ```
  data/vectorstore/
  ├── chroma.sqlite3
  ├── embeddings.parquet
  └── index/
  ```

---

## Switching Databases

### From Pinecone to ChromaDB

```bash
# 1. Ingest to ChromaDB
.venv/bin/python -m src.ingest --db chromadb

# 2. Update src/chat.py to use ChromaDB (see above)

# 3. Restart chat
.venv/bin/python -m src.chat
```

### From ChromaDB to Pinecone

```bash
# 1. Ingest to Pinecone
.venv/bin/python -m src.ingest --db pinecone

# 2. Revert src/chat.py to use Pinecone (original code)

# 3. Restart chat
.venv/bin/python -m src.chat
```

---

## Advantages & Disadvantages

### Pinecone ✅

**Advantages:**
- Managed infrastructure (scalable)
- High performance
- 100M vector capacity (free tier)
- No maintenance

**Disadvantages:**
- Requires internet connection
- API credentials needed
- Cloud dependency

### ChromaDB ✅

**Advantages:**
- Runs locally (no internet needed)
- Zero configuration
- Open source
- Unlimited vectors (disk-limited)
- No vendor lock-in

**Disadvantages:**
- Single-machine only (no distributed)
- Slower for massive datasets
- Manual backup responsibility

---

## Example Usage Scenarios

### Scenario 1: Development (Local)
Use **ChromaDB** for quick testing without API setup.

```fish
.venv/bin/python -m src.ingest --db chromadb
.venv/bin/python -m src.chat
```

### Scenario 2: Production (Scalable)
Use **Pinecone** for high availability and scalability.

```fish
.venv/bin/python -m src.ingest --db pinecone
.venv/bin/python -m src.chat
```

### Scenario 3: Hybrid
Use ChromaDB for testing, Pinecone for production:

```bash
# Development
.venv/bin/python -m src.ingest --db chromadb

# Production
.venv/bin/python -m src.ingest --db pinecone
```

---

## Troubleshooting

### ChromaDB Issues

**Problem:** `No module named 'chromadb'`
```bash
.venv/bin/pip install chromadb>=0.4.0
```

**Problem:** `data/vectorstore/` is empty
```bash
# Make sure you ran ingest first
.venv/bin/python -m src.ingest --db chromadb
```

### Pinecone Issues

**Problem:** `PINECONE_API_KEY is required`
```bash
# Check .env file
cat .env | grep PINECONE_API_KEY
```

**Problem:** `Cannot connect to Pinecone`
- Check internet connection
- Verify API key is correct
- Verify index exists in Pinecone console

---

## Future: Add CLI Support to Chat

For even more flexibility, you could modify `src/chat.py` to accept a `--db` argument:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", choices=["pinecone", "chromadb"], default="pinecone")
    args = parser.parse_args()
    
    # Load appropriate vectorstore based on args.db
    # Then launch chat
```

This would allow:
```bash
.venv/bin/python -m src.chat --db chromadb
```

---

## Summary

- **Default:** Pinecone (cloud, production)
- **Alternative:** ChromaDB (local, development)
- **Switch:** Use `--db` argument during ingestion
- **Chat:** May need code update to switch databases

Choose based on your use case:
- **Local testing?** → ChromaDB
- **Production scale?** → Pinecone

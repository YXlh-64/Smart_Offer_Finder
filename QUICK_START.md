# 🚀 Smart Offer Finder - Quick Reference

## Terminal 1: Start Ollama
```bash
ollama serve
```
**Keep this running while using the app!**

## Terminal 2: Run Everything

### Step 1: Setup (first time only)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Configure
```bash
# Edit .env with your Pinecone credentials
nano .env
```

### Step 3: Add Documents
```bash
# Copy your PDFs to this folder
cp your_documents.pdf data/raw/
```

### Step 4: Ingest Data
```bash
python -m src.ingest
```
Expected output:
```
Successfully indexed XXX chunks to Pinecone index 'smart-offer-finder'.
```

### Step 5: Run Chat
```bash
python -m src.chat
```
Then open: **http://localhost:7860**

---

## 🔑 Essential Environment Variables

| Variable | Value | Source |
|----------|-------|--------|
| `PINECONE_API_KEY` | `pcsk-...` | https://app.pinecone.io/ |
| `PINECONE_ENVIRONMENT` | `us-east-1-xxx` | Pinecone dashboard |
| `PINECONE_INDEX_NAME` | `smart-offer-finder` | Your index name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Your Ollama server |

---

## 🛠️ Common Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull embedding model
ollama pull qllama/multilingual-e5-base

# List Ollama models
ollama list

# Ingest documents
python -m src.ingest

# Run chat interface
python -m src.chat

# Clear Pinecone index (delete all vectors)
# This requires manual deletion in Pinecone dashboard
```

---

## 📦 Available Ollama Models

### Embedding Models
- `multilingual-e5-base` ← **Recommended** (fast, multilingual)
- `nomic-embed-text` (lightweight)
- `mxbai-embed-large` (high quality)

### Chat Models
- `mistral` ← **Recommended** (fast, capable)
- `neural-chat` (smaller, faster)
- `llama2` (larger, slower)
- `openchat` (balanced)

**Pull models:**
```bash
ollama pull mistral
ollama pull llama2
ollama pull neural-chat
```

---

## 🔗 Important Links

| Service | URL |
|---------|-----|
| **Ollama** | https://ollama.ai |
| **Pinecone** | https://app.pinecone.io |
| **LangChain** | https://python.langchain.com |
| **Gradio** | https://gradio.app |

---

## ❌ Troubleshooting

| Error | Solution |
|-------|----------|
| `Connection refused (Ollama)` | Run `ollama serve` in another terminal |
| `PINECONE_API_KEY not found` | Check `.env` file, make sure it's not empty |
| `No PDFs found` | Add PDF files to `data/raw/` |
| `Index not found in Pinecone` | Create index in Pinecone dashboard |
| `Slow responses` | Use smaller models or reduce search results |

---

## 📂 Project Structure

```
.
├── src/
│   ├── config.py      ← Configuration
│   ├── ingest.py      ← Process documents
│   └── chat.py        ← Chat interface
├── data/raw/          ← Your PDFs go here
├── .env               ← Your secrets
├── requirements.txt   ← Dependencies
├── README.md          ← Full documentation
└── SETUP_GUIDE.md     ← Detailed guide
```

---

## 🎯 Workflow

```
1. ollama serve                    ← Start Ollama
2. source .venv/bin/activate      ← Activate Python
3. python -m src.ingest           ← Process documents
4. python -m src.chat             ← Start web interface
5. Open http://localhost:7860     ← Chat!
```

---

## 💰 Cost Breakdown

| Service | Cost |
|---------|------|
| **Ollama** | Free (local) |
| **Pinecone** | Free tier: 100M vectors |
| **Gradio** | Free (local) |
| **Total** | 💰 **Free!** |

---

## 📱 Web Interface

- **Address**: http://localhost:7860
- **Port**: 7860 (change in chat.py if needed)
- **Protocol**: http (secure in production)

---

## 🐛 Debug Mode

To see more details, add to the top of `src/chat.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🎓 Learning Resources

1. **Understanding RAG**: https://python.langchain.com/docs/use_cases/question_answering/
2. **Pinecone Quickstart**: https://docs.pinecone.io/quickstart
3. **Ollama Models**: https://ollama.ai/library
4. **Gradio Tutorial**: https://gradio.app/quickstart/

---

Last Updated: December 2024

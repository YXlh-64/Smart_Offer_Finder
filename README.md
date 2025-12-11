# Smart Offer Finder

An end-to-end Retrieval Augmented Generation (RAG) system for intelligent document search and Q&A. It surfaces relevant offers, conventions, and operational guides through a conversational AI interface. Everything runs locally or with free services—no expensive APIs required.

## ✨ Key Features

- **Local Embeddings**: Uses Ollama with multilingual-e5-base for semantic search
- **Vector Database**: Pinecone for scalable and fast similarity search
- **Local LLM**: Uses Ollama (free) or any API-based LLM (OpenAI, etc.)
- **Web Interface**: Gradio provides an intuitive chat UI
- **Source Citation**: Automatically cites documents used to answer questions
- **Fully Configurable**: Easy to swap models and services

## 📦 What You Get

- **Modular Architecture**: Separate config, ingest, and chat modules
- **Production-Ready**: Error handling, logging, and graceful degradation
- **Document Ingestion**: PDF loading with intelligent chunking
- **Conversation Memory**: Context-aware multi-turn conversations
- **Source Tracking**: Know which documents informed each answer

## 📁 Repository Layout

```
Smart_Offer_Finder/
├── src/
│   ├── __init__.py
│   ├── config.py          # Settings from environment variables
│   ├── ingest.py          # Load PDFs → chunk → embed → Pinecone
│   └── chat.py            # Gradio chat interface + RAG chain
├── data/
│   ├── raw/               # Your PDF files go here
│   └── vectorstore/       # (unused with Pinecone)
├── .env                   # Your secrets (DO NOT COMMIT)
├── .env.example           # Template
├── requirements.txt       # Python dependencies
├── setup.fish             # Quick setup script (Linux/macOS)
├── SETUP_GUIDE.md         # Detailed setup instructions
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Install Ollama

Download from [ollama.ai](https://ollama.ai) and install.

### 2. Pull Embedding Model

```bash
ollama pull multilingual-e5-base
```

### 3. Create Pinecone Account

- Sign up at [pinecone.io](https://pinecone.io)
- Create an index named `smart-offer-finder` with 384 dimensions
- Get your API key and environment

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and add:
# - PINECONE_API_KEY
# - PINECONE_ENVIRONMENT
```

### 5. Setup & Install

```bash
./setup.fish              # Linux/macOS with fish shell
# OR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6. Add Documents

Place your PDFs in `data/raw/`:
```bash
cp path/to/offers.pdf data/raw/
cp path/to/conventions.pdf data/raw/
```

### 7. Ingest to Pinecone

```bash
python -m src.ingest
```

### 8. Run Chat Interface

In terminal 1 (keep running):
```bash
ollama serve
```

In terminal 2:
```bash
python -m src.chat
```

Open browser to: **http://localhost:7860**

## 🔧 Configuration

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `PINECONE_API_KEY` | Pinecone authentication | `pcak-...` |
| `PINECONE_INDEX_NAME` | Index name | `smart-offer-finder` |
| `PINECONE_ENVIRONMENT` | Pinecone region | `us-east-1-abc1d23` |
| `OLLAMA_BASE_URL` | Ollama server | `http://localhost:11434` |
| `EMBEDDING_MODEL` | Embedding model | `ollama/multilingual-e5-base` |
| `LLM_MODEL` | Chat model | `ollama/mistral` or `gpt-4` |
| `CHUNK_SIZE` | Doc chunk size | `800` |
| `CHUNK_OVERLAP` | Chunk overlap | `120` |

### Using Different Models

**Ollama models** (local, free):
```bash
ollama pull mistral           # Fast, capable
ollama pull neural-chat       # Smaller, efficient
ollama pull llama2            # Larger, slower
```

**API-based models** (OpenAI, etc.):
```env
LLM_MODEL=gpt-4
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
```

## 📖 Full Documentation

For detailed setup and troubleshooting, see **[SETUP_GUIDE.md](SETUP_GUIDE.md)**.

Topics covered:
- ✅ Step-by-step Ollama installation
- ✅ Pinecone account setup
- ✅ Environment configuration
- ✅ Data ingestion process
- ✅ Running the application
- ✅ Troubleshooting common issues
- ✅ Advanced model selection

## 🏗️ How It Works

```
User Question
     ↓
  [Gradio UI]
     ↓
  [LangChain RAG Pipeline]
     ↓
  [Ollama: Question → Vector]
     ↓
  [Pinecone: Find Similar Docs]
     ↓
  [Retrieved Documents]
     ↓
  [Ollama/ChatGPT: Generate Answer]
     ↓
  [Answer + Sources] → UI
```

### Components

| Component | Role | Technology |
|-----------|------|-----------|
| **Embeddings** | Convert text to vectors | Ollama (multilingual-e5-base) |
| **Vector DB** | Store & search vectors | Pinecone |
| **LLM** | Generate answers | Ollama or OpenAI |
| **Web UI** | Chat interface | Gradio |
| **Orchestration** | Tie it all together | LangChain |

## 💡 Customization

### Add Business Logic

Edit `src/chat.py` to add custom prompts:

```python
system_prompt = """You are an intelligent assistant for Algeria Telecom.
- Only answer based on provided documents
- If unsure, say 'I don't have that information'
- Always cite sources
"""
```

### Change Chunk Size

Edit `.env`:
```env
CHUNK_SIZE=1000     # Larger chunks for broad context
CHUNK_OVERLAP=150   # More overlap for edge cases
```

### Add Metadata Filtering

Modify `src/ingest.py` to attach metadata to chunks, then filter in retriever.

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| "Connection refused" | Run `ollama serve` in separate terminal |
| "PINECONE_API_KEY not found" | Add API key to `.env` |
| "No PDFs found" | Add files to `data/raw/` |
| "Slow responses" | Use smaller model: `ollama/neural-chat` |
| "High memory" | Reduce model size or close other apps |

See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for more troubleshooting.

## 🔒 Security

- ✅ `.env` is in `.gitignore` (secrets stay safe)
- ✅ API keys never logged or exposed
- ✅ Local LLM option (no external API calls)
- ✅ Document access through Pinecone only

## 🎯 Next Steps

1. Follow the [Quick Start](#-quick-start)
2. Read [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed help
3. Customize prompts for your use case
4. Add your documents and test
5. Deploy to production (with proper security)

## 📚 Resources

- [Ollama Documentation](https://github.com/jmorganca/ollama)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Gradio Documentation](https://gradio.app/)

## 📝 License

This project is provided as-is for the Smart Offer Finder challenge.

---

**Ready to get started?** Follow the [Quick Start](#-quick-start) above or read [SETUP_GUIDE.md](SETUP_GUIDE.md) for comprehensive instructions.
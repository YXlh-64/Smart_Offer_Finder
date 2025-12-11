# Smart Offer Finder - Complete Setup Guide

This guide will help you set up the Smart Offer Finder RAG system with Ollama embeddings and Pinecone vector database.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Install Ollama](#step-1-install-ollama)
3. [Step 2: Set Up Pinecone](#step-2-set-up-pinecone)
4. [Step 3: Configure Environment](#step-3-configure-environment)
5. [Step 4: Install Python Dependencies](#step-4-install-python-dependencies)
6. [Step 5: Prepare Data](#step-5-prepare-data)
7. [Step 6: Ingest Data to Pinecone](#step-6-ingest-data-to-pinecone)
8. [Step 7: Run the Application](#step-7-run-the-application)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.10+** - Download from [python.org](https://www.python.org/)
- **Ollama** - For local embeddings and LLM inference
- **Pinecone Account** - Free tier available at [pinecone.io](https://www.pinecone.io/)
- **Basic knowledge** of terminal/command line

---

## Step 1: Install Ollama

### What is Ollama?
Ollama allows you to run large language models and embeddings locally on your machine without needing API keys.

### Installation

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
Download and install from [ollama.ai](https://ollama.ai/download/mac)

**Windows:**
Download and install from [ollama.ai](https://ollama.ai/download/windows)

### Pull the Embedding Model

After installation, open a terminal and run:

```bash
ollama pull multilingual-e5-base
```

This downloads the multilingual embedding model (~500MB). You only need to do this once.

### Verify Installation

```bash
ollama list
```

You should see `multilingual-e5-base` in the list.

---

## Step 2: Set Up Pinecone

### Create a Pinecone Account

1. Go to [pinecone.io](https://www.pinecone.io/)
2. Sign up for a free account
3. Complete email verification
4. Navigate to the **API Keys** section

### Create Your API Key

1. In the Pinecone dashboard, go to **API Keys**
2. Click **Create API Key**
3. Copy your API key (you'll need this in the next section)

### Create an Index

1. Go to **Indexes** in the Pinecone dashboard
2. Click **Create Index**
3. Use these settings:
   - **Index Name**: `smart-offer-finder`
   - **Dimensions**: `384` (for multilingual-e5-base embeddings)
   - **Metric**: `cosine`
   - **Serverless Config**: Select your preferred region (e.g., `us-east-1`)
4. Click **Create Index**

Wait for the index to become ready (usually takes 1-2 minutes).

### Find Your Environment

Your **environment** is shown next to your index name. It typically looks like: `us-east-1-abc1d23` or similar. You'll need this in the configuration.

---

## Step 3: Configure Environment

### Copy Environment Template

```bash
cp .env.example .env
```

Or create a `.env` file with the following content:

```properties
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=smart-offer-finder
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=ollama/multilingual-e5-base

# LLM Configuration (using Ollama - completely free and local)
LLM_MODEL=ollama/mistral

# Optional: Use API-based LLM instead (requires keys)
# LLM_MODEL=gpt-4
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=your_openai_api_key_here

# Ingestion Configuration
VECTORSTORE_PATH=data/vectorstore
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

### Fill in Your Values

Replace the placeholder values:
- `your_pinecone_api_key_here` → Your actual Pinecone API key from Step 2
- `your_pinecone_environment_here` → Your Pinecone environment (e.g., `us-east-1-abc1d23`)

**⚠️ Security Note:** Never commit `.env` to version control. It's already in `.gitignore`.

---

## Step 4: Install Python Dependencies

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- LangChain and related packages for RAG
- Pinecone client
- Gradio for web UI
- Ollama integration
- PDF loader for document processing

---

## Step 5: Prepare Data

### Add Your PDFs

1. Create the data directory if it doesn't exist:
   ```bash
   mkdir -p data/raw
   ```

2. Place your PDF files in `data/raw/`:
   - Offers documents
   - Convention documents
   - Guides and handbooks
   - Any other relevant PDFs

Example structure:
```
data/
└── raw/
    ├── offers.pdf
    ├── conventions.pdf
    ├── operational_guide.pdf
    └── other_documents.pdf
```

**Supported formats:** PDF files only

---

## Step 6: Ingest Data to Pinecone

### Start Ollama (if not already running)

Open a **separate terminal** and run:

```bash
ollama serve
```

This starts the Ollama server and keeps it running. You'll see output like:
```
binding 0.0.0.0:11434
```

**Keep this terminal open while using the application.**

### Run Ingestion

In your main terminal (with `.venv` activated), run:

```bash
python -m src.ingest
```

You should see output like:
```
[ingest] Processing documents...
[ingest] Splitting documents into chunks...
[ingest] Creating embeddings with Ollama...
[ingest] Uploading to Pinecone...
Successfully indexed 145 chunks to Pinecone index 'smart-offer-finder'.
```

This process:
1. Reads all PDFs from `data/raw/`
2. Splits them into chunks (800 characters with 120 overlap)
3. Generates embeddings using Ollama's multilingual-e5-base
4. Uploads vectors to your Pinecone index

**⏱️ Time:** Depends on document size. Usually 2-5 minutes for 10-20 MB of PDFs.

---

## Step 7: Run the Application

### Start the Gradio Interface

Make sure Ollama is still running in the separate terminal, then run:

```bash
python -m src.chat
```

You should see:
```
Initializing Smart Offer Finder...
✅ Chain initialized successfully!
🚀 Launching Gradio interface...
Running on http://0.0.0.0:7860
```

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:7860
```

You should see the Smart Offer Finder chat interface.

### Start Chatting

1. Type your question in the text box
2. Click **Send** or press Enter
3. The chatbot will:
   - Retrieve relevant documents from Pinecone
   - Generate an answer using the Ollama LLM
   - Display sources at the bottom

---

## Architecture Overview

```
User Question
     ↓
[Gradio UI]
     ↓
[LangChain] ← Orchestrates the RAG pipeline
     ↓
[Ollama Embeddings] ← Converts question to vector locally
     ↓
[Pinecone] ← Searches for similar documents
     ↓
[Retrieved Documents]
     ↓
[Ollama LLM] ← Generates answer locally
     ↓
Answer + Sources
     ↓
[Gradio UI] ← Displays to user
```

---

## Key Components

| Component | Purpose | Setup |
|-----------|---------|-------|
| **Ollama** | Local embeddings & LLM | `ollama serve` |
| **Pinecone** | Vector database | Free account + API key |
| **LangChain** | RAG framework | Installed via pip |
| **Gradio** | Web interface | Installed via pip |

---

## Troubleshooting

### Issue: "Connection refused" when starting chat

**Solution:** Make sure Ollama is running in a separate terminal:
```bash
ollama serve
```

### Issue: "PINECONE_API_KEY not found"

**Solution:** Check your `.env` file:
```bash
cat .env
```

Make sure `PINECONE_API_KEY` is set to your actual API key (not a placeholder).

### Issue: "Pinecone index not found"

**Solution:** 
1. Verify the index name in `.env` matches your Pinecone index
2. Check that the index is **Ready** in Pinecone dashboard
3. Re-run ingestion: `python -m src.ingest`

### Issue: "No documents in data/raw"

**Solution:** Add PDF files to the `data/raw/` directory:
```bash
ls data/raw/
```

You should see `.pdf` files. If empty, copy your PDFs there.

### Issue: Slow response times

**Causes:**
- Ollama model still downloading (first run)
- Network latency with Pinecone
- Large document retrieval

**Solutions:**
- Use smaller models: `ollama pull mistral` instead of larger ones
- Reduce search results: Edit `chat.py` line 56 from `{"k": 4}` to `{"k": 2}`

### Issue: "Embedding model not found"

**Solution:** Pull the model explicitly:
```bash
ollama pull multilingual-e5-base
```

Verify it's installed:
```bash
ollama list
```

### Issue: High memory usage

**Solution:** 
1. Reduce model size: Use `ollama/orca-mini` instead of `mistral`
2. Close other applications
3. Check available RAM: `free -h` (Linux) or Task Manager (Windows)

### Issue: Python version error

**Solution:** Check your Python version:
```bash
python --version
```

Ensure it's **3.10 or higher**. If not, install Python 3.10+ and create a new virtual environment.

---

## Advanced Configuration

### Use Different Embedding Model

Edit `.env`:
```properties
EMBEDDING_MODEL=ollama/nomic-embed-text
```

Then pull the model:
```bash
ollama pull nomic-embed-text
```

### Use Different LLM

Pull a different model and set it in `.env`:
```bash
ollama pull neural-chat    # Smaller, faster
ollama pull mistral        # Default
ollama pull llama2         # Larger, slower
ollama pull openchat       # Good balance
```

Then update `.env`:
```properties
LLM_MODEL=ollama/neural-chat
```

### Use API-Based LLM (OpenAI, Claude, etc.)

Edit `.env`:
```properties
LLM_MODEL=gpt-4
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...your_key_here...
```

---

## Project Structure

```
Smart_Offer_Finder/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration from environment
│   ├── ingest.py           # Data ingestion to Pinecone
│   └── chat.py             # Gradio chat interface
├── data/
│   ├── raw/                # Your PDF files
│   └── vectorstore/        # (Unused with Pinecone)
├── .env                    # Your secrets (DO NOT COMMIT)
├── .env.example            # Template (safe to commit)
├── requirements.txt        # Python dependencies
├── README.md               # Original README
└── SETUP_GUIDE.md         # This file
```

---

## Next Steps

1. ✅ Install Ollama and pull `multilingual-e5-base`
2. ✅ Create Pinecone account and get API key
3. ✅ Configure `.env` with your credentials
4. ✅ Install Python dependencies: `pip install -r requirements.txt`
5. ✅ Add PDF files to `data/raw/`
6. ✅ Ingest data: `python -m src.ingest`
7. ✅ Run chat: `python -m src.chat`
8. ✅ Open browser to `http://localhost:7860`

---

## Support & Resources

- **Ollama Docs:** https://github.com/jmorganca/ollama
- **Pinecone Docs:** https://docs.pinecone.io/
- **LangChain Docs:** https://python.langchain.com/
- **Gradio Docs:** https://gradio.app/
- **Available Ollama Models:** https://ollama.ai/library

---

## Environment Variables Reference

```properties
# REQUIRED: Pinecone
PINECONE_API_KEY=sk-...                    # Your Pinecone API key
PINECONE_INDEX_NAME=smart-offer-finder     # Index name to use
PINECONE_ENVIRONMENT=us-east-1-abc1d23     # Your Pinecone environment

# REQUIRED: Ollama
OLLAMA_BASE_URL=http://localhost:11434     # Ollama server address
EMBEDDING_MODEL=ollama/multilingual-e5-base # Embedding model name

# REQUIRED: LLM (choose one)
# Option 1: Local Ollama
LLM_MODEL=ollama/mistral

# Option 2: API-based (uncomment and fill)
# LLM_MODEL=gpt-4
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=sk-...

# OPTIONAL: Tuning
CHUNK_SIZE=800             # Document chunk size
CHUNK_OVERLAP=120          # Overlap between chunks
VECTORSTORE_PATH=data/vectorstore  # (Unused with Pinecone)
```

---

Happy chatting! 🚀

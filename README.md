# Smart Offer Finder

An end-to-end Retrieval Augmented Generation (RAG) starter for the "SMART OFFER FINDER" challenge. It targets a lightweight but production-ready path to build a chatbot that surfaces relevant offers, conventions, and operational guides for Algeria Telecom agents. Everything is in English for clarity; adjust prompts and UI text as needed.

## What you get
- Opinionated RAG baseline with LangChain, FAISS, and FastAPI.
- Simple ingest script for PDFs (offers, conventions, NGBSS guide, catalogue).
- FastAPI endpoint for chat with grounded answers and cited sources.
- Clear folder structure and env template to plug in your own data/keys.

## Repository layout
- `data/raw/` — drop your PDFs here (offers, conventions, guides).
- `data/vectorstore/` — serialized FAISS index (created after ingest).
- `src/` — Python code (config, ingest, chat API).
- `.env.example` — copy to `.env` and fill secrets.
- `requirements.txt` — minimal dependencies.

## Prerequisites
- Python 3.10+ recommended.
- **Ollama** (free, lightweight embedding engine). Install from [ollama.ai](https://ollama.com/download), then pull the embedding model:
  ```bash
  ollama pull nomic-embed-text
  ollama serve  # Start in a separate terminal
  ```
- OpenAI API key or OpenRouter key (for the LLM chat model). The embeddings are handled locally via Ollama, so no API calls there.
- Basic build tools to install Python wheels (for `faiss-cpu`).

## Quickstart
1) Start Ollama (in a separate terminal)

```bash
ollama serve
```

2) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Configure environment

```bash
cp .env.example .env
# Fill OPENAI_API_KEY with your OpenRouter or OpenAI key
# EMBEDDING_MODEL defaults to nomic-embed-text (local via Ollama)
# OLLAMA_BASE_URL defaults to http://localhost:11434
# Adjust LLM_MODEL as needed (default: gpt-4o via OpenRouter)
```

4) Add data

- Place all challenge PDFs under `data/raw/`. Example:
	- `Offres_Fixe_Fibre.pdf`
	- `Conventions_Partenaires.pdf`
	- `Guide_NGBSS.pdf`
	- `Catalogue_Interconnexion_AT.pdf`

5) Build the vector store

```bash
python -m src.ingest
```

This will chunk the PDFs and embed them using Ollama's `nomic-embed-text`, then save a FAISS index at `data/vectorstore/faiss_index`.

6) Run the chatbot API

```bash
python -m src.chat
# Visit http://localhost:8000/docs to try the `/chat` endpoint
```

7) Example request (once the server is running)

```bash
curl -X POST http://localhost:8000/chat \
	-H "Content-Type: application/json" \
	-d '{"question": "What offer fits a small business needing 50 Mbps and VoIP?"}'
```

## How it works
- **Embeddings (Ollama)**: Uses `nomic-embed-text` model running locally via Ollama. Fast, free, and no API calls needed.
- **Ingest (`src/ingest.py`)**: Loads PDFs, splits with `RecursiveCharacterTextSplitter`, embeds with Ollama, and saves a FAISS index.
- **Serve (`src/chat.py`)**: FastAPI + `ConversationalRetrievalChain` (ChatOpenAI/OpenRouter LLM + FAISS retriever + short-term memory). Returns answer plus source file hints.

## Adapting to the challenge
- **Prompting**: Add business guardrails (e.g., "Only answer with information from Algeria Telecom offers; if unsure, say you do not know").
- **Metadata**: Extend ingestion to capture product names, dates, and pricing fields in `metadata` for better filtering.
- **French-first UX**: Switch the user-facing prompt, answers, and UI copy to French. Keep system prompts focused on grounding and refusal when unsure.
- **Relevance tuning**: Adjust `k` in the retriever and chunk sizes for your corpus; smaller chunks help precision, larger help recall.
- **Evaluation**: Create a small set of Q/A pairs from real agent questions; use them to manually test or with `ragas` style checks.

## Minimal design notes for a web UI (not included)
- Keep a clear separation: static frontend hits the FastAPI backend. Use a simple chat layout with cited sources.
- Provide quick filters (e.g., by segment: entreprise, particulier, administration) mapped to metadata filters in the retriever.
- Add a "hallucination guard" banner reminding agents to verify critical details (tarifs, durées, exclusions) against the cited PDF page.

## Common tweaks
- Change LLM: set `LLM_MODEL` in `.env` (e.g., `gpt-4o`, `gpt-4o-mini`, or any OpenRouter-exposed model). For OpenRouter, set `OPENAI_BASE_URL=https://openrouter.ai/api/v1` and put your OpenRouter key in `OPENAI_API_KEY`.
- Change embedding model: run `ollama pull <model>` (e.g., `ollama pull nomic-embed-text:latest` or `ollama pull mxbai-embed-large`), then set `EMBEDDING_MODEL=<model>` in `.env`. Ensure Ollama is serving before ingesting.
- Faster iteration: during ingest, temporarily set `CHUNK_SIZE=1200` and `CHUNK_OVERLAP=80` to cut chunk count, then tighten later for quality.
- If Ollama is not on localhost: update `OLLAMA_BASE_URL` in `.env` (e.g., `http://192.168.1.10:11434`).

## Testing ideas
- After ingest, inspect `data/vectorstore/faiss_index` size and log output to confirm chunk count.
- Hit `/health` to ensure the chain is initialized.
- Ask contrastive questions to verify grounding: "What differs between offer A and B for SMEs?" and ensure sources are returned.

## Troubleshooting
- `No PDF files found`: add files to `data/raw/` and rerun ingest.
- `Vector store not found`: run `python -m src.ingest` before starting the API.
- `[ingest] failed: Connection error to Ollama`: ensure Ollama is running (`ollama serve`) on the address specified in `OLLAMA_BASE_URL`.
- `Connection refused` on `http://localhost:11434`: start Ollama in a separate terminal with `ollama serve`.
- Embedding model not found: run `ollama pull nomic-embed-text` (or the model specified in `EMBEDDING_MODEL`).

## Next steps (suggested)
- Add UI (Streamlit, React, or a simple HTML page) that calls `/chat`.
- Log interactions and top retrieved chunks to spot gaps in coverage.
- Add lightweight eval scripts (e.g., retrieval precision @k against a small labeled set).
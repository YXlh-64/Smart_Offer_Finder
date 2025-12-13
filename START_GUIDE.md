# Smart Offer Finder - Quick Start Guide

## Prerequisites
- Python 3.8+ installed
- Node.js 18+ installed
- Your documents already ingested in ChromaDB (data/chroma_db/)

## Starting the Application

### 1. Start the Backend (FastAPI)

Open a terminal and run:

```bash
cd C:\Users\Hachem\Documents\GitHub\Smart_Offer_Finder
python main.py
```

The backend will start on: **http://localhost:8000**

You should see:
- ✅ Chain initialized successfully on startup
- Server running on http://0.0.0.0:8000

### 2. Start the Frontend (React + Vite)

Open a **new terminal** and run:

```bash
cd C:\Users\Hachem\Documents\GitHub\Smart_Offer_Finder\frontend
npm run dev
```

The frontend will start on: **http://localhost:8080**

### 3. Access the Application

Open your browser and go to: **http://localhost:8080**

You should see:
- ✅ White background with clean UI
- ✅ Blue sidebar on the left
- ✅ "Connecté" toast notification (green)

## Architecture

```
Frontend (React/Vite) → Port 8080
    ↓ (API calls via /api/*)
Backend (FastAPI) → Port 8000
    ↓ (uses data from)
./data/Convention/    - Convention documents
./data/Offres/        - Offer documents  
./data/chroma_db/     - Vector database
./src/chat.py         - RAG chain logic
./src/hybrid_retriever.py - Hybrid search
./src/semantic_cache.py   - Caching
```

## How It Works

1. **User asks a question** in the frontend chat
2. **Frontend sends** POST request to `/api/chat/stream`
3. **Backend processes** using:
   - Hybrid retriever (BM25 + vector search)
   - Reranker for better results
   - LLM generates answer
   - Semantic cache for faster responses
4. **Frontend receives** streaming response with:
   - Answer chunks (displayed in real-time)
   - Source citations (PDF documents)

## Troubleshooting

### Backend won't start
- Check if ChromaDB exists: `data/chroma_db/chroma.sqlite3`
- If not, run: `python src/ingest.py` to ingest documents first

### Frontend shows "Backend non connecté"
- Make sure backend is running on port 8000
- Check: http://localhost:8000/health
- Should return: `{"status":"ok","chain_initialized":true}`

### Port already in use
- Backend: Change port in `main.py` (line 343)
- Frontend: Change port in `frontend/vite.config.ts` (line 9)

## Testing the Connection

1. Backend health check:
```bash
curl http://localhost:8000/health
```

2. Test a query:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Quelles sont les offres Idoom?","session_id":"test"}'
```

## Color Scheme (Preserved)
- **Background**: White (#FFFFFF)
- **Sidebar**: Blue gradient (#2563EB to #1D4ED8)
- **Primary**: Blue (#2563EB)
- **Text**: Dark gray (#0F172A)
- **Borders**: Light gray (#E2E8F0)

The UI design and colors are preserved from your original project!

# Streaming Implementation Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMART OFFER FINDER SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

                              FRONTEND                                
                         ┌──────────────────┐
                         │    React App     │
                         │  ChatApp.jsx     │
                         └────────┬─────────┘
                                  │
                    POST /chat/stream (Fetch API)
                                  │
                    Streams JSON lines over SSE
                                  │
        ┌─────────────────────────▼──────────────────────────┐
        │                                                      │
        │               BACKEND - FastAPI (main.py)           │
        │                                                      │
        │  ┌──────────────────────────────────────────────┐   │
        │  │  /health          - Health check              │   │
        │  │  /chat            - Non-streaming endpoint    │   │
        │  │  /chat/stream     - Streaming endpoint (SSE)  │   │
        │  └──────────────────────────────────────────────┘   │
        │                                                      │
        │  ┌──────────────────────────────────────────────┐   │
        │  │     stream_chat_response() Function           │   │
        │  │  - Calls chain.invoke()                       │   │
        │  │  - Streams response chunks                    │   │
        │  │  - Sends sources metadata                     │   │
        │  │  - Handles errors gracefully                  │   │
        │  └──────────────────────────────────────────────┘   │
        │                                                      │
        └──────────────────┬───────────────────────────────────┘
                           │
        ┌──────────────────▼───────────────────────────────┐
        │          RAG Pipeline (Existing)                 │
        │                                                  │
        │  ┌─────────────────────────────────────────┐   │
        │  │  ChromaDB Vector Store                   │   │
        │  │  (Retrieval)                             │   │
        │  └──────────────┬──────────────────────────┘   │
        │                 │                               │
        │                 ├── BGE Reranker (if enabled)  │
        │                 │                               │
        │  ┌──────────────▼──────────────────────────┐   │
        │  │  LLM (Deepseek or Ollama)                │   │
        │  │  (Generation)                            │   │
        │  └──────────────────────────────────────────┘   │
        │                                                  │
        └──────────────────────────────────────────────────┘
```

## Streaming Flow Diagram

```
User Types Question
        │
        ▼
┌────────────────────┐
│ Frontend Sends     │ POST /chat/stream
│ Fetch Request      │
└────────┬───────────┘
         │
         ▼
┌────────────────────────────────┐
│ Backend Initializes Response   │
│ (stream_chat_response)         │
└────────┬───────────────────────┘
         │
         ▼
    ┌────────────┐
    │ Invoke     │ Retrieval + LLM
    │ Chain      │
    └────┬───────┘
         │
         ▼
┌────────────────────────────────┐
│ Generate Response Text         │
│ Split into Chunks              │
└────────┬───────────────────────┘
         │
         ├──► {"type": "chunk", "content": "..."}
         │
         ├──► {"type": "chunk", "content": "..."}
         │
         ├──► {"type": "chunk", "content": "..."}
         │
         ├──► {"type": "sources", "content": [...]}
         │
         └──► {"type": "complete"}
              │
              ▼
         ┌────────────────────────┐
         │ Frontend Receives Each │
         │ Chunk and Updates UI   │
         │ in Real-Time           │
         └────────────────────────┘
```

## Message Update Sequence

```
Frontend:                          Backend:
                                  
User: "What offers?"               
  │                                
  └──► Fetch Request ──────────────► Listen on /chat/stream
  │                                │
  │                  Response.body.getReader()
  │                  ◄─────────────┤
  │                                │
  │◄──── chunk 1 ─────────────────┤ "The available..."
  │  Update: "The available..."     │
  │                                │
  │◄──── chunk 2 ─────────────────┤ "The available... offers"
  │  Update: "The available... offers"
  │                                │
  │◄──── chunk 3 ─────────────────┤ "The available... offers include..."
  │  Update: "The available... offers include..."
  │                                │
  │◄──── sources ──────────────────┤ ["doc1.pdf", "doc2.pdf"]
  │  Display sources              │
  │                                │
  │◄──── complete ─────────────────┤ End of stream
  │  Stop loading animation        │
  │                                │
  ✓ Response fully displayed
```

## Data Flow

```
REQUEST
────────

Client
  │
  └─→ POST /chat/stream
      │
      └─→ {
          "question": "What are available offers?",
          "session_id": "default"
        }

STREAMING RESPONSE
──────────────────

Backend Stream (Server-Sent Events)
  │
  ├─→ {"type": "chunk", "content": "The available"}
  │
  ├─→ {"type": "chunk", "content": "The available offers"}
  │
  ├─→ {"type": "chunk", "content": "The available offers include..."}
  │
  ├─→ {"type": "sources", "content": ["document1.pdf", "document2.pdf"]}
  │
  └─→ {"type": "complete"}

CLIENT RECEIVES
───────────────

Message State Evolution:
  1. {text: "", sources: []} (initial)
  2. {text: "The available", sources: []}
  3. {text: "The available offers", sources: []}
  4. {text: "The available offers include...", sources: []}
  5. {text: "The available offers include...", sources: ["doc1.pdf", "doc2.pdf"]}
  ✓ Done, stop loading
```

## Comparison: Before vs After

```
BEFORE (Non-Streaming)              AFTER (Streaming)
─────────────────────────           ─────────────────

User waits...                        User sees response
    │                                appearing in real-time
    │                                
    ▼                                  ▼
[Loading Spinner]                    "The available..."
    │                                  │
    │ (Full response generation       ▼
    │  happens here)                 "The available 
    │                                 offers..."
    │                                  │
    ▼                                  ▼
✓ Full response arrives              "The available 
                                      offers include..."
                                      │
                                      ▼
                                     ✓ Complete

Perceived Time: Slow                 Perceived Time: Fast
User Experience: Waiting             User Experience: 
                                     Interactive
```

## Technology Stack

```
Frontend:
  • React 18+
  • Fetch API (native)
  • CSS for styling
  
Backend:
  • FastAPI 0.104+
  • Uvicorn ASGI server
  • Python 3.8+
  
RAG Components (Existing):
  • ChromaDB (Vector Store)
  • LangChain (Orchestration)
  • Ollama/Deepseek (LLM)
  • BGE Reranker (Optional)

Communication:
  • Server-Sent Events (SSE)
  • JSON over HTTP
```

## Files Changed Summary

```
NEW FILES:
  ✓ main.py                           FastAPI application
  ✓ STREAMING_GUIDE.md               Complete documentation
  ✓ STREAMING_QUICK_START.md         Quick reference
  ✓ start.fish                       Startup script

MODIFIED FILES:
  ✓ requirements.txt                 Added: fastapi, uvicorn
  ✓ frontend/src/ChatApp.jsx        Updated to stream responses
  ✓ This file (Architecture diagram)
```

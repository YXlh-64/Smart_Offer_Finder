"""
FastAPI application with streaming support for Smart Offer Finder.
"""

import sys
from pathlib import Path
from typing import AsyncGenerator
import asyncio
import json
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chat import initialize_chain, build_chain, get_settings, cached_chain_invoke
from src import chat as chat_module  # Import the module to access its globals
from src.timing_visualizer import get_visualizer

# Note: We'll use chat_module.chain instead of a local chain variable


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize chain on startup"""
    
    try:
        # Initialize the chain using the initialize_chain function
        # This will set the global variables in src.chat module
        settings = get_settings()
        chat_module.settings = settings
        chat_module.chain = build_chain(settings)
        print("✅ Chain initialized successfully on startup")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Cleanup on shutdown (if needed)
    print("🛑 Shutting down...")


# FastAPI app setup with lifespan
app = FastAPI(title="Smart Offer Finder API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chain and settings
chain = None
settings = None
chat_memory = {}  # Store conversation memory per session


class ChatMessage(BaseModel):
    """Chat message model"""
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: list[str] = []


async def stream_chat_response(question: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    """
    Stream chat response token by token.
    
    Args:
        question: User's question
        session_id: Unique session identifier
        
    Yields:
        JSON strings containing chunks of the response or metadata
    """
    import time
    
    try:
        if chat_module.chain is None:
            yield json.dumps({
                "type": "error",
                "content": "Chat interface not initialized. Please ensure ChromaDB is properly configured and documents have been ingested."
            }) + "\n"
            return
        
        # Start timing the chain invocation
        chain_start = time.time()
        
        # Invoke the chain with semantic caching
        result = cached_chain_invoke(question)
        
        chain_time = (time.time() - chain_start) * 1000  # Convert to milliseconds
        
        # Check if it was a cache hit
        is_cache_hit = result.get("cache_hit", False)
        
        answer = result.get("answer", "No response generated.")
        source_docs = result.get("source_documents", []) or []
        sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
        
        # Extract timing information from the retriever if available
        retrieval_time = 0.0
        reranking_time = 0.0
        generation_time = 0.0
        
        try:
            retriever = chat_module.chain.retriever
            if hasattr(retriever, 'timing_data') and retriever.timing_data:
                retrieval_time = retriever.timing_data.get("vectorstore_search", 0)
                reranking_time = retriever.timing_data.get("reranking", 0)
                generation_time = chain_time - (retrieval_time + reranking_time)
                
                # Log timing information to console
                print("\n" + "="*80)
                print("⏱️  TIMING BREAKDOWN (Per Phase)")
                print("="*80)
                
                total = retrieval_time + reranking_time + generation_time
                
                # Calculate percentages
                retrieval_pct = (retrieval_time / total * 100) if total > 0 else 0
                reranking_pct = (reranking_time / total * 100) if total > 0 else 0
                generation_pct = (generation_time / total * 100) if total > 0 else 0
                
                # Create bar visualizations
                bar_width = 40
                retrieval_bar_len = int((retrieval_pct / 100) * bar_width)
                reranking_bar_len = int((reranking_pct / 100) * bar_width)
                generation_bar_len = int((generation_pct / 100) * bar_width)
                
                # Display bars
                retrieval_bar = "█" * retrieval_bar_len + "░" * (bar_width - retrieval_bar_len)
                print(f"🔍 Retrieval     [{retrieval_bar}] {retrieval_pct:>5.1f}%  ({retrieval_time:>7.2f}ms)")
                
                if reranking_time > 0:
                    reranking_bar = "█" * reranking_bar_len + "░" * (bar_width - reranking_bar_len)
                    print(f"🎯 Reranking     [{reranking_bar}] {reranking_pct:>5.1f}%  ({reranking_time:>7.2f}ms)")
                
                generation_bar = "█" * generation_bar_len + "░" * (bar_width - generation_bar_len)
                print(f"✨ Generation    [{generation_bar}] {generation_pct:>5.1f}%  ({generation_time:>7.2f}ms)")
                
                print("-" * 80)
                print(f"{'Total Time':<16} {'':>40} ({total:>7.2f}ms)")
                print("="*80 + "\n")
        except Exception as e:
            print(f"Warning: Could not extract timing data: {e}")
        
        # Stream the answer character by character preserving formatting
        # Send chunks to simulate streaming while preserving newlines and markdown
        chunk_size = 50  # Send 50 characters at a time
        full_text = ""
        
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            full_text += chunk
            
            # Send accumulated text
            yield json.dumps({
                "type": "chunk",
                "content": full_text  # Send full accumulated text with preserved formatting
            }) + "\n"
            await asyncio.sleep(0.03)  # Small delay for smoother streaming
        
        # Make sure the complete text is sent at the end
        if full_text != answer:
            yield json.dumps({
                "type": "chunk",
                "content": answer  # Send complete answer with all formatting
            }) + "\n"
        
        # Send sources at the end
        if sources:
            yield json.dumps({
                "type": "sources",
                "content": sources
            }) + "\n"
        
        # Send completion signal
        yield json.dumps({
            "type": "complete"
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({
            "type": "error",
            "content": f"Error processing message: {str(e)}"
        }) + "\n"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "chain_initialized": chat_module.chain is not None
    }


@app.post("/reload")
async def reload_chain():
    """Reload the chain with latest settings and code"""
    
    try:
        # Reload settings
        settings = get_settings()
        # Rebuild chain with new settings
        chat_module.settings = settings
        chat_module.chain = build_chain(settings)
        return {
            "status": "success",
            "message": "Chain reloaded successfully",
            "settings": {
                "llm_model": settings.llm_model,
                "use_reranker": settings.use_reranker,
                "max_tokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading chain: {str(e)}"
        )


@app.post("/chat")
async def chat(message: ChatMessage):
    """
    Non-streaming chat endpoint for backward compatibility.
    
    Returns complete response with sources.
    """
    
    if chat_module.chain is None:
        raise HTTPException(
            status_code=503,
            detail="Chat interface not initialized. Please ensure ChromaDB is properly configured and documents have been ingested."
        )
    
    try:
        result = cached_chain_invoke(message.question)
        answer = result.get("answer", "No response generated.")
        source_docs = result.get("source_documents", []) or []
        sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
        
        return ChatResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(message: ChatMessage):
    """
    Streaming chat endpoint.
    
    Streams response as server-sent events (SSE) with the following formats:
    - {"type": "chunk", "content": "partial response text"}
    - {"type": "sources", "content": ["source1", "source2"]}
    - {"type": "complete"}
    - {"type": "error", "content": "error message"}
    """
    
    if chat_module.chain is None:
        async def error_stream():
            yield json.dumps({
                "type": "error",
                "content": "Chat interface not initialized. Please ensure ChromaDB is properly configured and documents have been ingested."
            }) + "\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )
    
    return StreamingResponse(
        stream_chat_response(message.question, message.session_id),
        media_type="text/event-stream"
    )


@app.get("/stats/timing")
async def get_timing_stats():
    """
    Get timing statistics for all completed requests.
    
    Returns:
        - total_requests: Number of requests processed
        - average_timings: Average times for each phase
        - recent_timings: Last 10 request timings
    """
    visualizer = get_visualizer()
    
    avg_timings = visualizer.get_average_timings()
    recent = [t.to_dict() for t in visualizer.history[-10:]]
    
    return {
        "total_requests": len(visualizer.history),
        "average_timings": avg_timings,
        "recent_timings": recent,
        "statistics": {
            "slowest_request": max((t.total for t in visualizer.history), default=0),
            "fastest_request": min((t.total for t in visualizer.history), default=0),
        }
    }


@app.get("/stats/timing/export")
async def export_timing_stats():
    """
    Export timing statistics as JSON for analysis.
    
    Returns:
        Complete timing history with statistics
    """
    visualizer = get_visualizer()
    
    avg_timings = visualizer.get_average_timings()
    
    return {
        "export_timestamp": str(datetime.now().isoformat()),
        "total_records": len(visualizer.history),
        "timings": [t.to_dict() for t in visualizer.history],
        "average": avg_timings,
        "statistics": {
            "slowest_request": max((t.total for t in visualizer.history), default=0),
            "fastest_request": min((t.total for t in visualizer.history), default=0),
            "median_request": sorted([t.total for t in visualizer.history])[len(visualizer.history)//2] if visualizer.history else 0,
            "total_requests": len(visualizer.history)
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "Smart Offer Finder API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /reload": "Reload chain with latest settings",
            "POST /chat": "Non-streaming chat endpoint",
            "POST /chat/stream": "Streaming chat endpoint (Server-Sent Events)",
            "GET /stats/timing": "Get timing statistics",
            "GET /stats/timing/export": "Export timing data"
        },
        "usage": {
            "chat": {
                "description": "Send a question and get a complete response",
                "payload": {"question": "What are the available offers?", "session_id": "optional"}
            },
            "chat_stream": {
                "description": "Send a question and receive streaming response",
                "payload": {"question": "What are the available offers?", "session_id": "optional"},
                "response_format": "Server-Sent Events (SSE) with chunks"
            },
            "reload": {
                "description": "Reload the chain without restarting server",
                "method": "POST",
                "use_case": "After changing .env or code files"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Use import string format for auto-reload to work
    uvicorn.run(
        "main:app",  # Import string instead of app object
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload on code changes
    )

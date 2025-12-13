"""
FastAPI application with streaming support for Smart Offer Finder.
"""

import sys
from pathlib import Path
from typing import AsyncGenerator, Optional
import asyncio
import json
from datetime import datetime
import tempfile
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from urllib.parse import unquote
import mimetypes

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chat import initialize_chain, build_chain, get_settings, load_vectorstore, choose_embeddings, get_chain_for_language
from src.hybrid_retriever import detect_language
from src.timing_visualizer import get_visualizer
from src.history_manager import get_history_manager, HistoryManager

# FastAPI app setup
app = FastAPI(title="Smart Offer Finder API")

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
chain_fr = None  # French chain
chain_ar = None  # Arabic chain
settings = None
history_manager: HistoryManager = None
whisper_model = None
semantic_cache = None
embeddings = None


class ChatMessage(BaseModel):
    """Chat message model"""
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: list[str] = []


class TranscriptionResponse(BaseModel):
    """Transcription response model"""
    text: str
    language: str = "fr"


class SessionCreate(BaseModel):
    """Session creation model"""
    title: Optional[str] = "Nouvelle conversation"


class SessionUpdate(BaseModel):
    """Session update model"""
    title: str


class BatchQuestionsRequest(BaseModel):
    """Batch questions request model"""
    equipe: str
    question: dict[str, dict[str, str]]  # {"categorie_01": {"1": "question1", "2": "question2"}}


class BatchAnswersResponse(BaseModel):
    """Batch answers response model (without sources)"""
    equipe: str
    reponses: dict[str, dict[str, str]]  # {"categorie_01": {"1": "answer1", "2": "answer2"}}


class BatchAnswersWithSourcesResponse(BaseModel):
    """Batch answers response model (with sources)"""
    equipe: str
    reponses: dict[str, dict[str, dict]]  # {"categorie_01": {"1": {"answer": "...", "sources": [...]}}}



async def stream_chat_response(question: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    """
    Stream chat response token by token.
    
    Args:
        question: User's question
        session_id: Unique session identifier
        
    Yields:
        JSON strings containing chunks of the response or metadata
    """
    global chain, chain_fr, chain_ar, settings, history_manager, semantic_cache, embeddings
    import time
    
    try:
        if chain is None:
            yield json.dumps({
                "type": "error",
                "content": "Chat interface not initialized. Please ensure ChromaDB is properly configured and documents have been ingested."
            }) + "\n"
            return
        
        # Detect language and select appropriate chain
        query_language = detect_language(question)
        active_chain = chain_ar if query_language == 'ar' else chain_fr
        if active_chain is None:
            active_chain = chain  # Fallback
        print(f"  🌐 Query language: {query_language.upper()} → using {query_language.upper()} chain")
        
        # Ensure session exists, create if not
        if history_manager:
            session = history_manager.get_session(session_id)
            if not session:
                # Create session with first message as title
                title = history_manager.generate_title_from_message(question)
                history_manager.create_session(title)
                # Update the session ID to match the created one
                session = history_manager.get_session(session_id)
                if not session:
                    # If still doesn't exist, create with provided ID manually
                    from datetime import datetime
                    conn = history_manager._get_connection()
                    cursor = conn.cursor()
                    now = datetime.now().isoformat()
                    title = history_manager.generate_title_from_message(question)
                    cursor.execute(
                        "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (session_id, title, now, now)
                    )
                    conn.commit()
                    conn.close()
        
        # Start timing
        chain_start = time.time()
        
        # Check semantic cache first (if enabled)
        cache_hit = False
        if semantic_cache is not None and embeddings is not None:
            try:
                query_embedding = embeddings.embed_query(question)
                cached_result = semantic_cache.get(question, query_embedding)
                
                if cached_result is not None:
                    cache_hit = True
                    answer = cached_result["answer"]
                    sources = cached_result.get("sources", [])
                    cache_latency = cached_result.get("latency_ms", 0)
                    
                    print(f"✅ [Semantic Cache] HIT - serving cached response ({cache_latency:.2f}ms)")
                    
                    # Stream the cached answer
                    lines = answer.split('\n')
                    for line_idx, line in enumerate(lines):
                        if line.strip():
                            words = line.split(' ')
                            for word in words:
                                if word:
                                    yield json.dumps({
                                        "type": "chunk",
                                        "content": word + " "
                                    }) + "\n"
                                    await asyncio.sleep(0.01)  # Faster for cached responses
                        
                        if line_idx < len(lines) - 1:
                            yield json.dumps({
                                "type": "chunk",
                                "content": "\n"
                            }) + "\n"
                    
                    # Send sources
                    if sources:
                        yield json.dumps({
                            "type": "sources",
                            "content": sources
                        }) + "\n"
                    
                    # Save to history
                    if history_manager:
                        history_manager.save_turn(session_id, question, answer, sources if sources else None)
                    
                    # Send completion signal with cache indicator
                    yield json.dumps({
                        "type": "complete",
                        "cache_hit": True
                    }) + "\n"
                    return
                    
            except Exception as cache_error:
                print(f"⚠️ [Semantic Cache] Error checking cache: {cache_error}")
        
        # Cache miss - invoke the full chain (using language-appropriate chain)
        result = active_chain.invoke({"question": question})
        
        chain_time = (time.time() - chain_start) * 1000  # Convert to milliseconds
        
        answer = result.get("answer", "No response generated.")
        source_docs = result.get("source_documents", []) or []
        
        # Extract file paths for download links (prefer file_path, fallback to source)
        sources = []
        seen_paths = set()
        for doc in source_docs:
            file_path = doc.metadata.get("file_path", "") or doc.metadata.get("source", "unknown")
            if file_path not in seen_paths:
                seen_paths.add(file_path)
                sources.append(file_path)
        
        # Store result in semantic cache (if enabled)
        if semantic_cache is not None and embeddings is not None and not cache_hit:
            try:
                query_embedding = embeddings.embed_query(question)
                semantic_cache.set(
                    query=question,
                    query_embedding=query_embedding,
                    response=answer,
                    sources=sources
                )
            except Exception as cache_error:
                print(f"⚠️ [Semantic Cache] Error storing result: {cache_error}")
        
        # Save the conversation turn to history
        if history_manager:
            history_manager.save_turn(session_id, question, answer, sources if sources else None)
        
        # Extract timing information from the retriever if available
        retrieval_time = 0.0
        reranking_time = 0.0
        generation_time = 0.0
        
        try:
            retriever = active_chain.retriever
            if hasattr(retriever, 'timing_data') and retriever.timing_data:
                timing_data = retriever.timing_data
                
                # Check if this is a hybrid retriever (has bm25_search key)
                if 'bm25_search' in timing_data:
                    # Hybrid retriever timing
                    bm25_time = timing_data.get("bm25_search", 0)
                    dense_time = timing_data.get("dense_search", 0)
                    fusion_time = timing_data.get("rrf_fusion", 0)
                    reranking_time = timing_data.get("reranking", 0)
                    hybrid_time = timing_data.get("hybrid_search", 0)
                    generation_time = chain_time - (hybrid_time + reranking_time)
                    
                    # Log timing information to console
                    print("\n" + "="*80)
                    print("⏱️  TIMING BREAKDOWN (Hybrid Search)")
                    print("="*80)
                    
                    total = hybrid_time + reranking_time + generation_time
                    bar_width = 40
                    
                    # BM25 search
                    bm25_pct = (bm25_time / total * 100) if total > 0 else 0
                    bm25_bar = "█" * int((bm25_pct / 100) * bar_width) + "░" * (bar_width - int((bm25_pct / 100) * bar_width))
                    print(f"📝 BM25 Search   [{bm25_bar}] {bm25_pct:>5.1f}%  ({bm25_time:>7.2f}ms)")
                    
                    # Dense search
                    dense_pct = (dense_time / total * 100) if total > 0 else 0
                    dense_bar = "█" * int((dense_pct / 100) * bar_width) + "░" * (bar_width - int((dense_pct / 100) * bar_width))
                    print(f"🔍 Dense Search  [{dense_bar}] {dense_pct:>5.1f}%  ({dense_time:>7.2f}ms)")
                    
                    # RRF Fusion
                    fusion_pct = (fusion_time / total * 100) if total > 0 else 0
                    fusion_bar = "█" * int((fusion_pct / 100) * bar_width) + "░" * (bar_width - int((fusion_pct / 100) * bar_width))
                    print(f"🔀 RRF Fusion    [{fusion_bar}] {fusion_pct:>5.1f}%  ({fusion_time:>7.2f}ms)")
                    
                    # Reranking (if used)
                    if reranking_time > 0:
                        reranking_pct = (reranking_time / total * 100) if total > 0 else 0
                        reranking_bar = "█" * int((reranking_pct / 100) * bar_width) + "░" * (bar_width - int((reranking_pct / 100) * bar_width))
                        print(f"🎯 Reranking     [{reranking_bar}] {reranking_pct:>5.1f}%  ({reranking_time:>7.2f}ms)")
                    
                    # LLM Generation
                    generation_pct = (generation_time / total * 100) if total > 0 else 0
                    generation_bar = "█" * int((generation_pct / 100) * bar_width) + "░" * (bar_width - int((generation_pct / 100) * bar_width))
                    print(f"✨ Generation    [{generation_bar}] {generation_pct:>5.1f}%  ({generation_time:>7.2f}ms)")
                    
                    print("-" * 80)
                    print(f"{'Total Time':<16} {'':>40} ({total:>7.2f}ms)")
                    print("="*80 + "\n")
                else:
                    # Standard retriever timing
                    retrieval_time = timing_data.get("vectorstore_search", 0)
                    reranking_time = timing_data.get("reranking", 0)
                    generation_time = chain_time - (retrieval_time + reranking_time)
                    
                    # Log timing information to console
                    print("\n" + "="*80)
                    print("⏱️  TIMING BREAKDOWN (Per Phase)")
                    print("="*80)
                    
                    total = retrieval_time + reranking_time + generation_time
                    bar_width = 40
                    
                    # Calculate percentages
                    retrieval_pct = (retrieval_time / total * 100) if total > 0 else 0
                    reranking_pct = (reranking_time / total * 100) if total > 0 else 0
                    generation_pct = (generation_time / total * 100) if total > 0 else 0
                    
                    # Create bar visualizations
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
        
        # Stream the answer while preserving markdown structure (newlines are critical for tables!)
        # Split by lines first to preserve table structure, then by words within each line
        lines = answer.split('\n')
        
        for line_idx, line in enumerate(lines):
            if line.strip():  # Non-empty line
                words = line.split(' ')
                for word in words:
                    if word:  # Skip empty strings from multiple spaces
                        yield json.dumps({
                            "type": "chunk",
                            "content": word + " "
                        }) + "\n"
                        await asyncio.sleep(0.015)  # 15ms delay for smooth streaming
            
            # Add newline after each line (except the last one)
            if line_idx < len(lines) - 1:
                yield json.dumps({
                    "type": "chunk",
                    "content": "\n"
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


@app.on_event("startup")
async def startup_event():
    """Initialize chain, history manager, semantic cache, and Whisper model on startup"""
    global chain, chain_fr, chain_ar, settings, history_manager, whisper_model, semantic_cache, embeddings
    
    try:
        from src.chat import get_settings
        settings = get_settings()
        from src.chat import build_chain
        
        print("\n" + "="*60)
        print("🚀 Building language-specific chains...")
        print("="*60)
        
        # Build French chain
        chain_fr = build_chain(settings, language='fr')
        
        # Build Arabic chain
        chain_ar = build_chain(settings, language='ar')
        
        # Set default chain to French for backward compatibility
        chain = chain_fr
        
        history_manager = get_history_manager()
        
        print("\n" + "="*60)
        print("✅ Both chains initialized successfully!")
        print("   - French chain: Ready")
        print("   - Arabic chain: Ready")
        print("✅ History manager initialized")
        
        # Print retriever configuration
        if settings.use_hybrid_search:
            print(f"✅ Hybrid Search enabled (BM25: {settings.hybrid_bm25_weight}, Dense: {settings.hybrid_dense_weight})")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize Semantic Cache if enabled
    if settings and settings.use_semantic_cache:
        try:
            from src.semantic_cache import SemanticCache, REDIS_AVAILABLE
            
            if not REDIS_AVAILABLE:
                print("⚠️ Semantic cache disabled: Redis search module not installed")
                semantic_cache = None
            else:
                print("⏳ Initializing semantic cache...")
                semantic_cache = SemanticCache(
                    redis_host=settings.redis_host,
                    redis_port=settings.redis_port,
                    redis_password=settings.redis_password,
                    embedding_dim=768,  # multilingual-e5-base dimension
                    similarity_threshold=settings.cache_similarity_threshold,
                    ttl_seconds=settings.cache_ttl_seconds
                )
                
                # Initialize embeddings for cache lookups
                embeddings = choose_embeddings(settings)
                print(f"✅ Semantic cache initialized (threshold: {settings.cache_similarity_threshold})")
        except Exception as e:
            print(f"⚠️ Semantic cache not initialized: {e}")
            semantic_cache = None
    else:
        print("ℹ️ Semantic cache disabled (set USE_SEMANTIC_CACHE=true to enable)")
    
    # Initialize Whisper model (lazy load on first use for faster startup)
    try:
        from faster_whisper import WhisperModel
        import torch
        
        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        # Use "tiny" model for fastest transcription (good for short voice messages)
        print(f"⏳ Loading Whisper model (tiny) on {device}...")
        whisper_model = WhisperModel("tiny", device=device, compute_type=compute_type)
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        print(f"⚠️ Whisper model not loaded: {e}")
        whisper_model = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "chain_initialized": chain is not None,
        "chain_fr_initialized": chain_fr is not None,
        "chain_ar_initialized": chain_ar is not None,
        "history_initialized": history_manager is not None,
        "whisper_initialized": whisper_model is not None,
        "semantic_cache_initialized": semantic_cache is not None
    }


# ============== Batch Questions Endpoints ==============

def process_single_question(question: str) -> dict:
    """
    Process a single question through the RAG chain.
    Returns answer and sources.
    Uses language-appropriate chain based on detected query language.
    """
    global chain, chain_fr, chain_ar
    
    if chain is None:
        return {
            "answer": "Error: Chain not initialized",
            "sources": []
        }
    
    try:
        # Detect language and select appropriate chain
        query_language = detect_language(question)
        active_chain = chain_ar if query_language == 'ar' else chain_fr
        if active_chain is None:
            active_chain = chain  # Fallback
        
        result = active_chain.invoke({"question": question})
        answer = result.get("answer", "No response generated.")
        source_docs = result.get("source_documents", []) or []
        
        # Extract unique sources
        sources = []
        seen_paths = set()
        for doc in source_docs:
            file_path = doc.metadata.get("file_path", "") or doc.metadata.get("source", "unknown")
            if file_path not in seen_paths:
                seen_paths.add(file_path)
                sources.append(file_path)
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "sources": []
        }


@app.post("/batch", response_model=BatchAnswersResponse)
async def batch_questions(request: BatchQuestionsRequest):
    """
    Process multiple questions in batch and return answers.
    
    Input format:
    {
        "equipe": "team_name",
        "question": {
            "categorie_01": {
                "1": "Question 1",
                "2": "Question 2"
            },
            "categorie_02": {
                "1": "Question 3"
            }
        }
    }
    
    Output format:
    {
        "equipe": "team_name",
        "reponses": {
            "categorie_01": {
                "1": "Answer 1",
                "2": "Answer 2"
            },
            "categorie_02": {
                "1": "Answer 3"
            }
        }
    }
    """
    global chain
    
    if chain is None:
        raise HTTPException(
            status_code=503,
            detail="Chat chain not initialized. Please ensure the backend is properly started."
        )
    
    reponses = {}
    total_questions = sum(len(qs) for qs in request.question.values())
    processed = 0
    
    print(f"\n{'='*60}")
    print(f"📋 BATCH REQUEST from team: {request.equipe}")
    print(f"   Total questions: {total_questions}")
    print(f"{'='*60}")
    
    for category, questions in request.question.items():
        reponses[category] = {}
        
        for q_num, q_text in questions.items():
            processed += 1
            print(f"\n[{processed}/{total_questions}] Processing: {category} > Q{q_num}")
            print(f"   Question: {q_text[:80]}{'...' if len(q_text) > 80 else ''}")
            
            result = process_single_question(q_text)
            reponses[category][q_num] = result["answer"]
            
            print(f"   ✓ Answered ({len(result['answer'])} chars)")
    
    print(f"\n{'='*60}")
    print(f"✅ BATCH COMPLETE - {total_questions} questions processed")
    print(f"{'='*60}\n")
    
    return BatchAnswersResponse(
        equipe=request.equipe,
        reponses=reponses
    )


@app.post("/batch-with-sources", response_model=BatchAnswersWithSourcesResponse)
async def batch_questions_with_sources(request: BatchQuestionsRequest):
    """
    Process multiple questions in batch and return answers WITH sources.
    
    Input format: Same as /batch
    
    Output format:
    {
        "equipe": "team_name",
        "reponses": {
            "categorie_01": {
                "1": {
                    "answer": "Answer 1",
                    "sources": ["path/to/doc1.pdf", "path/to/doc2.pdf"]
                },
                "2": {
                    "answer": "Answer 2",
                    "sources": ["path/to/doc3.pdf"]
                }
            }
        }
    }
    """
    global chain
    
    if chain is None:
        raise HTTPException(
            status_code=503,
            detail="Chat chain not initialized. Please ensure the backend is properly started."
        )
    
    reponses = {}
    total_questions = sum(len(qs) for qs in request.question.values())
    processed = 0
    
    print(f"\n{'='*60}")
    print(f"📋 BATCH REQUEST (with sources) from team: {request.equipe}")
    print(f"   Total questions: {total_questions}")
    print(f"{'='*60}")
    
    for category, questions in request.question.items():
        reponses[category] = {}
        
        for q_num, q_text in questions.items():
            processed += 1
            print(f"\n[{processed}/{total_questions}] Processing: {category} > Q{q_num}")
            print(f"   Question: {q_text[:80]}{'...' if len(q_text) > 80 else ''}")
            
            result = process_single_question(q_text)
            reponses[category][q_num] = {
                "answer": result["answer"],
                "sources": result["sources"]
            }
            
            print(f"   ✓ Answered ({len(result['answer'])} chars, {len(result['sources'])} sources)")
    
    print(f"\n{'='*60}")
    print(f"✅ BATCH COMPLETE - {total_questions} questions processed")
    print(f"{'='*60}\n")
    
    return BatchAnswersWithSourcesResponse(
        equipe=request.equipe,
        reponses=reponses
    )



# ============== Speech-to-Text Endpoint ==============

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file to text using Whisper.
    
    Accepts audio files (webm, wav, mp3, m4a, ogg).
    Returns transcribed text.
    """
    global whisper_model
    
    if whisper_model is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper model not initialized. Speech-to-text is unavailable."
        )
    
    # Validate file type
    allowed_types = ["audio/webm", "audio/wav", "audio/mp3", "audio/mpeg", 
                     "audio/m4a", "audio/ogg", "audio/x-m4a", "video/webm"]
    if audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format: {audio.content_type}. Allowed: webm, wav, mp3, m4a, ogg"
        )
    
    try:
        # Save uploaded file to temp location
        suffix = Path(audio.filename).suffix if audio.filename else ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Transcribe with Whisper
            segments, info = whisper_model.transcribe(
                tmp_path,
                language="fr",  # French language
                beam_size=5,
                vad_filter=True  # Filter out silence
            )
            
            # Combine all segments
            text = " ".join([segment.text.strip() for segment in segments])
            
            return TranscriptionResponse(
                text=text,
                language=info.language
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription error: {str(e)}"
        )


# ============== File Serving Endpoint ==============

# Define allowed directories for file serving (relative to project root)
ALLOWED_DATA_DIRS = ["data/Convention", "data/Offres", "data/Offres en arabe", 
                     "data/Dépot Vente", "data/Guide NGBSS"]

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serve files from the data directory securely.
    
    Args:
        file_path: URL-encoded relative path to the file within data/
        
    Returns:
        FileResponse with appropriate MIME type
    """
    # Decode URL-encoded path
    decoded_path = unquote(file_path)
    
    # Security: Block directory traversal attacks
    if ".." in decoded_path or decoded_path.startswith("/") or decoded_path.startswith("\\"):
        raise HTTPException(status_code=403, detail="Access denied: Invalid path")
    
    # Normalize path separators
    normalized_path = decoded_path.replace("\\", "/")
    
    # Construct full path relative to project root
    project_root = Path(__file__).parent
    full_path = project_root / normalized_path
    
    # Resolve to absolute path and verify it's within allowed directories
    try:
        resolved_path = full_path.resolve()
        project_root_resolved = project_root.resolve()
        
        # Check if path is within project directory
        if not str(resolved_path).startswith(str(project_root_resolved)):
            raise HTTPException(status_code=403, detail="Access denied: Path outside project")
        
        # Check if path is within allowed data directories
        is_allowed = False
        for allowed_dir in ALLOWED_DATA_DIRS:
            allowed_path = (project_root / allowed_dir).resolve()
            if str(resolved_path).startswith(str(allowed_path)):
                is_allowed = True
                break
        
        if not is_allowed:
            raise HTTPException(status_code=403, detail="Access denied: Path not in allowed directories")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")
    
    # Check if file exists
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {decoded_path}")
    
    if not resolved_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(resolved_path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    
    # Get filename for Content-Disposition header
    filename = resolved_path.name
    
    # Return file with appropriate headers
    return FileResponse(
        path=str(resolved_path),
        media_type=mime_type,
        filename=filename
    )


# ============== Session Management Endpoints ==============

@app.get("/sessions")
async def get_sessions():
    """Get all chat sessions for the sidebar"""
    global history_manager
    
    if history_manager is None:
        raise HTTPException(status_code=503, detail="History manager not initialized")
    
    sessions = history_manager.get_all_sessions()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at,
                "updated_at": s.updated_at
            }
            for s in sessions
        ]
    }


@app.post("/sessions")
async def create_session(session_data: SessionCreate = None):
    """Create a new chat session"""
    global history_manager
    
    if history_manager is None:
        raise HTTPException(status_code=503, detail="History manager not initialized")
    
    title = session_data.title if session_data else "Nouvelle conversation"
    session = history_manager.create_session(title)
    
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session with all its messages"""
    global history_manager
    
    if history_manager is None:
        raise HTTPException(status_code=503, detail="History manager not initialized")
    
    session = history_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = history_manager.get_messages(session_id)
    
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "sources": m.sources
            }
            for m in messages
        ]
    }


@app.patch("/sessions/{session_id}")
async def update_session(session_id: str, session_data: SessionUpdate):
    """Update a session's title"""
    global history_manager
    
    if history_manager is None:
        raise HTTPException(status_code=503, detail="History manager not initialized")
    
    success = history_manager.update_session_title(session_id, session_data.title)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"success": True, "title": session_data.title}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    global history_manager
    
    if history_manager is None:
        raise HTTPException(status_code=503, detail="History manager not initialized")
    
    success = history_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"success": True}


# ============== Chat Endpoints ==============

@app.post("/chat")
async def chat(message: ChatMessage):
    """
    Non-streaming chat endpoint for backward compatibility.
    
    Returns complete response with sources.
    Uses language-appropriate chain based on detected query language.
    """
    global chain, chain_fr, chain_ar
    
    if chain is None:
        raise HTTPException(
            status_code=503,
            detail="Chat interface not initialized. Please ensure ChromaDB is properly configured and documents have been ingested."
        )
    
    try:
        # Detect language and select appropriate chain
        query_language = detect_language(message.question)
        active_chain = chain_ar if query_language == 'ar' else chain_fr
        if active_chain is None:
            active_chain = chain  # Fallback
        
        result = active_chain.invoke({"question": message.question})
        answer = result.get("answer", "No response generated.")
        source_docs = result.get("source_documents", []) or []
        
        # Extract file paths for download links (prefer file_path, fallback to source)
        sources = []
        seen_paths = set()
        for doc in source_docs:
            file_path = doc.metadata.get("file_path", "") or doc.metadata.get("source", "unknown")
            if file_path not in seen_paths:
                seen_paths.add(file_path)
                sources.append(file_path)
        
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
    global chain
    
    if chain is None:
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
            "POST /chat": "Non-streaming chat endpoint",
            "POST /chat/stream": "Streaming chat endpoint (Server-Sent Events)",
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
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False
    )

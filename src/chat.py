import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any
import os
import requests
import json
import time
import re
from datetime import datetime

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel
from langchain.schema import BaseRetriever, Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.prompts import PromptTemplate
import gradio as gr
import chromadb

from .config import get_settings
from .reranker import BGEReranker
from .timing_visualizer import visualize_timing

load_dotenv()


def detect_language(text: str) -> str:
    """
    Detect if the input text is in Arabic or French.
    
    Args:
        text: Input text to analyze
        
    Returns:
        'ar' for Arabic, 'fr' for French (default)
    """
    # Arabic Unicode range: \u0600-\u06FF (Arabic), \u0750-\u077F (Arabic Supplement)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')
    
    # Count Arabic characters
    arabic_chars = len(arabic_pattern.findall(text))
    total_chars = len(text.strip())
    
    # If more than 30% of characters are Arabic, consider it Arabic
    if total_chars > 0 and (arabic_chars / total_chars) > 0.3:
        return 'ar'
    
    return 'fr'  # Default to French

# Timing utilities
class TimingTracker:
    """Track execution time of different steps."""
    
    def __init__(self):
        self.steps = {}
        self.start_time = None
    
    def start_overall(self):
        """Start overall timing."""
        self.start_time = time.time()
    
    def record(self, step_name: str, start_time: float):
        """Record step duration."""
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.steps[step_name] = duration
    
    def print_summary(self):
        """Print timing summary to console."""
        if not self.steps:
            return
        
        print("\n" + "="*60)
        print("⏱️  EXECUTION TIME BREAKDOWN")
        print("="*60)
        
        total_time = sum(self.steps.values())
        
        for step_name, duration in self.steps.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            bar_length = int(percentage / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{step_name:<40} {duration:>8.2f}ms [{bar}] {percentage:>5.1f}%")
        
        print("-" * 60)
        print(f"{'Total Time':<40} {total_time:>8.2f}ms")
        print("="*60 + "\n")

# Global timing tracker
timing_tracker = TimingTracker()

# Global variables for chain and settings
chain = None
settings = None
semantic_cache = None  # Global semantic cache instance
embedding_model = None  # Global embedding model for cache


class DeepseekLLM(LLM):
    """Custom LLM wrapper for Deepseek API"""
    
    model: str
    api_key: str
    api_url: str
    temperature: float = 0.8
    max_tokens: int = 2048  # Maximum tokens to generate
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Deepseek API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            return answer
        
        except Exception as e:
            raise RuntimeError(f"Error calling Deepseek API: {str(e)}")


def choose_embeddings(settings):
    """Use Ollama embeddings (multilingual-e5-base, local, no API needed)."""
    model_name = settings.embedding_model
    return OllamaEmbeddings(
        model=model_name,
        base_url=settings.ollama_base_url
    )


def load_vectorstore(settings):
    """Load ChromaDB vector store (local only)."""
    embeddings = choose_embeddings(settings)
    
    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    
    # Create vector store from existing collection
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        client=chroma_client
    )
    
    return vectorstore


def build_llm(settings) -> BaseLanguageModel:
    """Build the LLM based on configuration"""
    
    if settings.llm_model.startswith("qllama/"):
        # Use Ollama local LLM
        llm_model_name = settings.llm_model
        return Ollama(
            model=llm_model_name,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature
        )
    else:
        # Use Deepseek or other API
        if not settings.llm_api_key:
            raise RuntimeError(
                f"LLM_API_KEY is required for {settings.llm_model}. "
                "Set it in .env file."
            )
        
        # Assume Deepseek API
        api_url = settings.llm_base_url or "https://api.modelarts-maas.com/v2/chat/completions"
        # Ensure the URL ends with /chat/completions, don't append if already present
        if not api_url.rstrip('/').endswith('/chat/completions'):
            api_url = api_url.rstrip('/') + '/chat/completions'
        return DeepseekLLM(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            api_url=api_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )


class RerankerRetriever(BaseRetriever):
    """Custom retriever that uses reranking to improve retrieval quality."""
    
    vectorstore: Chroma
    reranker: BGEReranker
    initial_k: int = 20
    timing_data: dict = {}  # Store timing for last retrieval
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve documents and rerank them with timing."""
        # Reset timing data
        self.timing_data = {}
        
        # Step 1: Retrieve initial documents
        retrieval_start = time.time()
        initial_docs = self.vectorstore.similarity_search(query, k=self.initial_k)
        self.timing_data["vectorstore_search"] = (time.time() - retrieval_start) * 1000
        
        # Step 2: Rerank documents
        reranking_start = time.time()
        reranked_docs = self.reranker.rerank(query, initial_docs)
        self.timing_data["reranking"] = (time.time() - reranking_start) * 1000
        
        return reranked_docs


class DynamicLanguageChain:
    """
    Wrapper around ConversationalRetrievalChain that dynamically selects
    the prompt template based on the detected language of the input question.
    """
    
    def __init__(self, llm, retriever, memory, settings):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.settings = settings
        
        # Pre-create both French and Arabic prompt templates
        self.prompts = {
            'fr': PromptTemplate(
                template=settings.prompt_fr,
                input_variables=["context", "question"]
            ),
            'ar': PromptTemplate(
                template=settings.prompt_ar,
                input_variables=["context", "question"]
            )
        }
        
        # Store the last detected language for debugging
        self.last_detected_language = None
        
        # Store the last created chain for accessing retriever timing
        self.last_chain = None
    
    def invoke(self, inputs: dict) -> dict:
        """
        Invoke the chain with dynamic language detection.
        
        Args:
            inputs: Dictionary with 'question' key
            
        Returns:
            Dictionary with 'answer' and 'source_documents'
        """
        question = inputs.get("question", "")
        
        # Detect the language of the question
        detected_lang = detect_language(question)
        self.last_detected_language = detected_lang
        
        # Select the appropriate prompt
        qa_prompt = self.prompts[detected_lang]
        
        # Create a new chain with the selected prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": qa_prompt,
                "document_separator": "\n\n---\n\n"
            },
            verbose=False
        )
        
        # Store for timing access
        self.last_chain = chain
        
        # Invoke the chain
        result = chain.invoke(inputs)
        
        # Log the detected language
        print(f"[Language Detection] Detected: {detected_lang.upper()} for question: {question[:50]}...")
        
        return result


def build_chain(settings) -> ConversationalRetrievalChain:
    """Build the conversational retrieval chain with optional reranking and semantic caching."""
    global semantic_cache, embedding_model
    
    print("\n[build_chain] Starting chain construction...")
    chain_start = time.time()
    
    # Step 0: Initialize Semantic Cache (if enabled)
    if settings.use_semantic_cache:
        try:
            from .semantic_cache import SemanticCache
            
            print("  [0/5] Initializing Semantic Cache...")
            step_start = time.time()
            
            # Initialize embedding model for cache (reuse the same as vectorstore)
            if embedding_model is None:
                embedding_model = OllamaEmbeddings(
                    base_url=settings.ollama_base_url,
                    model=settings.embedding_model.replace("ollama/", "")
                )
            
            semantic_cache = SemanticCache(
                redis_host=settings.redis_host,
                redis_port=settings.redis_port,
                redis_db=settings.redis_db,
                redis_password=settings.redis_password,
                embedding_dim=768,  # multilingual-e5-base dimension
                similarity_threshold=settings.cache_similarity_threshold,
                ttl_seconds=settings.cache_ttl_seconds,
            )
            
            cache_time = (time.time() - step_start) * 1000
            print(f"       ✓ Semantic Cache initialized in {cache_time:.2f}ms")
            print(f"       → Similarity threshold: {settings.cache_similarity_threshold}")
            print(f"       → TTL: {settings.cache_ttl_seconds}s ({settings.cache_ttl_seconds // 3600}h)")
        except Exception as e:
            print(f"       ⚠️  Semantic Cache initialization failed: {e}")
            print(f"       → Continuing without caching...")
            semantic_cache = None
    else:
        print("  [0/5] Semantic Cache disabled")
        semantic_cache = None
    
    # Step 1: Build LLM
    step_start = time.time()
    print("  [1/5] Building LLM...")
    llm = build_llm(settings)
    llm_time = (time.time() - step_start) * 1000
    print(f"       ✓ LLM built in {llm_time:.2f}ms")
    
    # Step 2: Load vectorstore
    step_start = time.time()
    print("  [2/5] Loading vectorstore...")
    vectorstore = load_vectorstore(settings)
    vectorstore_time = (time.time() - step_start) * 1000
    print(f"       ✓ Vectorstore loaded in {vectorstore_time:.2f}ms")
    
    # Step 3: Setup retriever
    step_start = time.time()
    print("  [3/5] Setting up retriever...")
    # Conditionally use reranker based on configuration
    if settings.use_reranker:
        print("       → Reranker enabled - using two-stage retrieval")
        # Initialize reranker
        reranker = BGEReranker(
            model_name=settings.reranker_model,
            top_k=settings.reranker_top_k
        )
        
        # Create custom retriever with reranking
        retriever = RerankerRetriever(
            vectorstore=vectorstore,
            reranker=reranker,
            initial_k=settings.initial_retrieval_k
        )
    else:
        print("       → Reranker disabled - using standard retrieval")
        # Use standard retriever without reranking
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.reranker_top_k}
        )
    retriever_time = (time.time() - step_start) * 1000
    print(f"       ✓ Retriever setup in {retriever_time:.2f}ms")
    
    # Step 4: Create memory and dynamic language chain
    step_start = time.time()
    print("  [4/5] Creating conversation chain with dynamic language detection...")
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    # Create dynamic language chain that selects prompt based on input language
    print("       → Using dynamic language detection (FR/AR)")
    final_chain = DynamicLanguageChain(
        llm=llm,
        retriever=retriever,
        memory=memory,
        settings=settings
    )
    
    chain_creation_time = (time.time() - step_start) * 1000
    print(f"       ✓ Chain created in {chain_creation_time:.2f}ms")
    
    # Print total build time
    total_build_time = (time.time() - chain_start) * 1000
    print("\n" + "="*60)
    print(f"Chain build completed in {total_build_time:.2f}ms total")
    print("="*60 + "\n")
    
    return final_chain


def cached_chain_invoke(question: str) -> dict:
    """
    Invoke the chain with semantic caching.
    Checks cache first, returns cached response if similarity > threshold.
    Otherwise invokes the chain and caches the result.
    
    Args:
        question: User's question
        
    Returns:
        Dictionary with 'answer', 'source_documents', and 'cache_hit'
    """
    global chain, semantic_cache, embedding_model, settings
    
    # Check if chain is initialized
    if chain is None:
        raise RuntimeError("Chain is not initialized. Please ensure the chain is built before invoking.")
    
    # If cache is disabled or not initialized, invoke chain directly
    if semantic_cache is None or not settings.use_semantic_cache:
        result = chain.invoke({"question": question})
        result["cache_hit"] = False
        return result
    
    # Step 1: Generate embedding for the question
    cache_start = time.time()
    query_embedding = embedding_model.embed_query(question)
    embedding_time = (time.time() - cache_start) * 1000
    
    # Step 2: Check cache
    cached_result = semantic_cache.get(
        query=question,
        query_embedding=query_embedding,
    )
    
    if cached_result:
        # Cache hit! Return immediately
        # Convert sources back to Document objects for compatibility
        from langchain.schema import Document
        source_docs = [
            Document(page_content="", metadata={"source": src})
            for src in cached_result["sources"]
        ]
        
        return {
            "answer": cached_result["answer"],
            "source_documents": source_docs,
            "cache_hit": True,
            "similarity": cached_result["similarity"],
            "cached_query": cached_result["cached_query"],
            "latency_ms": cached_result["latency_ms"],
        }
    
    # Step 3: Cache miss - invoke the chain
    print(f"[Cache] Invoking chain for new query...")
    chain_start = time.time()
    result = chain.invoke({"question": question})
    chain_time = (time.time() - chain_start) * 1000
    
    # Step 4: Store result in cache
    answer = result.get("answer", "")
    source_docs = result.get("source_documents", []) or []
    sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
    
    semantic_cache.set(
        query=question,
        query_embedding=query_embedding,
        response=answer,
        sources=sources,
    )
    
    # Add cache info to result
    result["cache_hit"] = False
    result["chain_time_ms"] = chain_time
    
    return result


def initialize_chain():
    """Initialize the chain on startup."""
    global chain, settings
    try:
        settings = get_settings()
        chain = build_chain(settings)
        print("[chat] Chain initialized successfully.")
        return True
    except Exception as exc:
        print(f"[chat] Startup error: {exc}")
        import traceback
        traceback.print_exc()
        return False


def chat_response(message: str, chat_history):
    """
    Process chat message and return response with sources.
    
    Args:
        message: User's question
        chat_history: List of [user_msg, bot_msg] tuples (Gradio format)
    
    Returns:
        Updated chat_history as list of tuples
    """
    global chain, settings
    
    # Start timing for this request
    request_start = time.time()
    steps_timing = {}
    detailed_steps = {}
    
    # Step 1: Validate chain initialization
    step_start = time.time()
    if chain is None:
        error_msg = "Chat interface not initialized. Make sure ChromaDB is properly configured and documents have been ingested."
        if chat_history is None:
            chat_history = []
        chat_history.append([message, error_msg])
        return chat_history
    steps_timing["Chain validation"] = (time.time() - step_start) * 1000

    try:
        # Step 2: Invoke chain (retrieval + LLM call)
        step_start = time.time()
        result = chain.invoke({"question": message})
        chain_invocation_time = (time.time() - step_start) * 1000
        steps_timing["Chain invocation (retrieval + LLM)"] = chain_invocation_time
        
        # Extract detailed timing from retriever if available
        # Try to get timing from the retriever in the chain
        retrieval_time = 0.0
        reranking_time = 0.0
        llm_time = 0.0
        
        try:
            retriever = chain.retriever
            if hasattr(retriever, 'timing_data') and retriever.timing_data:
                retrieval_time = retriever.timing_data.get("vectorstore_search", 0)
                reranking_time = retriever.timing_data.get("reranking", 0)
                llm_time = chain_invocation_time - (retrieval_time + reranking_time)
                
                detailed_steps["  ├─ Vectorstore search"] = retrieval_time
                detailed_steps["  ├─ Reranking"] = reranking_time
                detailed_steps["  └─ LLM generation"] = llm_time
            else:
                # Fallback: estimate LLM time as remainder
                llm_time = chain_invocation_time
                detailed_steps["  └─ LLM generation"] = chain_invocation_time
        except Exception:
            llm_time = chain_invocation_time
            detailed_steps["  └─ LLM generation"] = chain_invocation_time
        
        # Step 3: Extract answer
        step_start = time.time()
        answer = result.get("answer", "No response generated.")
        source_docs = result.get("source_documents", []) or []
        steps_timing["Extract answer"] = (time.time() - step_start) * 1000
        
        # Step 4: Format sources
        step_start = time.time()
        sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
        sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {source}" for source in sources) if sources else ""
        full_response = answer + sources_text
        steps_timing["Format sources"] = (time.time() - step_start) * 1000
        
        # Step 5: Update chat history
        step_start = time.time()
        if chat_history is None:
            chat_history = []
        chat_history.append([message, full_response])
        steps_timing["Update chat history"] = (time.time() - step_start) * 1000
        
        # Print timing summary with detailed breakdown
        total_time = (time.time() - request_start) * 1000
        _print_timing_summary(steps_timing, detailed_steps, total_time)
        
        # Visualize timing breakdown for the three main phases
        visualize_timing(retrieval_time, reranking_time, llm_time)
        
        return chat_history
    
    except Exception as exc:
        error_msg = f"Error processing message: {str(exc)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        if chat_history is None:
            chat_history = []
        
        chat_history.append([message, error_msg])
        return chat_history


def _print_timing_summary(steps_timing: dict, detailed_steps: dict = None, total_time: float = 0):
    """Print timing summary for a chat response with detailed breakdown.
    
    Args:
        steps_timing: Dict of step names to duration in milliseconds
        detailed_steps: Dict of sub-steps (e.g., retrieval breakdown) to duration in ms
        total_time: Total duration in milliseconds
    """
    if detailed_steps is None:
        detailed_steps = {}
    
    print("\n" + "="*80)
    print("⏱️  CHAT RESPONSE TIMING BREAKDOWN")
    print("="*80)
    
    for step_name, duration_ms in steps_timing.items():
        percentage = (duration_ms / total_time * 100) if total_time > 0 else 0
        bar_length = min(int(percentage / 5), 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"{step_name:<50} {duration_ms:>8.2f}ms [{bar}] {percentage:>6.1f}%")
        
        # Print detailed sub-steps if this is the chain invocation step
        if "Chain invocation" in step_name and detailed_steps:
            for detail_name, detail_duration in detailed_steps.items():
                if detail_duration > 0:
                    detail_percentage = (detail_duration / total_time * 100) if total_time > 0 else 0
                    detail_bar_length = min(int(detail_percentage / 5), 20)
                    detail_bar = "█" * detail_bar_length + "░" * (20 - detail_bar_length)
                    print(f"{detail_name:<50} {detail_duration:>8.2f}ms [{detail_bar}] {detail_percentage:>6.1f}%")
    
    print("-" * 80)
    print(f"{'Total Response Time':<50} {total_time:>8.2f}ms")
    print("="*80 + "\n")


def create_gradio_interface():
    """Create and launch Gradio interface."""
    with gr.Blocks(title="Smart Offer Finder") as demo:
        gr.Markdown(
            """
            # Smart Offer Finder
            
            An intelligent RAG-powered chatbot to help you find relevant offers, conventions, and operational guides.
            
            **Note:** Make sure you have ingested documents first.
            - Ingest documents: `python -m src.ingest`
            """
        )
        
        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
        )
        
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask a question about offers, conventions, or guides...",
            lines=2,
        )
        
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")
        
        # Set up interactions
        submit_btn.click(
            fn=chat_response,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        msg.submit(
            fn=chat_response,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        gr.Markdown(
            """
            ---
            **Built with:**
            - [LangChain](https://langchain.com/) - RAG framework
            - [ChromaDB](https://www.trychroma.com/) - Local vector database
            - [Ollama](https://ollama.ai/) - Local embeddings
            - [Deepseek](https://www.deepseek.com/) - LLM
            - [Gradio](https://gradio.app/) - Web interface
            """
        )
    
    return demo


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🚀 Initializing Smart Offer Finder...")
    print("="*60)
    
    startup_start = time.time()
    
    if not initialize_chain():
        print("\n❌ Failed to initialize chain. Please check:")
        print("   1. ChromaDB data exists at: data/chroma_db")
        print("   2. LLM_API_KEY is set in .env (if using remote LLM)")
        print("   3. Documents have been ingested: python -m src.ingest --db chromadb")
        sys.exit(1)
    
    startup_time = (time.time() - startup_start) * 1000
    
    print("\n✅ Chain initialized successfully!")
    print(f"   Startup time: {startup_time:.2f}ms")
    print("\n🚀 Launching Gradio interface...")
    print("   Open browser to: http://localhost:7860")
    print("="*60 + "\n")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
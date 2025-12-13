import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any
from urllib.parse import quote
import os
import requests
import json
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel
from langchain.schema import BaseRetriever, Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
import gradio as gr
import chromadb

from .config import get_settings
from .reranker import BGEReranker
from .timing_visualizer import visualize_timing
from .hybrid_retriever import get_hybrid_retriever_from_vectorstore, HybridRetriever, detect_language, filter_documents_by_language

load_dotenv()

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


class E5EmbeddingsWrapper(Embeddings):
    """
    Wrapper for E5 embeddings that adds proper prefixes.
    
    E5 models require:
    - "query: " prefix for search queries
    - "passage: " prefix for documents (already added in ingest.py)
    
    This wrapper ensures queries get the correct prefix.
    """
    
    def __init__(self, base_embeddings: OllamaEmbeddings):
        self.base_embeddings = base_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents - already have 'passage:' prefix from ingest."""
        return self.base_embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with 'query:' prefix for E5 model."""
        # Add "query: " prefix for E5 model
        prefixed_query = f"query: {text}"
        return self.base_embeddings.embed_query(prefixed_query)
        
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

# Global variables for chains and settings
chain_fr = None  # French chain
chain_ar = None  # Arabic chain
chain = None     # Default chain (for backward compatibility)
settings = None


class DeepseekLLM(LLM):
    """Custom LLM wrapper for Deepseek API"""
    
    model: str
    api_key: str
    api_url: str
    temperature: float = 0.7
    
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
            return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            raise RuntimeError(f"Error calling Deepseek API: {str(e)}")


def choose_embeddings(settings):
    """
    Use Ollama embeddings (multilingual-e5-base, local, no API needed).
    
    Wraps with E5EmbeddingsWrapper to add proper "query:" prefix for searches.
    """
    model_name = settings.embedding_model
    base_embeddings = OllamaEmbeddings(
        model=model_name,
        base_url=settings.ollama_base_url
    )
    
    # Wrap with E5 prefix handler for proper query formatting
    return E5EmbeddingsWrapper(base_embeddings)


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
            temperature=0.7
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
            temperature=0.7
        )


class RerankerRetriever(BaseRetriever):
    """Custom retriever that uses reranking to improve retrieval quality."""
    
    vectorstore: Chroma
    reranker: BGEReranker
    initial_k: int = 20
    timing_data: dict = {}  # Store timing for last retrieval
    filter_by_language: bool = True  # Enable language filtering by default
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve documents and rerank them with timing and language filtering."""
        # Reset timing data
        self.timing_data = {}
        
        # Step 1: Retrieve initial documents
        retrieval_start = time.time()
        initial_docs = self.vectorstore.similarity_search(query, k=self.initial_k)
        self.timing_data["vectorstore_search"] = (time.time() - retrieval_start) * 1000
        
        # Step 2: Language filtering
        if self.filter_by_language:
            filter_start = time.time()
            query_language = detect_language(query)
            pre_filter_count = len(initial_docs)
            initial_docs = filter_documents_by_language(initial_docs, query_language)
            self.timing_data["language_filter"] = (time.time() - filter_start) * 1000
            print(f"  🌐 Language filter: {query_language.upper()} ({pre_filter_count} → {len(initial_docs)} docs)")
        
        # Step 3: Rerank documents
        reranking_start = time.time()
        reranked_docs = self.reranker.rerank(query, initial_docs)
        self.timing_data["reranking"] = (time.time() - reranking_start) * 1000
        
        return reranked_docs


# ============== Language-Specific QA Prompts ==============

# French prompt template
QA_PROMPT_TEMPLATE_FR = """Vous êtes un assistant intelligent qui aide les utilisateurs à trouver des informations sur les offres et services télécom.

Utilisez le contexte suivant pour répondre à la question. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

Contexte:
{context}

Question: {question}

Réponse:"""

# Arabic prompt template
QA_PROMPT_TEMPLATE_AR = """أنت مساعد ذكي يساعد المستخدمين في العثور على معلومات حول عروض وخدمات الاتصالات.

استخدم السياق التالي للإجابة على السؤال. إذا كنت لا تعرف الإجابة، قل ببساطة أنك لا تعرف، ولا تحاول اختلاق إجابة.

السياق:
{context}

السؤال: {question}

الإجابة:"""

QA_PROMPT_FR = PromptTemplate(
    template=QA_PROMPT_TEMPLATE_FR,
    input_variables=["context", "question"]
)

QA_PROMPT_AR = PromptTemplate(
    template=QA_PROMPT_TEMPLATE_AR,
    input_variables=["context", "question"]
)


def get_qa_prompt(language: str) -> PromptTemplate:
    """Get the appropriate QA prompt based on detected language."""
    if language == 'ar':
        return QA_PROMPT_AR
    return QA_PROMPT_FR  # Default to French


def build_chain(settings, language: str = 'fr') -> ConversationalRetrievalChain:
    """Build the conversational retrieval chain with optional hybrid search and reranking.
    
    Args:
        settings: Application settings
        language: Language for the QA prompt ('fr' or 'ar')
    """
    
    print(f"\n[build_chain] Building chain for language: {language.upper()}...")
    chain_start = time.time()
    
    # Step 1: Build LLM
    step_start = time.time()
    print("  [1/4] Building LLM...")
    llm = build_llm(settings)
    llm_time = (time.time() - step_start) * 1000
    print(f"       ✓ LLM built in {llm_time:.2f}ms")
    
    # Step 2: Load vectorstore
    step_start = time.time()
    print("  [2/4] Loading vectorstore...")
    vectorstore = load_vectorstore(settings)
    vectorstore_time = (time.time() - step_start) * 1000
    print(f"       ✓ Vectorstore loaded in {vectorstore_time:.2f}ms")
    
    # Step 3: Setup retriever (priority: hybrid > reranker > standard)
    step_start = time.time()
    print("  [3/4] Setting up retriever...")
    
    if settings.use_hybrid_search:
        # Use hybrid search (BM25 + Dense with optional reranking)
        print("       → Hybrid search enabled (BM25 + Dense)")
        print(f"         BM25 weight: {settings.hybrid_bm25_weight}, Dense weight: {settings.hybrid_dense_weight}")
        
        retriever = get_hybrid_retriever_from_vectorstore(
            vectorstore=vectorstore,
            bm25_weight=settings.hybrid_bm25_weight,
            dense_weight=settings.hybrid_dense_weight,
            bm25_k=settings.hybrid_bm25_k,
            dense_k=settings.hybrid_dense_k,
            use_reranker=settings.use_reranker,
            reranker_model=settings.reranker_model,
            reranker_top_k=settings.reranker_top_k
        )
        
        if settings.use_reranker:
            print("         + Reranking enabled")
        
    elif settings.use_reranker:
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
        print("       → Standard retrieval (semantic only)")
        # Use standard retriever without reranking
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.reranker_top_k}
        )
    retriever_time = (time.time() - step_start) * 1000
    print(f"       ✓ Retriever setup in {retriever_time:.2f}ms")
    
    # Step 4: Create memory and chain with language-specific prompt
    step_start = time.time()
    print("  [4/4] Creating conversation chain...")
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    # Get language-specific prompt
    qa_prompt = get_qa_prompt(language)
    
    final_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    chain_creation_time = (time.time() - step_start) * 1000
    print(f"       ✓ Chain created in {chain_creation_time:.2f}ms")
    
    # Print total build time
    total_build_time = (time.time() - chain_start) * 1000
    print(f"   Chain ({language.upper()}) build completed in {total_build_time:.2f}ms")
    
    return final_chain


def initialize_chain():
    """Initialize both language chains on startup."""
    global chain, chain_fr, chain_ar, settings
    try:
        settings = get_settings()
        
        print("\n" + "="*60)
        print("🚀 Building language-specific chains...")
        print("="*60)
        
        # Build French chain
        chain_fr = build_chain(settings, language='fr')
        
        # Build Arabic chain
        chain_ar = build_chain(settings, language='ar')
        
        # Set default chain to French for backward compatibility
        chain = chain_fr
        
        print("\n" + "="*60)
        print("✅ Both chains initialized successfully!")
        print("   - French chain: Ready")
        print("   - Arabic chain: Ready")
        print("="*60 + "\n")
        
        return True
    except Exception as exc:
        print(f"[chat] Startup error: {exc}")
        import traceback
        traceback.print_exc()
        return False


def get_chain_for_language(language: str) -> ConversationalRetrievalChain:
    """Get the appropriate chain based on detected language."""
    global chain_fr, chain_ar, chain
    if language == 'ar':
        return chain_ar if chain_ar else chain
    return chain_fr if chain_fr else chain


def chat_response(message: str, chat_history):
    """
    Process chat message and return response with sources.
    
    Args:
        message: User's question
        chat_history: List of [user_msg, bot_msg] tuples (Gradio format)
    
    Returns:
        Updated chat_history as list of tuples
    """
    global chain_fr, chain_ar, chain, settings
    
    # Start timing for this request
    request_start = time.time()
    steps_timing = {}
    detailed_steps = {}
    
    # Step 1: Validate chain initialization and detect language
    step_start = time.time()
    if chain is None:
        error_msg = "Chat interface not initialized. Make sure ChromaDB is properly configured and documents have been ingested."
        if chat_history is None:
            chat_history = []
        chat_history.append([message, error_msg])
        return chat_history
    
    # Detect language and select appropriate chain
    query_language = detect_language(message)
    active_chain = get_chain_for_language(query_language)
    print(f"  🌐 Query language: {query_language.upper()} → using {query_language.upper()} chain")
    
    steps_timing["Chain validation + language detection"] = (time.time() - step_start) * 1000

    try:
        # Step 2: Invoke chain (retrieval + LLM call)
        step_start = time.time()
        result = active_chain.invoke({"question": message})
        chain_invocation_time = (time.time() - step_start) * 1000
        steps_timing["Chain invocation (retrieval + LLM)"] = chain_invocation_time
        
        # Extract detailed timing from retriever if available
        # Try to get timing from the retriever in the chain
        retrieval_time = 0.0
        reranking_time = 0.0
        llm_time = 0.0
        
        try:
            retriever = active_chain.retriever
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
        
        # Step 4: Format sources as clickable markdown links
        step_start = time.time()
        
        # Deduplicate sources by file_path
        seen_paths = set()
        unique_sources = []
        for doc in source_docs:
            file_path = doc.metadata.get("file_path", "")
            source_name = doc.metadata.get("source", "unknown")
            
            # Use file_path as unique key, fallback to source name
            key = file_path if file_path else source_name
            if key not in seen_paths:
                seen_paths.add(key)
                unique_sources.append({
                    "name": source_name,
                    "file_path": file_path
                })
        
        # Build markdown links
        sources_lines = []
        for src in unique_sources:
            if src["file_path"]:
                # URL-encode the path for safe linking
                encoded_path = quote(src["file_path"], safe="/")
                sources_lines.append(f"- [{src['name']}](/files/{encoded_path})")
            else:
                # No file_path available, just show the name
                sources_lines.append(f"- {src['name']}")
        
        sources_text = "\n\n**Sources:**\n" + "\n".join(sources_lines) if sources_lines else ""
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
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import json
import gradio as gr
from dotenv import load_dotenv
# Use FAST GPU Embeddings
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from typing import Any, List, Optional, Sequence
import chromadb
import requests

# Import your modules
from src.reranker import BGEReranker
from src.config import get_settings

load_dotenv()

# Global reranker instance (lazy loaded)
_reranker_instance = None

def get_reranker():
    """Get or create the global reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker(top_k=5)
    return _reranker_instance

# --- CUSTOM RERANKER ADAPTER ---
class FlashRankCompressor(BaseDocumentCompressor):
    top_n: int = 5

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None
    ) -> Sequence[Document]:
        if not documents: return []

        # Use BGEReranker to rerank documents
        reranker = get_reranker()
        reranker.top_k = self.top_n
        reranked_docs = reranker.rerank_with_scores(query, list(documents))
        
        final_docs = []
        for doc, score in reranked_docs:
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)
        return final_docs

# --- STREAMING LLM FUNCTION ---
def stream_llm_response(prompt: str, settings) -> str:
    """Stream response from DeepSeek API token by token."""
    endpoint = settings.llm_base_url
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint.rstrip('/')}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.llm_api_key}"
    }
    
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "stream": True  # Enable streaming!
    }
    
    response = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=60)
    response.raise_for_status()
    
    # Parse Server-Sent Events (SSE)
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                if data.strip() == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get('choices', [])
                    if choices and len(choices) > 0:
                        delta = choices[0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

# --- BUILD COMPONENTS ---
components = None

# Keywords for NGBSS guide detection
NGBSS_KEYWORDS = [
    "ngbss", "étapes", "etapes", "comment faire", "procédure", "procedure",
    "guide", "tutoriel", "recharge", "bon de commande", "créer", "creer",
    "activer", "désactiver", "facturation", "paiement", "retour ressource",
    "pack idoom", "fibre", "4g", "adsl", "client portal"
]

def detect_query_type(query: str) -> str:
    """Detect if query is about NGBSS procedures or offers."""
    query_lower = query.lower()
    for keyword in NGBSS_KEYWORDS:
        if keyword in query_lower:
            return "ngbss"
    return "offers"

def build_components(settings):
    """Build retrievers for BOTH collections (called once)."""
    print("🚀 Loading Embeddings on CUDA...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    
    # --- OFFERS Collection ---
    offers_vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        client=chroma_client
    )
    offers_base_retriever = offers_vectorstore.as_retriever(search_kwargs={"k": 15})
    offers_compressor = FlashRankCompressor(top_n=5)
    offers_retriever = ContextualCompressionRetriever(
        base_compressor=offers_compressor,
        base_retriever=offers_base_retriever
    )
    
    # --- NGBSS Guides Collection ---
    try:
        ngbss_vectorstore = Chroma(
            collection_name="ngbss-guides",
            embedding_function=embeddings,
            client=chroma_client
        )
        ngbss_base_retriever = ngbss_vectorstore.as_retriever(search_kwargs={"k": 10})
        ngbss_compressor = FlashRankCompressor(top_n=5)
        ngbss_retriever = ContextualCompressionRetriever(
            base_compressor=ngbss_compressor,
            base_retriever=ngbss_base_retriever
        )
        print("✅ NGBSS guides collection loaded!")
    except Exception as e:
        print(f"⚠️ NGBSS collection not found, using offers only: {e}")
        ngbss_retriever = None
    
    return {
        "offers_retriever": offers_retriever,
        "ngbss_retriever": ngbss_retriever,
        "settings": settings
    }

# --- STREAMING CHAT FUNCTION ---
def chat_stream(message, history):
    """Stream the response word-by-word like ChatGPT."""
    global components
    if not components:
        components = build_components(get_settings())
    
    # 1. Detect query type and select retriever
    query_type = detect_query_type(message)
    print(f"\n🔍 Query type detected: {query_type.upper()}")  # Debug print
    
    if query_type == "ngbss" and components["ngbss_retriever"]:
        retriever = components["ngbss_retriever"]
        context_type = "📋 NGBSS Guide"
        system_prompt = "You are a helpful assistant that explains NGBSS procedures step by step. Be precise about button locations and navigation paths."
    else:
        retriever = components["offers_retriever"]
        context_type = "📦 Offres"
        system_prompt = "You are a helpful assistant that answers questions about Algérie Télécom offers. Be concise and accurate."
    
    # 2. Retrieve documents
    query_prefixed = f"query: {message}"
    docs = retriever.invoke(query_prefixed)
    
    # 3. Build context from retrieved docs
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # 4. Build prompt with appropriate system message
    prompt = f"""Based on the following context, answer the question accurately.

Context:
{context}

Question: {message}

Answer:"""
    
    # 5. Format sources
    sources_text = f"\n\n**{context_type} - Sources:**"
    seen_sources = set()
    for doc in docs:
        name = doc.metadata.get("source", "Unknown")
        score = doc.metadata.get("relevance_score", 0)
        # For NGBSS, show procedure and step
        if query_type == "ngbss":
            procedure = doc.metadata.get("procedure_name", "")
            step = doc.metadata.get("step_order", "")
            if procedure:
                name = f"{procedure} (Étape {step})"
        if name not in seen_sources:
            sources_text += f"\n- {name} (Score: {score:.2f})"
            seen_sources.add(name)
    
    # 6. Stream the response
    history = history + [[message, ""]]
    partial_response = ""
    
    try:
        for token in stream_llm_response(prompt, components["settings"]):
            partial_response += token
            history[-1][1] = partial_response
            yield "", history
    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
        yield "", history
        return
    
    # 7. Append sources at the end
    history[-1][1] = partial_response + sources_text
    yield "", history

def main():
    with gr.Blocks(title="Fast Offer Finder 🚀") as demo:
        gr.Markdown("## ⚡ Smart Offer Finder (Streaming)")
        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(label="Question", placeholder="Ask about offers...")
        btn = gr.Button("Envoyer")
        
        # Use streaming with generators
        btn.click(chat_stream, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(chat_stream, inputs=[msg, chatbot], outputs=[msg, chatbot])

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
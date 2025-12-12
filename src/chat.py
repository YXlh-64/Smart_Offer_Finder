import os
# Disable ChromaDB telemetry BEFORE any chromadb imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
from typing import List, Optional, Sequence, Any

import gradio as gr

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import chromadb
from chromadb.config import Settings as ChromaSettings

# --- IMPORT YOUR RERANKER ---
try:
    from .reranker import reranker
except ImportError:
    from src.reranker import reranker

try:
    from .config import get_settings
except ImportError:
    from src.config import get_settings

load_dotenv()

# --- CUSTOM RERANKER ADAPTER ---
class BGECompressor(BaseDocumentCompressor):
    top_n: int = 5

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Any] = None
    ) -> Sequence[Document]:
        if not documents:
            return []

        class NodeWrapper:
            def __init__(self, doc):
                self.text = doc.page_content
                self.original_doc = doc
                self.score = 0.0

        nodes = [NodeWrapper(doc) for doc in documents]
        
        # Use existing reranker
        print(f"🔄 Reranking {len(nodes)} documents...")
        reranked_nodes = reranker._rerank_nodes(query, nodes, top_k=self.top_n, log=True)

        final_docs = []
        for node in reranked_nodes:
            doc = node.original_doc
            doc.metadata["relevance_score"] = node.score
            final_docs.append(doc)

        return final_docs

# --- LLM WRAPPER ---
class DeepseekLLM(LLM):
    model: str
    api_key: str
    api_url: str
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        import requests
        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "stream": False,
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Error calling Deepseek API: {str(e)}")

# --- CHAIN BUILDER ---
def build_chain(settings) -> ConversationalRetrievalChain:
    if settings.llm_model.startswith("ollama/"):
        llm = Ollama(model=settings.llm_model.replace("ollama/", ""), base_url=settings.ollama_base_url, temperature=0.1)
    else:
        llm = DeepseekLLM(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            api_url=settings.llm_base_url
        )

    embeddings = OllamaEmbeddings(model=settings.embedding_model.replace("ollama/", ""), base_url=settings.ollama_base_url)
    
    # Create ChromaDB client with telemetry disabled
    chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        client=chroma_client
    )
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
    
    compressor = BGECompressor(top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True
    )

# --- GRADIO UI (FIXED FOR VERSION 4.20) ---

chain = None

def chat_response(message, history):
    """
    Logic adapted for Gradio 4.20 (List of Lists format).
    history is: [[user_msg, bot_msg], [user_msg, bot_msg], ...]
    """
    global chain
    if not chain:
        settings = get_settings()
        chain = build_chain(settings)

    # 1. Run Chain
    result = chain.invoke({"question": message})
    answer = result.get("answer")
    source_docs = result.get("source_documents", [])
    
    # 2. Format Sources
    sources_text = "\n\n**Sources (Verified by AI):**"
    for i, doc in enumerate(source_docs):
        score = doc.metadata.get("relevance_score", 0.0)
        source_name = doc.metadata.get("source", "Unknown")
        icon = "🟢" if score > 0.5 else "🟠"
        sources_text += f"\n{icon} [{score:.3f}] {source_name}"
    
    full_response = answer + sources_text
    
    # 3. Append to History (Format: List of [User, Bot])
    # Note: Gradio automatically appends the current pair if we return it here? 
    # Actually, for 'click' or 'submit', we usually return the updated list.
    
    # Check if history is None
    if history is None:
        history = []
        
    history.append([message, full_response])
            
    return "", history # Return empty string to clear textbox, and updated history

def main():
    with gr.Blocks(title="Smart Offer Finder") as demo:
        gr.Markdown("# 🇩🇿 Smart Offer Finder (With BGE Reranker)")
        
        # REMOVED type="messages" to fix your error
        chatbot = gr.Chatbot(height=600) 
        
        msg = gr.Textbox(label="Question")
        btn = gr.Button("Envoyer")
        
        # For version 4.20, we pass the inputs and outputs
        btn.click(chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
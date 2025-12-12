import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import gradio as gr
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# Use FAST GPU Embeddings
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain.llms.base import LLM
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from typing import Any, List, Optional, Sequence
import chromadb
import requests # Added missing import

# Import your modules
from src.reranker import reranker
from src.config import get_settings

load_dotenv()

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

        class NodeWrapper:
            def __init__(self, doc):
                self.text = doc.page_content
                self.original_doc = doc
                self.score = 0.0

        nodes = [NodeWrapper(doc) for doc in documents]
        
        # Call the fast FlashRank reranker
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
    temperature: float = 0.3 # Lower temperature for factual accuracy

    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            # Fix URL path if missing
            endpoint = self.api_url
            if not endpoint.endswith("/chat/completions"):
                endpoint = f"{endpoint.rstrip('/')}/chat/completions"

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "stream": False,
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"API Error: {str(e)}")

# --- CHAIN BUILDER ---
def build_chain(settings):
    # 1. Initialize Fast GPU Embeddings
    print("🚀 Loading Embeddings on CUDA...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Connect to VectorDB
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        client=chroma_client
    )
    
    # 3. Base Retriever (Fetch more docs for reranking)
    # CRITICAL: We fetch 15 docs to ensure we find the right table, then rerank down to 5
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # 4. Attach Reranker
    compressor = FlashRankCompressor(top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # 5. Initialize LLM
    llm = DeepseekLLM(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        api_url=settings.llm_base_url
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever, # Use the Reranker!
        memory=memory,
        return_source_documents=True,
        # Optimize prompt to force using context
        combine_docs_chain_kwargs={"prompt": None} 
    )

# --- GRADIO UI ---
chain = None

def chat_response(message, history):
    global chain
    if not chain:
        chain = build_chain(get_settings())

    # CRITICAL: Add "query: " prefix for E5 model accuracy
    query_for_retrieval = f"query: {message}"
    
    # We pass the prefixed query to the chain
    # Note: Depending on LangChain version, it might display the prefix in chat. 
    # Ideally, we modify the retriever to handle this, but passing it here is the quickest fix.
    result = chain.invoke({"question": query_for_retrieval})
    
    answer = result.get("answer")
    source_docs = result.get("source_documents", [])
    
    sources_text = "\n\n**Sources:**"
    seen_sources = set()
    for doc in source_docs:
        name = doc.metadata.get("source", "Unknown")
        score = doc.metadata.get("relevance_score", 0)
        if name not in seen_sources:
            sources_text += f"\n- {name} (Score: {score:.2f})"
            seen_sources.add(name)
            
    return "", history + [[message, answer + sources_text]]

def main():
    with gr.Blocks(title="Fast Offer Finder 🚀") as demo:
        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(label="Question")
        btn = gr.Button("Envoyer")
        
        # Fixed for Gradio 4.x
        btn.click(chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
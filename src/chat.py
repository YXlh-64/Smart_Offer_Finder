import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any
import os
import requests
import json

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel
from langchain.schema import BaseRetriever, Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
import gradio as gr
from pinecone import Pinecone

from .config import get_settings
from .reranker import BGEReranker

load_dotenv()

# Global variables for chain and settings
chain = None
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
    """Use Ollama embeddings (multilingual-e5-base, local, no API needed)."""
    model_name = settings.embedding_model
    return OllamaEmbeddings(
        model=model_name,
        base_url=settings.ollama_base_url
    )


def load_vectorstore(settings) -> PineconeVectorStore:
    """Load Pinecone vector store."""
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to start the chat interface.")
    
    embeddings = choose_embeddings(settings)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    # Create vector store from existing index
    vectorstore = PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=""
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
        return DeepseekLLM(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            api_url=settings.llm_base_url.rstrip('/') + '/chat/completions'
            if settings.llm_base_url 
            else "https://api.modelarts-maas.com/v2/chat/completions",
            temperature=0.7
        )


class RerankerRetriever(BaseRetriever):
    """Custom retriever that uses reranking to improve retrieval quality."""
    
    vectorstore: PineconeVectorStore
    reranker: BGEReranker
    initial_k: int = 20
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve documents and rerank them."""
        # First, retrieve more documents than needed
        initial_docs = self.vectorstore.similarity_search(query, k=self.initial_k)
        
        # Then rerank to get the most relevant ones
        reranked_docs = self.reranker.rerank(query, initial_docs)
        
        return reranked_docs


def build_chain(settings) -> ConversationalRetrievalChain:
    """Build the conversational retrieval chain with reranking."""
    
    llm = build_llm(settings)
    
    vectorstore = load_vectorstore(settings)
    
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
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )


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
        chat_history: List of chat messages (Gradio format)
    
    Returns:
        Tuple of (updated_chat_history, empty_string)
    """
    global chain
    
    if chain is None:
        error_msg = "Chat interface not initialized. Make sure Pinecone is properly configured."
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history

    try:
        result = chain.invoke({"question": message})
        answer = result.get("answer", "No response generated.")
        source_docs = result.get("source_documents", []) or []
        
        # Format sources
        sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
        sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {source}" for source in sources) if sources else ""
        
        full_response = answer + sources_text
        
        if chat_history is None:
            chat_history = []
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": full_response})
        
        return chat_history
    
    except Exception as exc:
        error_msg = f"Error processing message: {str(exc)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        if chat_history is None:
            chat_history = []
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return chat_history


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
            - [Pinecone](https://www.pinecone.io/) - Vector database
            - [Ollama](https://ollama.ai/) - Local embeddings
            - [Deepseek](https://www.deepseek.com/) - LLM
            - [Gradio](https://gradio.app/) - Web interface
            """
        )
    
    return demo


def main():
    """Main entry point."""
    print("Initializing Smart Offer Finder...")
    
    if not initialize_chain():
        print("\n❌ Failed to initialize chain. Please check:")
        print("   1. PINECONE_API_KEY is set in .env")
        print("   2. LLM_API_KEY is set in .env")
        print("   3. Documents have been ingested: python -m src.ingest")
        sys.exit(1)
    
    print("\n✅ Chain initialized successfully!")
    print("🚀 Launching Gradio interface...")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()

import sys
import argparse
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
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.llms.base import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLanguageModel
import gradio as gr
from pinecone import Pinecone

from .config import get_settings

load_dotenv()

# Global variables for chain and settings
chain = None
settings = None
vectordb_type = "chromadb"  # Default to chromadb


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


def load_vectorstore(settings, db_type: str = "chromadb"):
    """Load vector store (Pinecone or ChromaDB).
    
    Args:
        settings: Configuration settings
        db_type: "pinecone" or "chromadb"
    """
    embeddings = choose_embeddings(settings)
    
    if db_type == "pinecone":
        return load_vectorstore_pinecone(settings, embeddings)
    elif db_type == "chromadb":
        return load_vectorstore_chromadb(settings, embeddings)
    else:
        raise ValueError(f"Unknown database type: {db_type}. Choose 'pinecone' or 'chromadb'")


def load_vectorstore_pinecone(settings, embeddings):
    """Load Pinecone vector store."""
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to start the chat interface.")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    # Create vector store from existing index
    vectorstore = PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=""
    )
    
    return vectorstore


def load_vectorstore_chromadb(settings, embeddings):
    """Load ChromaDB vector store (local)."""
    persist_directory = settings.vectorstore_path
    
    # Load ChromaDB vector store
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name="smart-offer-finder"
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


def build_chain(settings, db_type: str = "chromadb") -> ConversationalRetrievalChain:
    """Build the conversational retrieval chain.
    
    Args:
        settings: Configuration settings
        db_type: "pinecone" or "chromadb"
    """
    
    llm = build_llm(settings)
    
    vectorstore = load_vectorstore(settings, db_type=db_type)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
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


def initialize_chain(db_type: str = "chromadb"):
    """Initialize the chain on startup.
    
    Args:
        db_type: "pinecone" or "chromadb"
    """
    global chain, settings, vectordb_type
    try:
        settings = get_settings()
        vectordb_type = db_type
        chain = build_chain(settings, db_type=db_type)
        print(f"[chat] Chain initialized successfully using {db_type}.")
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
        error_msg = "Chat interface not initialized. Make sure the vector database is properly configured."
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
            f"""
            # Smart Offer Finder
            
            An intelligent RAG-powered chatbot to help you find relevant offers, conventions, and operational guides.
            
            **Vector Database:** {vectordb_type.upper()}
            
            **Note:** Make sure you have ingested documents first.
            - Ingest to ChromaDB: `python -m src.ingest --db chromadb`
            - Ingest to Pinecone: `python -m src.ingest --db pinecone`
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
            - [ChromaDB](https://www.trychroma.com/) / [Pinecone](https://www.pinecone.io/) - Vector database
            - [Ollama](https://ollama.ai/) - Local embeddings
            - [Deepseek](https://www.deepseek.com/) - LLM
            - [Gradio](https://gradio.app/) - Web interface
            """
        )
    
    return demo


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch Smart Offer Finder chat interface"
    )
    parser.add_argument(
        "--db",
        choices=["pinecone", "chromadb"],
        default="chromadb",
        help="Vector database to use (default: chromadb)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    
    args = parser.parse_args()
    
    print("Initializing Smart Offer Finder...")
    
    if not initialize_chain(db_type=args.db):
        print("\n❌ Failed to initialize chain. Please check:")
        if args.db == "pinecone":
            print("   1. PINECONE_API_KEY is set in .env")
        else:
            print("   1. ChromaDB data exists at: data/vectorstore")
        print("   2. LLM_API_KEY is set in .env")
        print(f"   3. Documents have been ingested: python -m src.ingest --db {args.db}")
        sys.exit(1)
    
    print("\n✅ Chain initialized successfully!")
    print("🚀 Launching Gradio interface...")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=True,
    )


if __name__ == "__main__":
    main()

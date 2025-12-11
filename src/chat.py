import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import get_settings

load_dotenv()

app = FastAPI(title="Smart Offer Finder", version="0.1.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


def choose_embeddings(settings):
    """Use HuggingFace embeddings (sentence-transformers, local, no API needed)."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def load_vectorstore(settings) -> FAISS:
    path = Path(settings.vectorstore_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {path}. Run `python -m src.ingest` first."
        )
    embeddings = choose_embeddings(settings)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def build_chain(settings) -> ConversationalRetrievalChain:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to start the chat API.")

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    vectorstore = load_vectorstore(settings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )


try:
    settings = get_settings()
    chain = build_chain(settings)
except Exception as exc:  # pragma: no cover
    print(f"[chat] Startup error: {exc}")
    chain = None


@app.get("/health")
async def health():
    if chain is None:
        raise HTTPException(status_code=500, detail="Chain not initialized.")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if chain is None:
        raise HTTPException(status_code=500, detail="Chain not initialized.")

    result = chain.invoke({"question": request.question})
    answer = result.get("answer", "")
    source_docs = result.get("source_documents", []) or []
    sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
    return ChatResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    if chain is None:
        sys.exit("Chain failed to initialize. Check logs above.")

    uvicorn.run(app, host="0.0.0.0", port=8000)

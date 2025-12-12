"""
Example: Integrating Hybrid Retriever with RAG System

This script demonstrates how to use the hybrid retriever in your existing
RAG system. It shows three different approaches with different weight configurations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.chat import load_vectorstore, build_llm
from src.hybrid_retriever import (
    get_hybrid_retriever_from_vectorstore,
    get_balanced_retriever,
    get_keyword_focused_retriever,
    get_semantic_focused_retriever
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def example_1_basic_hybrid():
    """
    Example 1: Basic Hybrid Retriever with Equal Weights
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Hybrid Retriever (50/50 weights)")
    print("="*80 + "\n")
    
    settings = get_settings()
    
    # Load vectorstore
    print("Loading vectorstore...")
    vectorstore = load_vectorstore(settings)
    
    # Create hybrid retriever with equal weights
    print("\nCreating hybrid retriever...")
    retriever = get_hybrid_retriever_from_vectorstore(
        vectorstore=vectorstore,
        bm25_weight=0.5,      # 50% BM25
        dense_weight=0.5,     # 50% Dense
        bm25_k=10,
        dense_k=10,
        use_reranker=False
    )
    
    # Test retrieval
    print("\nTesting retrieval...")
    test_query = "What offers are available?"
    docs = retriever.invoke(test_query)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Retrieved {len(docs)} documents")
    print(f"\nTop 3 results:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. {doc.page_content[:200]}...")
        print(f"   Source: {doc.metadata.get('source', 'N/A')}")
    
    # Show timing
    if hasattr(retriever, 'timing_data') and retriever.timing_data:
        print(f"\nTiming:")
        for key, value in retriever.timing_data.items():
            print(f"  {key}: {value:.2f}ms")


def example_2_keyword_focused():
    """
    Example 2: Keyword-Focused Retriever (70% BM25, 30% Dense)
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Keyword-Focused Retriever (70/30 weights)")
    print("="*80 + "\n")
    
    settings = get_settings()
    vectorstore = load_vectorstore(settings)
    
    # Create keyword-focused retriever
    print("Creating keyword-focused retriever...")
    retriever = get_hybrid_retriever_from_vectorstore(
        vectorstore=vectorstore,
        bm25_weight=0.7,      # 70% BM25 (keyword)
        dense_weight=0.3,     # 30% Dense (semantic)
        bm25_k=15,            # Get more from BM25
        dense_k=10
    )
    
    # Test with keyword-heavy query
    test_query = "iPhone VPN discount"
    docs = retriever.invoke(test_query)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Retrieved {len(docs)} documents")
    print(f"\nTop 3 results:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. {doc.page_content[:200]}...")


def example_3_semantic_focused():
    """
    Example 3: Semantic-Focused Retriever (30% BM25, 70% Dense)
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Semantic-Focused Retriever (30/70 weights)")
    print("="*80 + "\n")
    
    settings = get_settings()
    vectorstore = load_vectorstore(settings)
    
    # Create semantic-focused retriever
    print("Creating semantic-focused retriever...")
    retriever = get_hybrid_retriever_from_vectorstore(
        vectorstore=vectorstore,
        bm25_weight=0.3,      # 30% BM25
        dense_weight=0.7,     # 70% Dense (semantic)
        bm25_k=10,
        dense_k=15            # Get more from Dense
    )
    
    # Test with semantic query
    test_query = "How can I save money on my phone bill?"
    docs = retriever.invoke(test_query)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Retrieved {len(docs)} documents")
    print(f"\nTop 3 results:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. {doc.page_content[:200]}...")


def example_4_with_reranking():
    """
    Example 4: Hybrid Retriever with Reranking
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Hybrid Retriever with Reranking")
    print("="*80 + "\n")
    
    settings = get_settings()
    vectorstore = load_vectorstore(settings)
    
    # Create hybrid retriever with reranking
    print("Creating hybrid retriever with reranking...")
    retriever = get_hybrid_retriever_from_vectorstore(
        vectorstore=vectorstore,
        bm25_weight=0.5,
        dense_weight=0.5,
        bm25_k=10,
        dense_k=10,
        use_reranker=True,    # Enable reranking
        reranker_top_k=5      # Return top 5 after reranking
    )
    
    # Test retrieval
    test_query = "What are the best offers?"
    docs = retriever.invoke(test_query)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Retrieved {len(docs)} documents (after reranking)")
    print(f"\nTop 3 results:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. {doc.page_content[:200]}...")
    
    # Show timing breakdown
    if hasattr(retriever, 'timing_data') and retriever.timing_data:
        print(f"\nTiming Breakdown:")
        hybrid_time = retriever.timing_data.get('hybrid_search', 0)
        rerank_time = retriever.timing_data.get('reranking', 0)
        total_time = hybrid_time + rerank_time
        
        print(f"  Hybrid Search: {hybrid_time:.2f}ms ({hybrid_time/total_time*100:.1f}%)")
        print(f"  Reranking:     {rerank_time:.2f}ms ({rerank_time/total_time*100:.1f}%)")
        print(f"  Total:         {total_time:.2f}ms")


def example_5_full_rag_chain():
    """
    Example 5: Full RAG Chain with Hybrid Retriever
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete RAG Chain with Hybrid Retriever")
    print("="*80 + "\n")
    
    settings = get_settings()
    
    # Load components
    print("Loading components...")
    vectorstore = load_vectorstore(settings)
    llm = build_llm(settings)
    
    # Create hybrid retriever
    print("Creating hybrid retriever...")
    retriever = get_hybrid_retriever_from_vectorstore(
        vectorstore=vectorstore,
        bm25_weight=0.5,
        dense_weight=0.5,
        use_reranker=True,
        reranker_top_k=5
    )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Build RAG chain
    print("Building RAG chain...")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    # Test conversation
    print("\n" + "-"*80)
    print("Starting conversation...")
    print("-"*80 + "\n")
    
    queries = [
        "What offers do you have?",
        "Tell me about phone deals",
        "What about VPN services?"
    ]
    
    for query in queries:
        print(f"\n👤 User: {query}")
        result = chain.invoke({"question": query})
        answer = result["answer"]
        sources = result.get("source_documents", [])
        
        print(f"🤖 Assistant: {answer[:300]}...")
        print(f"\n📚 Sources: {len(sources)} documents")
        
        # Show timing if available
        if hasattr(retriever, 'timing_data') and retriever.timing_data:
            total = sum(retriever.timing_data.values())
            print(f"⏱️  Retrieval time: {total:.2f}ms")
        
        print("-"*80)


def example_6_compare_weights():
    """
    Example 6: Compare Different Weight Configurations
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Comparing Different Weight Configurations")
    print("="*80 + "\n")
    
    settings = get_settings()
    vectorstore = load_vectorstore(settings)
    
    test_query = "What are the best offers?"
    
    weight_configs = [
        (0.7, 0.3, "Keyword-Focused"),
        (0.5, 0.5, "Balanced"),
        (0.3, 0.7, "Semantic-Focused")
    ]
    
    print(f"Test Query: '{test_query}'\n")
    
    for bm25_w, dense_w, name in weight_configs:
        print(f"\n{name} ({bm25_w*100:.0f}% BM25, {dense_w*100:.0f}% Dense)")
        print("-" * 60)
        
        retriever = get_hybrid_retriever_from_vectorstore(
            vectorstore=vectorstore,
            bm25_weight=bm25_w,
            dense_weight=dense_w,
            bm25_k=10,
            dense_k=10
        )
        
        docs = retriever.invoke(test_query)
        
        print(f"Retrieved: {len(docs)} documents")
        print(f"Top result: {docs[0].page_content[:150]}...")
        
        if hasattr(retriever, 'timing_data') and retriever.timing_data:
            total = sum(retriever.timing_data.values())
            print(f"Time: {total:.2f}ms")


def example_7_using_presets():
    """
    Example 7: Using Preset Retriever Configurations
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Using Preset Configurations")
    print("="*80 + "\n")
    
    settings = get_settings()
    vectorstore = load_vectorstore(settings)
    
    # Note: These presets require the original documents
    # For this example, we'll extract them from vectorstore
    print("Extracting documents from vectorstore...")
    collection = vectorstore._collection
    results = collection.get(include=["documents", "metadatas"])
    
    from langchain.schema import Document
    documents = []
    for doc_text, metadata in zip(results["documents"], results["metadatas"]):
        documents.append(Document(page_content=doc_text, metadata=metadata or {}))
    
    print(f"Extracted {len(documents)} documents\n")
    
    # Test each preset
    presets = [
        ("Balanced", get_balanced_retriever),
        ("Keyword-Focused", get_keyword_focused_retriever),
        ("Semantic-Focused", get_semantic_focused_retriever)
    ]
    
    test_query = "phone offers"
    
    for name, preset_func in presets:
        print(f"\n{name} Preset")
        print("-" * 60)
        
        retriever = preset_func(documents, vectorstore)
        docs = retriever.invoke(test_query)
        
        print(f"Query: '{test_query}'")
        print(f"Retrieved: {len(docs)} documents")
        print(f"Top result: {docs[0].page_content[:150]}...")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("HYBRID RETRIEVER EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Hybrid (50/50)", example_1_basic_hybrid),
        ("Keyword-Focused (70/30)", example_2_keyword_focused),
        ("Semantic-Focused (30/70)", example_3_semantic_focused),
        ("With Reranking", example_4_with_reranking),
        ("Full RAG Chain", example_5_full_rag_chain),
        ("Compare Weights", example_6_compare_weights),
        ("Using Presets", example_7_using_presets)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...")
    print("="*80 + "\n")
    
    try:
        for name, example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        print("✅ All examples completed!")
        print("="*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")


if __name__ == "__main__":
    # Run specific example or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_hybrid,
            example_2_keyword_focused,
            example_3_semantic_focused,
            example_4_with_reranking,
            example_5_full_rag_chain,
            example_6_compare_weights,
            example_7_using_presets
        ]
        
        if 1 <= example_num <= len(examples):
            print(f"\nRunning Example {example_num}...")
            examples[example_num - 1]()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Run all examples
        main()

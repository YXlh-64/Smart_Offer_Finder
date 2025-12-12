#!/usr/bin/env python3
"""
Quick test script to verify hybrid retriever is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 80)
    print("TEST 1: Importing Modules")
    print("=" * 80)
    
    try:
        print("  • Importing rank_bm25...", end=" ")
        from rank_bm25 import BM25Okapi
        print("✅")
        
        print("  • Importing hybrid_retriever...", end=" ")
        from src.hybrid_retriever import (
            get_hybrid_retriever,
            get_hybrid_retriever_from_vectorstore,
            HybridRetriever
        )
        print("✅")
        
        print("  • Importing LangChain components...", end=" ")
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        print("✅")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {str(e)}\n")
        return False


def test_basic_functionality():
    """Test basic hybrid retriever functionality."""
    print("=" * 80)
    print("TEST 2: Basic Functionality")
    print("=" * 80)
    
    try:
        from langchain.schema import Document
        from src.hybrid_retriever import get_hybrid_retriever
        from langchain_chroma import Chroma
        from langchain_community.embeddings.ollama import OllamaEmbeddings
        import chromadb
        from src.config import get_settings
        
        settings = get_settings()
        
        # Create sample documents
        print("\n  • Creating sample documents...", end=" ")
        documents = [
            Document(page_content="iPhone 15 Pro Max available with 20% discount", 
                    metadata={"source": "offer1"}),
            Document(page_content="Best smartphone deals for students", 
                    metadata={"source": "offer2"}),
            Document(page_content="VPN subscription with unlimited bandwidth", 
                    metadata={"source": "offer3"}),
            Document(page_content="Save money on mobile data plans", 
                    metadata={"source": "offer4"}),
            Document(page_content="Apple iPhone discount code for new users", 
                    metadata={"source": "offer5"}),
        ]
        print("✅")
        
        # Create a temporary vectorstore
        print("  • Creating temporary vectorstore...", end=" ")
        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
        
        chroma_client = chromadb.Client()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="test_collection",
            client=chroma_client
        )
        print("✅")
        
        # Create hybrid retriever
        print("  • Creating hybrid retriever (50/50)...", end=" ")
        retriever = get_hybrid_retriever(
            documents=documents,
            vectorstore=vectorstore,
            bm25_weight=0.5,
            dense_weight=0.5,
            bm25_k=3,
            dense_k=3
        )
        print("✅")
        
        # Test retrieval
        print("  • Testing retrieval...", end=" ")
        test_query = "iPhone discount"
        results = retriever.invoke(test_query)
        print("✅")
        
        print(f"\n  Query: '{test_query}'")
        print(f"  Retrieved: {len(results)} documents")
        
        if results:
            print(f"\n  Top result: {results[0].page_content}")
            print(f"  Source: {results[0].metadata.get('source', 'N/A')}")
        
        # Check timing data
        if hasattr(retriever, 'timing_data') and retriever.timing_data:
            print(f"\n  Timing:")
            for key, value in retriever.timing_data.items():
                print(f"    {key}: {value:.2f}ms")
        
        print("\n✅ Basic functionality test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_weight_configurations():
    """Test different weight configurations."""
    print("=" * 80)
    print("TEST 3: Weight Configurations")
    print("=" * 80)
    
    try:
        from langchain.schema import Document
        from src.hybrid_retriever import get_hybrid_retriever
        from langchain_chroma import Chroma
        from langchain_community.embeddings.ollama import OllamaEmbeddings
        import chromadb
        from src.config import get_settings
        
        settings = get_settings()
        
        # Create sample documents
        documents = [
            Document(page_content="iPhone 15 Pro Max", metadata={"id": "1"}),
            Document(page_content="Samsung Galaxy deals", metadata={"id": "2"}),
            Document(page_content="Smartphone discounts", metadata={"id": "3"}),
        ]
        
        # Create vectorstore
        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url
        )
        chroma_client = chromadb.Client()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="test_weights",
            client=chroma_client
        )
        
        # Test different weight configurations
        configs = [
            (0.7, 0.3, "Keyword-Focused"),
            (0.5, 0.5, "Balanced"),
            (0.3, 0.7, "Semantic-Focused")
        ]
        
        test_query = "iPhone"
        
        print(f"\n  Test Query: '{test_query}'\n")
        
        for bm25_w, dense_w, name in configs:
            print(f"  • Testing {name} ({bm25_w}/{dense_w})...", end=" ")
            
            retriever = get_hybrid_retriever(
                documents=documents,
                vectorstore=vectorstore,
                bm25_weight=bm25_w,
                dense_weight=dense_w,
                bm25_k=2,
                dense_k=2
            )
            
            results = retriever.invoke(test_query)
            print(f"✅ ({len(results)} docs)")
        
        print("\n✅ Weight configuration test passed!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("HYBRID RETRIEVER - QUICK TEST")
    print("="*80 + "\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Functionality Test", test_basic_functionality),
        ("Weight Config Test", test_weight_configurations)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {name} crashed: {str(e)}\n")
            failed += 1
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total:  {passed + failed}")
    print("=" * 80 + "\n")
    
    if failed == 0:
        print("🎉 All tests passed! Hybrid retriever is ready to use.\n")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

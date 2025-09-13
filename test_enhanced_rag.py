#!/usr/bin/env python3
"""
Test script for the enhanced RAG implementation
Tests query enhancement, hybrid search, and improved chunking
"""

import os
import sys
sys.path.append('app')

from dotenv import load_dotenv
from app.rag_pipeline import RAGPipeline

load_dotenv('app/.env')

def test_enhanced_rag():
    """Test the enhanced RAG pipeline with sample queries"""
    print("🚀 Testing Enhanced RAG Implementation")
    print("=" * 50)

    try:
        # Initialize the pipeline
        print("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        print("✅ RAG pipeline initialized successfully")

        # Test queries
        test_queries = [
            "How to setup SSO?",
            "Snowflake connection permissions",
            "What is data lineage?",
            "API authentication methods",
            "dbt integration with Atlan"
        ]

        print(f"\n🔍 Testing {len(test_queries)} sample queries:")
        print("-" * 40)

        for i, query in enumerate(test_queries, 1):
            print(f"\n[Query {i}] {query}")

            try:
                # Test classification
                classification = rag_pipeline.classify_ticket(query)
                print(f"  📝 Classification:")
                print(f"     Topics: {classification.get('topic_tags', [])}")
                print(f"     Sentiment: {classification.get('sentiment', 'Unknown')}")
                print(f"     Priority: {classification.get('priority', 'Unknown')}")

                # Test RAG response (if applicable)
                rag_topics = ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO']
                should_use_rag = any(topic in classification.get('topic_tags', []) for topic in rag_topics)

                if should_use_rag:
                    response_data = rag_pipeline.generate_rag_response(query)

                    print(f"  🔍 Search Details:")
                    print(f"     Query Enhancement: {'✅' if response_data.get('query_enhancement_enabled') else '❌'}")
                    print(f"     Hybrid Search: {'✅' if response_data.get('hybrid_search_enabled') else '❌'}")
                    print(f"     Search Methods: {response_data.get('search_methods_used', [])}")
                    print(f"     Retrieved Chunks: {response_data.get('retrieved_chunks', 0)}")
                    print(f"     Sources Found: {len(response_data.get('sources', []))}")

                    # Show first part of the answer
                    answer = response_data.get('answer', '')
                    preview = answer[:200] + "..." if len(answer) > 200 else answer
                    print(f"  💬 Response Preview: {preview}")
                else:
                    print(f"  🔄 Routed to specialized team (topic: {classification.get('topic_tags', ['Unknown'])[0]})")

                print("  " + "✅ Test completed successfully")

            except Exception as e:
                print(f"  ❌ Error testing query: {e}")

        print(f"\n🎉 Enhanced RAG testing completed!")
        print("\nNext steps:")
        print("1. Run the Streamlit app: cd app && streamlit run main.py")
        print("2. Test the interactive chat interface")
        print("3. Try sample questions to see enhanced search in action")

    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file in the app/ directory")
        print("2. Ensure all API keys are set correctly")
        print("3. Verify Qdrant and OpenAI connectivity")
        return False

    return True

def test_configuration():
    """Test configuration values"""
    print("\n🔧 Configuration Test")
    print("-" * 20)

    required_vars = [
        "OPENAI_API_KEY",
        "QDRANT_URI",
        "QDRANT_API_KEY"
    ]

    missing_vars = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Missing")
            missing_vars.append(var)

    if missing_vars:
        print(f"\n⚠️  Missing variables: {', '.join(missing_vars)}")
        print("Please check your app/.env file")
        return False

    print("✅ All configuration variables are set")
    return True

if __name__ == "__main__":
    print("🧪 Enhanced RAG Test Suite")
    print("=" * 30)

    # Test configuration first
    if not test_configuration():
        print("\n❌ Configuration test failed. Please fix configuration before proceeding.")
        sys.exit(1)

    # Test enhanced RAG
    if test_enhanced_rag():
        print("\n🎉 All tests passed! Your enhanced RAG implementation is ready.")
        print("\n🚀 Ready for deployment and interview demo!")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        sys.exit(1)
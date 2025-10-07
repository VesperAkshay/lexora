#!/usr/bin/env python3
"""
Demonstration of the RAG tools (create_corpus and add_data) usage.

This example shows how to:
1. Create a new document corpus
2. Add documents to the corpus with automatic embedding generation
3. Use the tools in a typical RAG workflow
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexora.tools import (
    CreateCorpusTool, AddDataTool, RAGQueryTool, ListCorporaTool,
    GetCorpusInfoTool, DeleteDocumentTool, DeleteCorpusTool,
    UpdateDocumentTool, BulkAddDataTool, HealthCheckTool
)
from lexora.vector_db.base_vector_db import MockVectorDB
from lexora.utils.embeddings import create_mock_embedding_manager
from lexora.utils.chunking import TextChunker


async def main():
    """Demonstrate RAG tools usage."""
    print("🚀 RAG Tools Demonstration\n")
    
    # Set up dependencies
    print("📋 Setting up dependencies...")
    vector_db = MockVectorDB()
    await vector_db.connect()
    
    embedding_manager = create_mock_embedding_manager(dimension=384)
    text_chunker = TextChunker()
    
    # Create tools
    create_corpus_tool = CreateCorpusTool(vector_db=vector_db)
    add_data_tool = AddDataTool(
        vector_db=vector_db,
        embedding_manager=embedding_manager,
        text_chunker=text_chunker
    )
    rag_query_tool = RAGQueryTool(
        vector_db=vector_db,
        embedding_manager=embedding_manager
    )
    list_corpora_tool = ListCorporaTool(vector_db=vector_db)
    get_corpus_info_tool = GetCorpusInfoTool(vector_db=vector_db)
    delete_document_tool = DeleteDocumentTool(vector_db=vector_db)
    delete_corpus_tool = DeleteCorpusTool(vector_db=vector_db)
    update_document_tool = UpdateDocumentTool(vector_db=vector_db, embedding_manager=embedding_manager)
    bulk_add_data_tool = BulkAddDataTool(vector_db=vector_db, embedding_manager=embedding_manager, text_chunker=text_chunker)
    health_check_tool = HealthCheckTool(vector_db=vector_db, embedding_manager=embedding_manager)
    
    print("✅ Dependencies ready\n")
    
    # Step 1: Create a corpus
    print("📁 Step 1: Creating a document corpus...")
    
    corpus_result = await create_corpus_tool.run(
        corpus_name="knowledge_base",
        description="A knowledge base for AI and machine learning topics",
        metadata={
            "domain": "AI/ML",
            "language": "English",
            "created_by": "demo_script"
        },
        similarity_metric="cosine"
    )
    
    if corpus_result.status.value == "success":
        print(f"✅ Corpus created: {corpus_result.data['corpus_name']}")
        print(f"   Description: {corpus_result.data['creation_parameters']['description']}")
        print(f"   Metadata: {corpus_result.data['creation_parameters']['metadata']}")
    else:
        print(f"❌ Failed to create corpus: {corpus_result.error}")
        return
    
    print()
    
    # Step 2: Add documents to the corpus
    print("📄 Step 2: Adding documents to the corpus...")
    
    # Sample documents about AI/ML topics
    documents = [
        {
            "id": "ml_intro",
            "content": """
            Machine Learning is a subset of artificial intelligence that enables computers to learn 
            and make decisions from data without being explicitly programmed. It involves algorithms 
            that can identify patterns in data and make predictions or classifications based on those patterns.
            """,
            "metadata": {"topic": "machine_learning", "difficulty": "beginner"}
        },
        {
            "id": "neural_networks",
            "content": """
            Neural networks are computing systems inspired by biological neural networks. They consist 
            of interconnected nodes (neurons) that process information using a connectionist approach. 
            Deep learning uses neural networks with multiple layers to model and understand complex patterns.
            """,
            "metadata": {"topic": "neural_networks", "difficulty": "intermediate"}
        },
        {
            "id": "nlp_overview",
            "content": """
            Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers 
            understand, interpret and manipulate human language. NLP combines computational linguistics 
            with statistical, machine learning, and deep learning models to enable computers to process 
            human language in a valuable way.
            """,
            "metadata": {"topic": "nlp", "difficulty": "intermediate"}
        }
    ]
    
    add_result = await add_data_tool.run(
        corpus_name="knowledge_base",
        documents=documents,
        chunk_documents=True,
        chunk_size=200,
        chunk_overlap=50,
        generate_embeddings=True,
        batch_size=10
    )
    
    if add_result.status.value == "success":
        print(f"✅ Documents added successfully!")
        print(f"   Documents added: {add_result.data['documents_added']}")
        print(f"   Chunks created: {add_result.data['chunks_created']}")
        print(f"   Embeddings generated: {add_result.data['embeddings_generated']}")
        print(f"   Final document count: {add_result.data['final_document_count']}")
    else:
        print(f"❌ Failed to add documents: {add_result.error}")
        return
    
    print()
    
    # Step 3: Add a single document
    print("📝 Step 3: Adding a single document...")
    
    single_doc_result = await add_data_tool.run(
        corpus_name="knowledge_base",
        content="""
        Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language 
        models with external knowledge retrieval. RAG systems first retrieve relevant information from 
        a knowledge base, then use that information to generate more accurate and contextually relevant responses.
        """,
        document_id="rag_explanation",
        metadata={"topic": "rag", "difficulty": "advanced", "source": "demo"},
        chunk_documents=False,
        generate_embeddings=True
    )
    
    if single_doc_result.status.value == "success":
        print(f"✅ Single document added successfully!")
        print(f"   Documents added: {single_doc_result.data['documents_added']}")
        print(f"   Final document count: {single_doc_result.data['final_document_count']}")
    else:
        print(f"❌ Failed to add single document: {single_doc_result.error}")
    
    print()
    
    # Step 4: Show corpus information
    print("📊 Step 4: Corpus information...")
    
    try:
        corpus_info = await vector_db.get_corpus_info("knowledge_base")
        print(f"✅ Corpus: {corpus_info.name}")
        print(f"   Total documents: {corpus_info.document_count}")
        print(f"   Created at: {corpus_info.created_at}")
        print(f"   Metadata: {corpus_info.metadata}")
    except Exception as e:
        print(f"❌ Failed to get corpus info: {e}")
    
    print()
    
    # Step 5: List all corpora
    print("📋 Step 5: Listing all corpora...")
    
    list_result = await list_corpora_tool.run(
        include_details=True,
        sort_by="document_count",
        sort_order="desc"
    )
    
    if list_result.status.value == "success":
        print(f"✅ Found {list_result.data['total_count']} corpora:")
        for corpus in list_result.data["corpora"]:
            print(f"   • {corpus['name']}: {corpus['document_count']} documents")
            if corpus.get('metadata'):
                print(f"     Metadata: {corpus['metadata']}")
    else:
        print(f"❌ Failed to list corpora: {list_result.error}")
    
    print()
    
    # Step 6: Perform RAG queries
    print("🔍 Step 6: Performing RAG queries...")
    
    # Query about machine learning
    query_result = await rag_query_tool.run(
        query="machine learning algorithms",
        corpus_name="knowledge_base",
        top_k=3,
        include_metadata=True
    )
    
    if query_result.status.value == "success":
        print(f"✅ Query: 'machine learning algorithms'")
        print(f"   Found {len(query_result.data['results'])} results")
        for i, result in enumerate(query_result.data["results"][:2]):  # Show first 2
            print(f"   Result {i+1}: {result['document_id']} (score: {result['score']:.3f})")
            print(f"   Content preview: {result['content'][:100]}...")
    else:
        print(f"❌ Failed to query: {query_result.error}")
    
    # Query with metadata filters
    filtered_query_result = await rag_query_tool.run(
        query="neural networks",
        corpus_name="knowledge_base",
        metadata_filters={"difficulty": "intermediate"},
        top_k=2
    )
    
    if filtered_query_result.status.value == "success":
        print(f"✅ Filtered query for intermediate difficulty:")
        print(f"   Found {len(filtered_query_result.data['results'])} results")
    else:
        print(f"❌ Failed filtered query: {filtered_query_result.error}")
    
    print()
    
    # Step 7: Get detailed corpus information
    print("📊 Step 7: Getting detailed corpus information...")
    
    corpus_info_result = await get_corpus_info_tool.run(
        corpus_name="knowledge_base",
        include_statistics=True,
        include_sample_documents=True,
        sample_size=2
    )
    
    if corpus_info_result.status.value == "success":
        info = corpus_info_result.data
        print(f"✅ Corpus '{info['corpus_name']}' details:")
        print(f"   Documents: {info['document_count']}")
        print(f"   Created: {info['created_at'][:10]}")
        print(f"   Age: {info.get('age_days', 'unknown')} days")
        print(f"   Empty: {info['is_empty']}")
        if info.get('sample_documents'):
            print(f"   Sample documents: {len(info['sample_documents'])}")
    else:
        print(f"❌ Failed to get corpus info: {corpus_info_result.error}")
    
    print()
    
    # Step 8: Demonstrate document deletion
    print("🗑️ Step 8: Demonstrating document deletion...")
    
    # First, let's see what documents we have
    query_result = await rag_query_tool.run(
        query="",  # Empty query to get any documents
        corpus_name="knowledge_base",
        top_k=5
    )
    
    if query_result.status.value == "success" and query_result.data["results"]:
        # Delete one document
        doc_to_delete = query_result.data["results"][0]["document_id"]
        
        delete_result = await delete_document_tool.run(
            corpus_name="knowledge_base",
            document_id=doc_to_delete,
            confirm_deletion=True,
            return_deleted_info=True
        )
        
        if delete_result.status.value == "success":
            print(f"✅ Deleted document: {doc_to_delete}")
            print(f"   Documents remaining: {delete_result.data.get('final_document_count', 'unknown')}")
        else:
            print(f"❌ Failed to delete document: {delete_result.error}")
    else:
        print("ℹ️ No documents available for deletion demo")
    
    print()
    
    # Step 9: Demonstrate corpus management
    print("📁 Step 9: Demonstrating corpus management...")
    
    # Create a temporary corpus for deletion demo
    temp_create_result = await create_corpus_tool.run(
        corpus_name="temp_demo_corpus",
        description="Temporary corpus for deletion demonstration"
    )
    
    if temp_create_result.status.value == "success":
        print("✅ Created temporary corpus for deletion demo")
        
        # Delete the temporary corpus
        delete_corpus_result = await delete_corpus_tool.run(
            corpus_name="temp_demo_corpus",
            confirm_deletion=True,
            confirmation_phrase="temp_demo_corpus"
        )
        
        if delete_corpus_result.status.value == "success":
            print("✅ Successfully deleted temporary corpus")
        else:
            print(f"❌ Failed to delete corpus: {delete_corpus_result.error}")
    else:
        print(f"❌ Failed to create temporary corpus: {temp_create_result.error}")
    
    print()
    
    # Step 10: Demonstrate optional tools
    print("🔧 Step 10: Demonstrating optional tools...")
    
    # Health check
    health_result = await health_check_tool.run(
        check_vector_db=True,
        check_embeddings=True,
        detailed_diagnostics=True
    )
    
    if health_result.status.value == "success":
        overall_status = health_result.data["overall_status"]
        components = health_result.data["components"]
        print(f"✅ System health check: {overall_status}")
        print(f"   Vector DB: {components.get('vector_database', {}).get('status', 'unknown')}")
        print(f"   Embeddings: {components.get('embedding_service', {}).get('status', 'unknown')}")
    else:
        print(f"❌ Health check failed: {health_result.error}")
    
    # Update document (if we have any)
    query_result = await rag_query_tool.run(
        query="",  # Try to get any documents
        corpus_name="knowledge_base",
        top_k=1
    )
    
    if query_result.status.value == "success" and query_result.data.get("results"):
        doc_id = query_result.data["results"][0]["document_id"]
        
        update_result = await update_document_tool.run(
            corpus_name="knowledge_base",
            document_id=doc_id,
            new_content="This document has been updated to demonstrate the update functionality.",
            new_metadata={"updated": True, "demo": True}
        )
        
        if update_result.status.value == "success":
            print(f"✅ Updated document: {doc_id}")
        else:
            print(f"❌ Document update failed: {update_result.error}")
    else:
        print("ℹ️ No documents available for update demo")
    
    # Bulk add data
    bulk_documents = [
        {"id": f"bulk_demo_{i}", "content": f"Bulk demo document {i} with content for testing."}
        for i in range(5)
    ]
    
    bulk_result = await bulk_add_data_tool.run(
        corpus_name="knowledge_base",
        documents=bulk_documents,
        batch_size=3,
        generate_embeddings=True,
        progress_reporting=True
    )
    
    if bulk_result.status.value == "success":
        summary = bulk_result.data["bulk_processing_summary"]
        print(f"✅ Bulk processing: {summary['successful_additions']} documents added")
        print(f"   Performance: {bulk_result.data['performance_metrics']['documents_per_second']:.2f} docs/sec")
    else:
        print(f"❌ Bulk processing failed: {bulk_result.error}")
    
    print()
    
    # Clean up
    await vector_db.disconnect()
    print("🎉 RAG Tools demonstration completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Corpus creation with metadata and configuration")
    print("   • Batch document addition with automatic chunking")
    print("   • Automatic embedding generation")
    print("   • Single document addition")
    print("   • Corpus listing and discovery")
    print("   • Semantic search and RAG queries")
    print("   • Metadata filtering and result ranking")
    print("   • Detailed corpus information and statistics")
    print("   • Document deletion with validation")
    print("   • Corpus deletion with safety measures")
    print("   • Document updates with re-embedding")
    print("   • Bulk data processing with performance optimization")
    print("   • Comprehensive system health monitoring")
    print("   • Error handling and validation")
    print("   • Structured tool responses")


if __name__ == "__main__":
    asyncio.run(main())
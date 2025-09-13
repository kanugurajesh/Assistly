import os
import sys
import argparse
from typing import List, Dict
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import time
from datetime import datetime

load_dotenv()

# Initialize clients
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client.get_database("Cluster0")
collection = db.get_collection("atlan_developer_docs")  # Default collection, can be overridden by args

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URI"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model
COLLECTION_NAME = "atlan_docs"
VECTOR_SIZE = 384  # BGE small model vector size
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Initialize FastEmbed model
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

def create_qdrant_collection(recreate=False):
    """Create or check Qdrant collection for Atlan docs"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if collection_exists and recreate:
            # Delete existing collection if recreate is True
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
            collection_exists = False
        elif collection_exists:
            print(f"Collection '{COLLECTION_NAME}' already exists")
            return True
        
        if not collection_exists:
            # Create new collection
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"Created collection: {COLLECTION_NAME}")
        
        return True
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False

def chunk_text(text: str, metadata: Dict) -> List[Dict]:
    """Chunk text using RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    
    chunks = text_splitter.split_text(text)
    
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Skip empty chunks
            chunk_doc = {
                "text": chunk,
                "source_url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "doc_type": "developer",  # Will be updated based on source
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunked_docs.append(chunk_doc)
    
    return chunked_docs

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using FastEmbed with improved progress tracking and error handling"""
    if not texts:
        return []
    
    try:
        print(f"Generating embeddings for {len(texts)} chunks using FastEmbed...")
        print("Note: First run may take time to download the model...")
        
        # Process in smaller batches for progress tracking and memory management
        batch_size = 50  # Reduced for better memory management
        all_embeddings = []
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch_start_time = time.time()
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_embeddings = list(embedding_model.embed(batch_texts))
                all_embeddings.extend(batch_embeddings)
                
                progress = min(i + batch_size, len(texts))
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                
                # Estimate remaining time
                if progress > 0:
                    eta = (elapsed_time / progress) * (len(texts) - progress)
                    print(f"Progress: {progress}/{len(texts)} embeddings ({progress/len(texts)*100:.1f}%) | Batch time: {batch_time:.2f}s | ETA: {eta:.1f}s")
                
            except Exception as batch_error:
                print(f"Error processing batch {i//batch_size + 1}: {batch_error}")
                # Add zero vectors for failed batch
                fallback_embeddings = [[0.0] * VECTOR_SIZE for _ in batch_texts]
                all_embeddings.extend(fallback_embeddings)
        
        total_time = time.time() - start_time
        print(f"Successfully generated {len(all_embeddings)} embeddings in {total_time:.2f} seconds")
        return all_embeddings
        
    except Exception as e:
        print(f"Critical error generating embeddings: {e}")
        # Fallback to zero vectors for all texts
        print(f"Using zero vectors as fallback for {len(texts)} texts")
        return [[0.0] * VECTOR_SIZE for _ in texts]

def get_existing_mongodb_ids():
    """Get MongoDB IDs that are already in Qdrant to avoid duplicates"""
    try:
        # Get all points from Qdrant with mongodb_id
        scroll_result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,  # Adjust based on your needs
            with_payload=["mongodb_id"]
        )
        
        existing_ids = set()
        for point in scroll_result[0]:
            mongodb_id = point.payload.get("mongodb_id")
            if mongodb_id:
                existing_ids.add(mongodb_id)
        
        return existing_ids
    except Exception as e:
        print(f"Warning: Could not check existing vectors: {e}")
        return set()

def process_mongodb_documents(source_url_filter=None, incremental=True):
    """Load documents from MongoDB and process them with incremental support"""
    print("Loading documents from MongoDB...")
    
    # Build query filter
    query_filter = {}
    if source_url_filter:
        query_filter["source_url"] = source_url_filter
        print(f"Filtering by source URL: {source_url_filter}")
    
    # Get all documents from scraped_pages collection
    documents = list(collection.find(query_filter))
    print(f"Found {len(documents)} documents in MongoDB")
    
    if not documents:
        print("No documents found in MongoDB. Please run scrape.py first.")
        return []
    
    # Check for existing vectors if incremental is enabled
    existing_mongodb_ids = set()
    if incremental:
        existing_mongodb_ids = get_existing_mongodb_ids()
        print(f"Found {len(existing_mongodb_ids)} existing vectors in Qdrant")
    
    all_chunks = []
    skipped_count = 0
    
    for doc_idx, doc in enumerate(documents):
        # Skip if already processed (incremental mode)
        doc_id_str = str(doc["_id"])
        if incremental and doc_id_str in existing_mongodb_ids:
            skipped_count += 1
            continue
        try:
            markdown_content = doc.get("markdown", "")
            metadata = doc.get("metadata", {})
            
            if not markdown_content.strip():
                continue
            
            # Determine doc_type based on source URL
            source_url = doc.get("source_url", "")
            doc_type = "developer" if "developer.atlan.com" in source_url else "docs"
            
            # Chunk the document
            chunks = chunk_text(markdown_content, metadata)
            
            # Update doc_type for all chunks
            for chunk in chunks:
                chunk["doc_type"] = doc_type
                chunk["mongodb_id"] = str(doc["_id"])
            
            all_chunks.extend(chunks)
            
            if (doc_idx + 1) % 10 == 0 or doc_idx == len(documents) - 1:
                print(f"Processed {doc_idx + 1}/{len(documents)} documents: {metadata.get('title', 'Untitled')} ({len(chunks)} chunks)")
            
        except Exception as e:
            print(f"Error processing document {doc_idx}: {e}")
            continue
    
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Skipped existing documents: {skipped_count}")
    return all_chunks

def ingest_to_qdrant(chunks: List[Dict]):
    """Ingest chunks with embeddings to Qdrant with improved error handling"""
    if not chunks:
        print("No chunks to ingest")
        return
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    # Extract texts for embedding generation
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batches
    embeddings = generate_embeddings(texts)
    
    if len(embeddings) != len(chunks):
        print(f"Error: Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        return
    
    print("Creating Qdrant points...")
    
    # Get the next available ID in Qdrant
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        next_id = info.points_count
    except:
        next_id = 0
    
    # Create points for Qdrant
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=next_id + i,  # Use sequential IDs to avoid conflicts
            vector=embedding,
            payload={
                "text": chunk["text"],
                "source_url": chunk["source_url"],
                "title": chunk["title"],
                "doc_type": chunk["doc_type"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "mongodb_id": chunk.get("mongodb_id", ""),
                "ingested_at": datetime.now().isoformat()
            }
        )
        points.append(point)
    
    # Upload to Qdrant in batches with error handling
    batch_size = 50  # Reduced batch size for better reliability
    successful_batches = 0
    failed_batches = 0
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(points) - 1) // batch_size + 1
        
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            successful_batches += 1
            print(f"✅ Uploaded batch {batch_num}/{total_batches} ({len(batch)} points)")
        except Exception as e:
            failed_batches += 1
            print(f"❌ Error uploading batch {batch_num}/{total_batches}: {e}")
            # Continue with next batch instead of failing completely
    
    print(f"\n📊 Ingestion Summary:")
    print(f"Total batches: {successful_batches + failed_batches}")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {failed_batches}")
    print(f"Estimated successful points: {successful_batches * batch_size}")

def main():
    """Main ingestion pipeline"""
    parser = argparse.ArgumentParser(description='Ingest MongoDB documents to Qdrant vector database')
    parser.add_argument('--source-url', help='Filter by source URL (e.g., https://docs.atlan.com)')
    parser.add_argument('--recreate', action='store_true', help='Recreate Qdrant collection (deletes existing data)')
    parser.add_argument('--no-incremental', action='store_true', help='Disable incremental processing (process all documents)')
    parser.add_argument('--collection', default='atlan_developer_docs', help='MongoDB collection name (default: atlan_developer_docs)')
    
    args = parser.parse_args()
    
    print("🚀 Starting MongoDB to Qdrant ingestion pipeline...")
    if args.source_url:
        print(f"🌐 Source URL filter: {args.source_url}")
    print(f"🗂️ MongoDB collection: {args.collection}")
    print(f"🔄 Incremental processing: {not args.no_incremental}")
    print(f"♾️ Recreate collection: {args.recreate}")
    print("=" * 50)
    
    # Use specified collection
    global collection
    collection = db.get_collection(args.collection)
    
    # Step 1: Create Qdrant collection
    if not create_qdrant_collection(recreate=args.recreate):
        print("Failed to create Qdrant collection. Exiting.")
        return
    
    # Step 2: Process MongoDB documents
    start_time = time.time()
    chunks = process_mongodb_documents(
        source_url_filter=args.source_url,
        incremental=not args.no_incremental
    )
    
    if not chunks:
        print("No chunks to process. Exiting.")
        return
    
    processing_time = time.time() - start_time
    print(f"⏱️ Document processing completed in {processing_time:.2f} seconds")
    
    # Step 3: Ingest to Qdrant
    print(f"🚀 Starting vector ingestion for {len(chunks)} chunks...")
    start_time = time.time()
    ingest_to_qdrant(chunks)
    ingestion_time = time.time() - start_time
    print(f"⏱️ Vector ingestion completed in {ingestion_time:.2f} seconds")
    
    # Step 4: Verify ingestion
    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"\n✅ Ingestion complete!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total points: {info.points_count}")
    print(f"Vector size: {info.config.params.vectors.size}")
    print(f"Total processing time: {(processing_time + ingestion_time):.2f} seconds")
    
    # Close connections
    mongo_client.close()

if __name__ == "__main__":
    main()
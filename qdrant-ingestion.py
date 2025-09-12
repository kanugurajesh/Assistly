import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

load_dotenv()

# Initialize clients
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client.get_database("Cluster0")
collection = db.get_collection("scraped_pages")

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

def create_qdrant_collection():
    """Create or recreate Qdrant collection for Atlan docs"""
    try:
        # Delete existing collection if it exists
        try:
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except:
            pass
        
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
    """Generate embeddings using FastEmbed with progress tracking"""
    try:
        print(f"Generating embeddings for {len(texts)} chunks using FastEmbed...")
        print("Note: First run may take time to download the model...")
        
        # Process in smaller batches for progress tracking
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = list(embedding_model.embed(batch_texts))
            all_embeddings.extend(batch_embeddings)
            
            progress = min(i + batch_size, len(texts))
            print(f"Progress: {progress}/{len(texts)} embeddings generated ({progress/len(texts)*100:.1f}%)")
        
        print(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Fallback to zero vectors
        return [[0.0] * VECTOR_SIZE for _ in texts]

def process_mongodb_documents():
    """Load documents from MongoDB and process them"""
    print("Loading documents from MongoDB...")
    
    # Get all documents from scraped_pages collection
    documents = list(collection.find({}))
    print(f"Found {len(documents)} documents in MongoDB")
    
    if not documents:
        print("No documents found in MongoDB. Please run scrape.py first.")
        return []
    
    all_chunks = []
    
    for doc_idx, doc in enumerate(documents):
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
            
            print(f"Processed document {doc_idx + 1}/{len(documents)}: {metadata.get('title', 'Untitled')} ({len(chunks)} chunks)")
            
        except Exception as e:
            print(f"Error processing document {doc_idx}: {e}")
            continue
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def ingest_to_qdrant(chunks: List[Dict]):
    """Ingest chunks with embeddings to Qdrant"""
    if not chunks:
        print("No chunks to ingest")
        return
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    # Extract texts for embedding generation
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batches
    embeddings = generate_embeddings(texts)
    
    print("Creating Qdrant points...")
    
    # Create points for Qdrant
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": chunk["text"],
                "source_url": chunk["source_url"],
                "title": chunk["title"],
                "doc_type": chunk["doc_type"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "mongodb_id": chunk.get("mongodb_id", "")
            }
        )
        points.append(point)
    
    # Upload to Qdrant in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            print(f"Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}")
        except Exception as e:
            print(f"Error uploading batch {i // batch_size + 1}: {e}")
    
    print(f"Successfully ingested {len(points)} points to Qdrant")

def main():
    """Main ingestion pipeline"""
    print("🚀 Starting MongoDB to Qdrant ingestion pipeline...")
    print("=" * 50)
    
    # Step 1: Create Qdrant collection
    if not create_qdrant_collection():
        print("Failed to create Qdrant collection. Exiting.")
        return
    
    # Step 2: Process MongoDB documents
    chunks = process_mongodb_documents()
    
    if not chunks:
        print("No chunks to process. Exiting.")
        return
    
    # Step 3: Ingest to Qdrant
    ingest_to_qdrant(chunks)
    
    # Step 4: Verify ingestion
    info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"\n✅ Ingestion complete!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total points: {info.points_count}")
    print(f"Vector size: {info.config.params.vectors.size}")
    
    # Close connections
    mongo_client.close()

if __name__ == "__main__":
    main()
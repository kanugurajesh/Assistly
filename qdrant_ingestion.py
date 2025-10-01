import os
import argparse
import logging
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from utils import get_mongodb_collection, close_mongodb_client

# Load environment variables from app/.env for deployment-ready structure
load_dotenv(os.path.join(os.path.dirname(__file__), 'app', '.env'))

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging with console and file handlers"""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler for detailed logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / f"qdrant_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger

# Initialize logger
logger = setup_logging()

MONGODB_DB = "Cluster0"
MONGODB_COLLECTION = "atlan_developer_docs"

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URI"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Configuration constants
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model
COLLECTION_NAME = "atlan_docs"
VECTOR_SIZE = 384  # BGE small model vector size
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 50  # Embedding and ingestion batch size
SCROLL_BATCH_SIZE = 1000  # Batch size for paginated scroll
PROGRESS_INTERVAL = 10  # Print progress every N documents

# Timestamp tracking configuration
CONFIG_DIR = Path("config")
LAST_INGESTION_FILE = CONFIG_DIR / "last_ingestion.txt"
LAST_FULL_REINDEX_FILE = CONFIG_DIR / "last_full_reindex.txt"
FULL_REINDEX_INTERVAL_DAYS = 7

# Initialize FastEmbed model
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

def read_timestamp(file_path: Path) -> datetime:
    """Read timestamp from file, return epoch if doesn't exist"""
    try:
        if file_path.exists():
            timestamp_str = file_path.read_text().strip()
            return datetime.fromisoformat(timestamp_str)
    except Exception as e:
        logger.warning(f"Could not read timestamp from {file_path}: {e}")
    # Default to epoch if file doesn't exist or has error
    return datetime(1970, 1, 1, tzinfo=timezone.utc)

def write_timestamp(file_path: Path, timestamp: datetime) -> None:
    """Write timestamp to file"""
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        file_path.write_text(timestamp.isoformat())
        logger.debug(f"Wrote timestamp to {file_path}: {timestamp.isoformat()}")
    except Exception as e:
        logger.warning(f"Could not write timestamp to {file_path}: {e}")

def should_do_full_reindex() -> bool:
    """Check if it's time for weekly full reindex"""
    last_full = read_timestamp(LAST_FULL_REINDEX_FILE)
    days_since = (datetime.now(timezone.utc) - last_full).days
    return days_since >= FULL_REINDEX_INTERVAL_DAYS

def create_qdrant_collection(collection_name: str, recreate: bool = False) -> bool:
    """Create or check Qdrant collection"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)

        if collection_exists and recreate:
            # Delete existing collection if recreate is True
            qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
            collection_exists = False
        elif collection_exists:
            logger.info(f"Collection '{collection_name}' already exists")
            return True

        if not collection_exists:
            # Create new collection
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {collection_name}")

        return True
    except Exception as e:
        logger.error(f"Error creating collection: {e}", exc_info=True)
        return False

def preserve_code_blocks(text: str) -> str:
    """Preserve code blocks by adding extra spacing around them"""
    # Pattern to match code blocks (both ``` and indented)
    code_block_pattern = r'(```[\s\S]*?```|\n    [^\n]*(?:\n    [^\n]*)*)'

    # Add extra newlines before and after code blocks to prevent splitting
    def replace_code_block(match):
        code_block = match.group(0)
        return f"\n\n{code_block}\n\n"

    # Apply the replacement
    preserved_text = re.sub(code_block_pattern, replace_code_block, text)
    return preserved_text

def chunk_text(text: str, metadata: Dict) -> List[Dict]:
    """Chunk text using enhanced RecursiveCharacterTextSplitter with code block preservation"""
    # Step 1: Preserve code blocks by adding extra spacing
    processed_text = preserve_code_blocks(text)

    # Step 2: Enhanced separators that preserve markdown structure and code blocks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n\n",  # Major section breaks
            "\n\n",    # Paragraph breaks
            "\n```\n", # Code block endings
            "```\n",   # Code block starts
            "\n# ",    # Major headers
            "\n## ",   # Section headers
            "\n### ",  # Subsection headers
            "\n#### ", # Sub-subsection headers
            "\n- ",    # List items
            "\n* ",    # Alternative list items
            "\n1. ",   # Numbered lists
            "\n2. ",   # Numbered lists continued
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "? ",      # Question endings
            "! ",      # Exclamation endings
            "; ",      # Semicolon breaks
            ", ",      # Comma breaks (for very long sentences)
            " ",       # Word boundaries
            ""
        ],
        keep_separator=True,
        is_separator_regex=False
    )

    # Step 3: Split the processed text
    chunks = text_splitter.split_text(processed_text)

    # Step 4: Create enhanced chunk metadata with quality indicators
    chunked_docs = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Skip empty chunks
            # Calculate chunk quality metrics
            has_code = '```' in chunk or chunk.count('    ') > 3
            has_headers = any(chunk.startswith(header) for header in ['# ', '## ', '### '])
            word_count = len(chunk.split())

            chunk_doc = {
                "text": chunk.strip(),
                "source_url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "doc_type": "developer",  # Will be updated based on source
                "chunk_index": i,
                "total_chunks": len(chunks),
                "word_count": word_count,
                "has_code": has_code,
                "has_headers": has_headers,
                "chunk_quality": "high" if (has_code or has_headers or word_count > 100) else "medium"
            }
            chunked_docs.append(chunk_doc)

    logger.info(f"Created {len(chunked_docs)} chunks from '{metadata.get('title', 'Unknown')}' ({len([c for c in chunked_docs if c['has_code']])} with code, {len([c for c in chunked_docs if c['has_headers']])} with headers)")
    return chunked_docs

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using FastEmbed with improved progress tracking and error handling"""
    if not texts:
        return []

    try:
        logger.info(f"Generating embeddings for {len(texts)} chunks using FastEmbed...")
        logger.info("Note: First run may take time to download the model...")

        # Process in smaller batches for progress tracking and memory management
        all_embeddings = []
        start_time = time.time()

        for i in range(0, len(texts), BATCH_SIZE):
            batch_start_time = time.time()
            batch_texts = texts[i:i + BATCH_SIZE]

            try:
                batch_embeddings = list(embedding_model.embed(batch_texts))
                all_embeddings.extend(batch_embeddings)

                progress = min(i + BATCH_SIZE, len(texts))
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time

                # Estimate remaining time
                if progress > 0:
                    eta = (elapsed_time / progress) * (len(texts) - progress)
                    logger.info(f"Progress: {progress}/{len(texts)} embeddings ({progress/len(texts)*100:.1f}%) | Batch time: {batch_time:.2f}s | ETA: {eta:.1f}s")

            except Exception as batch_error:
                logger.error(f"Error processing batch {i//BATCH_SIZE + 1}: {batch_error}")
                # Add zero vectors for failed batch
                fallback_embeddings = [[0.0] * VECTOR_SIZE for _ in batch_texts]
                all_embeddings.extend(fallback_embeddings)

        total_time = time.time() - start_time
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings in {total_time:.2f} seconds")
        return all_embeddings

    except Exception as e:
        logger.error(f"Critical error generating embeddings: {e}", exc_info=True)
        # Fallback to zero vectors for all texts
        logger.warning(f"Using zero vectors as fallback for {len(texts)} texts")
        return [[0.0] * VECTOR_SIZE for _ in texts]

def get_existing_mongodb_ids_paginated(collection_name: str) -> set:
    """Get MongoDB IDs that are already in Qdrant using paginated scroll to handle large collections"""
    existing_ids = set()
    offset = None
    batch_count = 0

    try:
        while True:
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=SCROLL_BATCH_SIZE,
                offset=offset,
                with_payload=["mongodb_id"]
            )

            points = scroll_result[0]
            next_offset = scroll_result[1]

            # Process batch
            for point in points:
                mongodb_id = point.payload.get("mongodb_id")
                if mongodb_id:
                    existing_ids.add(mongodb_id)

            batch_count += 1
            if batch_count % 10 == 0:
                logger.debug(f"Scrolled {len(existing_ids)} IDs so far...")

            # Break if no more results
            if next_offset is None or len(points) == 0:
                break

            offset = next_offset

        logger.info(f"Found {len(existing_ids)} existing vectors in Qdrant (checked {batch_count} batches)")
        return existing_ids

    except Exception as e:
        logger.warning(f"Could not check existing vectors: {e}")
        return set()

def process_incremental(collection, last_ingestion_time: datetime, source_url_filter: Optional[str] = None) -> List[Dict]:
    """Fast timestamp-based incremental processing

    Args:
        collection: MongoDB collection object
        last_ingestion_time: Only process documents after this timestamp
        source_url_filter: Optional URL filter for documents

    Returns:
        List of processed document chunks
    """
    logger.info("Running incremental update...")
    logger.info(f"Processing documents crawled after: {last_ingestion_time.isoformat()}")

    # Build query filter
    query_filter = {"crawled_at": {"$gt": last_ingestion_time}}
    if source_url_filter:
        query_filter["source_url"] = source_url_filter
        logger.info(f"Filtering by source URL: {source_url_filter}")

    # Query only new documents
    new_docs = list(collection.find(query_filter))
    logger.info(f"Found {len(new_docs)} new documents to process")

    if not new_docs:
        return []

    # Process new documents
    all_chunks = []
    for doc_idx, doc in enumerate(new_docs):
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

            if (doc_idx + 1) % PROGRESS_INTERVAL == 0 or doc_idx == len(new_docs) - 1:
                logger.info(f"Processed {doc_idx + 1}/{len(new_docs)} documents: {metadata.get('title', 'Untitled')} ({len(chunks)} chunks)")

        except Exception as e:
            logger.error(f"Error processing document {doc_idx}: {e}")
            continue

    logger.info(f"Total chunks created from incremental update: {len(all_chunks)}")
    return all_chunks

def process_full_reindex(collection, qdrant_collection_name: str, source_url_filter: Optional[str] = None) -> List[Dict]:
    """Complete reindex with duplicate detection using scroll

    Args:
        collection: MongoDB collection object
        qdrant_collection_name: Name of the Qdrant collection
        source_url_filter: Optional URL filter for documents

    Returns:
        List of processed document chunks (only for missing documents)
    """
    logger.info("Running weekly full reindex...")

    # Get existing vectors from Qdrant
    existing_ids = get_existing_mongodb_ids_paginated(qdrant_collection_name)

    # Build query filter
    query_filter = {}
    if source_url_filter:
        query_filter["source_url"] = source_url_filter
        logger.info(f"Filtering by source URL: {source_url_filter}")

    # Get all MongoDB documents
    all_docs = list(collection.find(query_filter))
    logger.info(f"Found {len(all_docs)} documents in MongoDB")

    if not all_docs:
        logger.warning("No documents found in MongoDB. Please run scrape.py first.")
        return []

    # Find missing documents (in MongoDB but not in Qdrant)
    docs_to_process = []
    for doc in all_docs:
        doc_id = str(doc["_id"])
        if doc_id not in existing_ids:
            docs_to_process.append(doc)

    logger.info(f"Found {len(docs_to_process)} documents to re-ingest")
    logger.info(f"Skipping {len(all_docs) - len(docs_to_process)} already-indexed documents")

    if not docs_to_process:
        return []

    # Process missing documents
    all_chunks = []
    for doc_idx, doc in enumerate(docs_to_process):
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

            if (doc_idx + 1) % PROGRESS_INTERVAL == 0 or doc_idx == len(docs_to_process) - 1:
                logger.info(f"Processed {doc_idx + 1}/{len(docs_to_process)} documents: {metadata.get('title', 'Untitled')} ({len(chunks)} chunks)")

        except Exception as e:
            logger.error(f"Error processing document {doc_idx}: {e}")
            continue

    logger.info(f"Total chunks created from full reindex: {len(all_chunks)}")
    return all_chunks

def ingest_to_qdrant(chunks: List[Dict], collection_name: str) -> None:
    """Ingest chunks with embeddings to Qdrant with improved error handling"""
    if not chunks:
        logger.info("No chunks to ingest")
        return

    logger.info(f"Generating embeddings for {len(chunks)} chunks...")

    # Extract texts for embedding generation
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings in batches
    embeddings = generate_embeddings(texts)

    if len(embeddings) != len(chunks):
        logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        return

    logger.info("Creating Qdrant points...")

    # Get the next available ID in Qdrant
    try:
        info = qdrant_client.get_collection(collection_name)
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
                "word_count": chunk.get("word_count", 0),
                "has_code": chunk.get("has_code", False),
                "has_headers": chunk.get("has_headers", False),
                "chunk_quality": chunk.get("chunk_quality", "medium"),
                "ingested_at": datetime.now().isoformat()
            }
        )
        points.append(point)
    
    # Upload to Qdrant in batches with error handling
    successful_batches = 0
    failed_batches = 0

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(points) - 1) // BATCH_SIZE + 1

        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
            successful_batches += 1
            logger.info(f"Uploaded batch {batch_num}/{total_batches} ({len(batch)} points)")
        except Exception as e:
            failed_batches += 1
            logger.error(f"Error uploading batch {batch_num}/{total_batches}: {e}")
            # Continue with next batch instead of failing completely

    logger.info("Ingestion Summary:")
    logger.info(f"Total batches: {successful_batches + failed_batches}")
    logger.info(f"Successful batches: {successful_batches}")
    logger.info(f"Failed batches: {failed_batches}")
    logger.info(f"Estimated successful points: {successful_batches * BATCH_SIZE}")

def main() -> None:
    """Main ingestion pipeline with hybrid timestamp + weekly full reindex"""
    parser = argparse.ArgumentParser(description="Ingest MongoDB documents to Qdrant vector database")
    parser.add_argument("--source-url", help="Filter by source URL (e.g., https://docs.atlan.com)")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection (deletes existing data)")
    parser.add_argument("--collection", default="atlan_developer_docs", help="MongoDB collection name (default: atlan_developer_docs)")
    parser.add_argument("--qdrant-collection", default="atlan_docs", help="Qdrant collection name (default: atlan_docs)")
    parser.add_argument("--force-full-reindex", action="store_true", help="Force full reindex regardless of schedule")
    parser.add_argument("--force-incremental", action="store_true", help="Force incremental mode even if full reindex is scheduled")

    args = parser.parse_args()

    current_time = datetime.now(timezone.utc)

    logger.info("Starting MongoDB to Qdrant ingestion pipeline...")
    if args.source_url:
        logger.info(f"Source URL filter: {args.source_url}")
    logger.info(f"MongoDB collection: {args.collection}")
    logger.info(f"Qdrant collection: {args.qdrant_collection}")
    logger.info(f"Recreate collection: {args.recreate}")
    logger.info("=" * 50)
    
    # Get MongoDB connection with specified collection
    mongo_client, db, collection = get_mongodb_collection(
        database_name=MONGODB_DB,
        collection_name=args.collection
    )
    
    # Step 1: Create Qdrant collection
    if not create_qdrant_collection(collection_name=args.qdrant_collection, recreate=args.recreate):
        logger.error("Failed to create Qdrant collection. Exiting.")
        close_mongodb_client(mongo_client)
        return

    # Step 2: Determine processing mode
    start_time = time.time()

    # Determine which mode to use
    if args.force_full_reindex:
        use_full_reindex = True
        logger.info("Forced full reindex mode (--force-full-reindex)")
    elif args.force_incremental:
        use_full_reindex = False
        logger.info("Forced incremental mode (--force-incremental)")
    elif args.recreate:
        use_full_reindex = False
        logger.info("Using incremental mode (collection was recreated)")
    else:
        use_full_reindex = should_do_full_reindex()
        if use_full_reindex:
            last_full = read_timestamp(LAST_FULL_REINDEX_FILE)
            days_since = (current_time - last_full).days
            logger.info(f"Automatic mode selection: Full reindex (last full reindex was {days_since} days ago)")
        else:
            last_ingestion = read_timestamp(LAST_INGESTION_FILE)
            hours_since = (current_time - last_ingestion).total_seconds() / 3600
            logger.info(f"Automatic mode selection: Incremental update (last run was {hours_since:.1f} hours ago)")

    # Process documents based on mode
    if use_full_reindex:
        chunks = process_full_reindex(
            collection=collection,
            qdrant_collection_name=args.qdrant_collection,
            source_url_filter=args.source_url
        )
    else:
        last_ingestion = read_timestamp(LAST_INGESTION_FILE)
        chunks = process_incremental(
            collection=collection,
            last_ingestion_time=last_ingestion,
            source_url_filter=args.source_url
        )

    processing_time = time.time() - start_time

    if not chunks:
        logger.info("No new chunks to process.")
        logger.info(f"Check completed in {processing_time:.2f} seconds")

        # Update timestamps even if no chunks
        write_timestamp(LAST_INGESTION_FILE, current_time)
        if use_full_reindex:
            write_timestamp(LAST_FULL_REINDEX_FILE, current_time)

        close_mongodb_client(mongo_client)
        return

    logger.info(f"Document processing completed in {processing_time:.2f} seconds")

    # Step 3: Ingest to Qdrant
    logger.info(f"Starting vector ingestion for {len(chunks)} chunks...")
    start_time = time.time()
    ingest_to_qdrant(chunks, args.qdrant_collection)
    ingestion_time = time.time() - start_time
    logger.info(f"Vector ingestion completed in {ingestion_time:.2f} seconds")

    # Step 4: Update timestamps
    write_timestamp(LAST_INGESTION_FILE, current_time)
    if use_full_reindex:
        write_timestamp(LAST_FULL_REINDEX_FILE, current_time)
        logger.info("Updated last_full_reindex timestamp")
    logger.info("Updated last_ingestion timestamp")

    # Step 5: Verify ingestion
    info = qdrant_client.get_collection(args.qdrant_collection)
    logger.info("Ingestion complete!")
    logger.info(f"Collection: {args.qdrant_collection}")
    logger.info(f"Total points: {info.points_count}")
    logger.info(f"Vector size: {info.config.params.vectors.size}")
    logger.info(f"Total processing time: {(processing_time + ingestion_time):.2f} seconds")

    # Close connections
    close_mongodb_client(mongo_client)

if __name__ == "__main__":
    main()
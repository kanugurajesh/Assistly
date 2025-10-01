from firecrawl import Firecrawl
from dotenv import load_dotenv
import os
import argparse
from datetime import datetime, timezone
from urllib.parse import urlparse
import json
import logging
import time
from tqdm import tqdm
from utils import get_mongodb_collection, close_mongodb_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from app/.env for deployment-ready structure
load_dotenv(os.path.join(os.path.dirname(__file__), 'app', '.env'))

# Configuration constants from environment
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
CRAWL_TIMEOUT = int(os.getenv("CRAWL_TIMEOUT", "300"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# Initialize with your API key
firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

def get_domain_name(url: str) -> str:
    """Extract domain name from URL for identification"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return "unknown"

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape and store web pages using Firecrawl")
    parser.add_argument("url", help="URL to crawl")
    parser.add_argument("--limit", type=int, default=700, help="Maximum pages to crawl (default: 700)")
    parser.add_argument("--collection", default="atlan_developer_docs", help="MongoDB collection name (default: atlan_developer_docs)")

    args = parser.parse_args()

    # Validate URL format
    if not args.url.startswith(('http://', 'https://')):
        logger.error("Invalid URL format. URL must start with http:// or https://")
        raise ValueError("URL must start with http:// or https://")

    logger.info(f"🚀 Starting crawl of {args.url}")
    logger.info(f"📊 Limit: {args.limit} pages")
    logger.info(f"🗂️ Collection: {args.collection}")
    logger.info("=" * 50)
    
    # Get MongoDB connection
    mongo_client, db, collection = get_mongodb_collection(
        collection_name=args.collection
    )

    # Crawl the URL with retry logic
    docs = None
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Crawling {args.url}")
            docs = firecrawl.crawl(url=args.url, limit=args.limit)
            logger.info("Crawl completed successfully")
            break
        except Exception as e:
            logger.error(f"Crawl attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All retry attempts failed")
                close_mongodb_client(mongo_client)
                raise

    if docs and hasattr(docs, "data") and docs.data:
        # Get domain name for identification
        domain_name = get_domain_name(args.url)
        
        # Get existing collection to avoid duplicates
        existing_urls = set()
        try:
            existing_docs = collection.find({"source_url": args.url}, {"metadata.url": 1})
            for doc in existing_docs:
                if "metadata" in doc and "url" in doc["metadata"]:
                    existing_urls.add(doc["metadata"]["url"])
            logger.info(f"📋 Found {len(existing_urls)} existing documents from {args.url}")
        except Exception as e:
            logger.warning(f"Could not check existing documents: {e}")
        
        # Prepare documents for MongoDB insertion
        documents_to_insert = []
        skipped_count = 0

        logger.info(f"Processing {len(docs.data)} pages...")
        for page in tqdm(docs.data, desc="Processing pages", unit="page"):
            # Skip if already exists
            page_url = getattr(page.metadata, "url", "") if hasattr(page, "metadata") else ""
            if page_url in existing_urls:
                skipped_count += 1
                continue
                
            page_dict = {
                "markdown": page.markdown if hasattr(page, "markdown") else "",
                "metadata": {},
                "crawled_at": datetime.now(timezone.utc),
                "source_url": args.url,
                "domain": domain_name
            }
            
            if hasattr(page, "metadata") and page.metadata:
                metadata = page.metadata
                # Extract only essential metadata fields
                metadata_dict = {}
                
                # Fields to exclude (unwanted metadata)
                excluded_fields = {
                    "og_title", "og_description", "og_url", "og_image", "og_audio", "og_determiner",
                    "og_locale", "og_locale_alternate", "og_site_name", "og_video", "favicon",
                    "dc_terms_created", "dc_date_created", "dc_date", "dc_terms_type", "dc_type",
                    "dc_terms_audience", "dc_terms_subject", "dc_subject", "dc_description",
                    "dc_terms_keywords", "modified_time", "published_time", "article_tag",
                    "article_section", "keywords", "robots", "status_code", "scrape_id",
                    "num_pages", "content_type", "proxy_used", "cache_state", "cached_at",
                    "credits_used", "error"
                }
                
                # Get all metadata using model_dump() method if available
                try:
                    if hasattr(metadata, "model_dump"):
                        all_metadata = metadata.model_dump()
                    elif hasattr(metadata, "dict"):
                        all_metadata = metadata.dict()
                    else:
                        all_metadata = {}
                    
                    # Filter out excluded fields
                    for key, value in all_metadata.items():
                        if key not in excluded_fields and value is not None:
                            metadata_dict[key] = value
                except:
                    # Fallback to manual extraction of essential fields
                    essential_fields = ["title", "description", "url", "source_url", "language"]
                    for field in essential_fields:
                        try:
                            value = getattr(metadata, field, None)
                            if value and isinstance(value, (str, int, float, bool, list, dict)):
                                metadata_dict[field] = value
                        except:
                            pass
                
                page_dict["metadata"] = metadata_dict
            
            documents_to_insert.append(page_dict)
        
        # Insert documents into MongoDB in batches
        try:
            if documents_to_insert:
                logger.info(f"Inserting {len(documents_to_insert)} documents in batches of {BATCH_SIZE}...")
                inserted_count = 0

                for i in range(0, len(documents_to_insert), BATCH_SIZE):
                    batch = documents_to_insert[i:i+BATCH_SIZE]
                    try:
                        result = collection.insert_many(batch)
                        inserted_count += len(result.inserted_ids)
                        logger.info(f"Inserted batch: {inserted_count}/{len(documents_to_insert)} documents")
                    except Exception as e:
                        logger.error(f"Error inserting batch at index {i}: {e}")
                        # Continue with next batch instead of failing completely
                        continue

                logger.info(f"✅ Successfully inserted {inserted_count} documents into MongoDB")
            else:
                logger.info("ℹ️ No new documents to insert")

            logger.info(f"📄 New pages crawled: {len(documents_to_insert)}")
            logger.info(f"⏭️ Skipped existing pages: {skipped_count}")
            logger.info(f"📊 Total pages processed: {len(docs.data)}")
            logger.info(f"🔗 Source URL: {args.url}")
            logger.info(f"🌐 Domain: {domain_name}")
            logger.info(f"🗄️ Database: {db.name}")
            logger.info(f"📦 Collection: {collection.name}")
            
            # Create backup filename based on domain
            filename = f"{domain_name.replace('.', '_')}_crawl_backup.json"
            crawl_data = {
                "success": docs.success if hasattr(docs, "success") else True,
                "total": docs.total if hasattr(docs, "total") else len(docs.data),
                "data": documents_to_insert
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(crawl_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Backup saved to '{filename}'")
            
        except Exception as e:
            logger.error(f"Error inserting into MongoDB: {e}")
            # Fallback to JSON file if MongoDB fails
            filename = f"{domain_name.replace('.', '_')}_crawl_backup.json"
            crawl_data = {
                "success": docs.success if hasattr(docs, "success") else True,
                "total": docs.total if hasattr(docs, "total") else len(docs.data),
                "data": documents_to_insert
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(crawl_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Backup saved to '{filename}'")
        finally:
            # Close MongoDB connection
            close_mongodb_client(mongo_client)
        
    else:
        logger.error("No data found in crawl result")
        logger.debug(f"Raw result: {docs}")

if __name__ == "__main__":
    main()
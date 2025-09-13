from firecrawl import Firecrawl
from dotenv import load_dotenv
import os
import sys
import argparse
from pymongo import MongoClient
from datetime import datetime, timezone
from urllib.parse import urlparse
import json

load_dotenv()

# Initialize with your API key
firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Initialize MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client.get_database("Cluster0")

def get_domain_name(url):
    """Extract domain name from URL for identification"""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '')
    except:
        return 'unknown'

def main():
    parser = argparse.ArgumentParser(description='Scrape and store web pages using Firecrawl')
    parser.add_argument('url', help='URL to crawl')
    parser.add_argument('--limit', type=int, default=700, help='Maximum pages to crawl (default: 700)')
    parser.add_argument('--collection', default='scraped_pages', help='MongoDB collection name (default: scraped_pages)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting crawl of {args.url}")
    print(f"📊 Limit: {args.limit} pages")
    print(f"🗂️ Collection: {args.collection}")
    print("=" * 50)
    
    # Use the specified collection
    collection = db.get_collection(args.collection)
    
    # Crawl the URL
    docs = firecrawl.crawl(url=args.url, limit=args.limit)

    if docs and hasattr(docs, 'data') and docs.data:
        # Get domain name for identification
        domain_name = get_domain_name(args.url)
        
        # Get existing collection to avoid duplicates
        existing_urls = set()
        try:
            existing_docs = collection.find({"source_url": args.url}, {"metadata.url": 1})
            for doc in existing_docs:
                if 'metadata' in doc and 'url' in doc['metadata']:
                    existing_urls.add(doc['metadata']['url'])
            print(f"📋 Found {len(existing_urls)} existing documents from {args.url}")
        except Exception as e:
            print(f"⚠️ Warning: Could not check existing documents: {e}")
        
        # Prepare documents for MongoDB insertion
        documents_to_insert = []
        skipped_count = 0
        
        for page in docs.data:
            # Skip if already exists
            page_url = getattr(page.metadata, 'url', '') if hasattr(page, 'metadata') else ''
            if page_url in existing_urls:
                skipped_count += 1
                continue
                
            page_dict = {
                'markdown': page.markdown if hasattr(page, 'markdown') else "",
                'metadata': {},
                'crawled_at': datetime.now(timezone.utc),
                'source_url': args.url,
                'domain': domain_name
            }
            
            if hasattr(page, 'metadata') and page.metadata:
                metadata = page.metadata
                # Extract only essential metadata fields
                metadata_dict = {}
                
                # Fields to exclude (unwanted metadata)
                excluded_fields = {
                    'og_title', 'og_description', 'og_url', 'og_image', 'og_audio', 'og_determiner', 
                    'og_locale', 'og_locale_alternate', 'og_site_name', 'og_video', 'favicon',
                    'dc_terms_created', 'dc_date_created', 'dc_date', 'dc_terms_type', 'dc_type',
                    'dc_terms_audience', 'dc_terms_subject', 'dc_subject', 'dc_description',
                    'dc_terms_keywords', 'modified_time', 'published_time', 'article_tag',
                    'article_section', 'keywords', 'robots', 'status_code', 'scrape_id',
                    'num_pages', 'content_type', 'proxy_used', 'cache_state', 'cached_at',
                    'credits_used', 'error'
                }
                
                # Get all metadata using model_dump() method if available
                try:
                    if hasattr(metadata, 'model_dump'):
                        all_metadata = metadata.model_dump()
                    elif hasattr(metadata, 'dict'):
                        all_metadata = metadata.dict()
                    else:
                        all_metadata = {}
                    
                    # Filter out excluded fields
                    for key, value in all_metadata.items():
                        if key not in excluded_fields and value is not None:
                            metadata_dict[key] = value
                except:
                    # Fallback to manual extraction of essential fields
                    essential_fields = ['title', 'description', 'url', 'source_url', 'language']
                    for field in essential_fields:
                        try:
                            value = getattr(metadata, field, None)
                            if value and isinstance(value, (str, int, float, bool, list, dict)):
                                metadata_dict[field] = value
                        except:
                            pass
                
                page_dict['metadata'] = metadata_dict
            
            documents_to_insert.append(page_dict)
        
        # Insert documents into MongoDB
        try:
            if documents_to_insert:
                result = collection.insert_many(documents_to_insert)
                print(f"✅ Successfully inserted {len(result.inserted_ids)} documents into MongoDB")
            else:
                print("ℹ️ No new documents to insert")
            
            print(f"📄 New pages crawled: {len(documents_to_insert)}")
            print(f"⏭️ Skipped existing pages: {skipped_count}")
            print(f"📊 Total pages processed: {len(docs.data)}")
            print(f"🔗 Source URL: {args.url}")
            print(f"🌐 Domain: {domain_name}")
            print(f"🗄️ Database: {db.name}")
            print(f"📦 Collection: {collection.name}")
            
            # Create backup filename based on domain
            filename = f"{domain_name.replace('.', '_')}_crawl_backup.json"
            crawl_data = {
                'success': docs.success if hasattr(docs, 'success') else True,
                'total': docs.total if hasattr(docs, 'total') else len(docs.data),
                'data': documents_to_insert
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(crawl_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Backup saved to '{filename}'")
            
        except Exception as e:
            print(f"❌ Error inserting into MongoDB: {e}")
            # Fallback to JSON file if MongoDB fails
            filename = f"{domain_name.replace('.', '_')}_crawl_backup.json"
            crawl_data = {
                'success': docs.success if hasattr(docs, 'success') else True,
                'total': docs.total if hasattr(docs, 'total') else len(docs.data),
                'data': documents_to_insert
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(crawl_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Backup saved to '{filename}'")
        finally:
            # Close MongoDB connection
            mongo_client.close()
        
    else:
        print("❌ No data found in crawl result")
        print("Raw result:", docs)

if __name__ == "__main__":
    main()
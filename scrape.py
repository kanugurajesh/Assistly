from firecrawl import Firecrawl
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from datetime import datetime, timezone

load_dotenv()

# Initialize with your API key
firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Initialize MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client.get_database("Cluster0")
collection = db.get_collection("scraped_pages")

docs = firecrawl.crawl(url="https://developer.atlan.com", limit=1)

# Store the scraped data
import json

if docs and hasattr(docs, 'data') and docs.data:
    # Prepare documents for MongoDB insertion
    documents_to_insert = []
    
    for page in docs.data:
        page_dict = {
            'markdown': page.markdown if hasattr(page, 'markdown') else "",
            'metadata': {},
            'crawled_at': datetime.now(timezone.utc),
            'source_url': "https://developer.atlan.com"
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
        result = collection.insert_many(documents_to_insert)
        print(f"✅ Successfully inserted {len(result.inserted_ids)} documents into MongoDB")
        print(f"📄 Pages crawled: {len(documents_to_insert)}")
        print(f"🔗 Source URL: https://developer.atlan.com")
        print(f"🗄️ Database: {db.name}")
        print(f"📦 Collection: {collection.name}")
    except Exception as e:
        print(f"❌ Error inserting into MongoDB: {e}")
        # Fallback to JSON file if MongoDB fails
        filename = "atlan_docs_crawl_backup.json"
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
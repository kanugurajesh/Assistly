from firecrawl import Firecrawl
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize with your API key
firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

docs = firecrawl.crawl(url="https://docs.atlan.com", limit=1)

# Store the scraped data
import json

if docs and hasattr(docs, 'data') and docs.data:
    # Save raw data to JSON file
    filename = "atlan_docs_crawl.json"
    
    # Convert to dict for JSON serialization
    crawl_data = {
        'success': docs.success if hasattr(docs, 'success') else True,
        'total': docs.total if hasattr(docs, 'total') else len(docs.data),
        'data': []
    }
    
    for page in docs.data:
        page_dict = {
            'markdown': page.markdown if hasattr(page, 'markdown') else "",
            'metadata': {}
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
            
            # Get all metadata using dict() method if available
            try:
                if hasattr(metadata, 'dict'):
                    all_metadata = metadata.dict()
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
        
        crawl_data['data'].append(page_dict)
    
    # Save to JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(crawl_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Crawl data saved to '{filename}'")
    print(f"📄 Pages crawled: {len(crawl_data['data'])}")
    print(f"🔗 Source URL: https://docs.atlan.com")
    
else:
    print("❌ No data found in crawl result")
    print("Raw result:", docs)
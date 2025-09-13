#!/usr/bin/env python3
"""
Test script for the updated scraping and ingestion pipeline
Demonstrates how to use the new URL-configurable system
"""

import os
import sys
import subprocess
from dotenv import load_dotenv

load_dotenv()

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Command executed successfully")
            print("STDOUT:", result.stdout)
        else:
            print("❌ Command failed")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return False

def main():
    print("🚀 Testing Updated Scraping and Ingestion Pipeline")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "name": "Help for scrape.py",
            "cmd": [sys.executable, "scrape.py", "--help"],
            "description": "Testing scrape.py help command"
        },
        {
            "name": "Help for qdrant-ingestion.py", 
            "cmd": [sys.executable, "qdrant-ingestion.py", "--help"],
            "description": "Testing qdrant-ingestion.py help command"
        }
    ]
    
    # Advanced test cases (commented out for safety)
    advanced_tests = [
        {
            "name": "Scrape docs.atlan.com (10 pages)",
            "cmd": [sys.executable, "scrape.py", "https://docs.atlan.com", "--limit", "10"],
            "description": "Testing small crawl of docs.atlan.com"
        },
        {
            "name": "Incremental ingestion",
            "cmd": [sys.executable, "qdrant-ingestion.py", "--source-url", "https://docs.atlan.com"],
            "description": "Testing incremental ingestion from MongoDB to Qdrant"
        }
    ]
    
    print("\n📋 Running Basic Tests")
    success_count = 0
    total_tests = len(test_cases)
    
    for test in test_cases:
        success = run_command(test["cmd"], test["description"])
        if success:
            success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    print("\n💡 Example Usage Commands:")
    print("-" * 40)
    print("# Scrape docs.atlan.com with 1000 pages limit:")
    print("python scrape.py https://docs.atlan.com --limit 1000")
    print()
    print("# Scrape developer.atlan.com:")
    print("python scrape.py https://developer.atlan.com --limit 700")
    print()
    print("# Ingest all documents to Qdrant:")
    print("python qdrant-ingestion.py")
    print()
    print("# Ingest only docs.atlan.com documents incrementally:")
    print("python qdrant-ingestion.py --source-url https://docs.atlan.com")
    print()
    print("# Recreate Qdrant collection and ingest everything:")
    print("python qdrant-ingestion.py --recreate")
    print()
    print("# Check collection status:")
    print("python -c \"from qdrant_client import QdrantClient; import os; from dotenv import load_dotenv; load_dotenv(); client = QdrantClient(url=os.getenv('QDRANT_URI'), api_key=os.getenv('QDRANT_API_KEY')); info = client.get_collection('atlan_docs'); print(f'Points: {info.points_count}')\"")
    
    print("\n✨ Pipeline Features:")
    print("-" * 20)
    print("✅ URL-configurable scraping with command line arguments")
    print("✅ Domain-specific handling (docs.atlan.com vs developer.atlan.com)")
    print("✅ Incremental processing (avoids duplicate work)")
    print("✅ Progress tracking and ETA estimates")
    print("✅ Robust error handling and resume capability")
    print("✅ Memory-optimized batch processing")
    print("✅ Single collection strategy for unified search")
    print("✅ Source URL filtering for targeted processing")

if __name__ == "__main__":
    main()
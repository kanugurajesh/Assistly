from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client.get_database("Cluster0")
collection = db.get_collection("scraped_pages")

try:
    # Delete all documents in the collection
    result = collection.delete_many({})
    print(f"✅ Successfully deleted {result.deleted_count} documents from the collection")
    print(f"🗄️ Database: {db.name}")
    print(f"📦 Collection: {collection.name}")
except Exception as e:
    print(f"❌ Error deleting documents: {e}")
finally:
    # Close MongoDB connection
    mongo_client.close()
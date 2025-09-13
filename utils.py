"""
Common utilities for the data pipeline scripts
"""
import os
from typing import Tuple
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
DEFAULT_DATABASE = "Cluster0"
DEFAULT_COLLECTION = "scraped_pages"

def get_mongodb_client() -> MongoClient:
    """Get MongoDB client with environment configuration"""
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise EnvironmentError("MONGODB_URI environment variable is required")

    return MongoClient(mongodb_uri)

def get_mongodb_collection(
    database_name: str = DEFAULT_DATABASE,
    collection_name: str = DEFAULT_COLLECTION
) -> Tuple[MongoClient, Database, Collection]:
    """
    Get MongoDB client, database, and collection

    Args:
        database_name: Name of the database
        collection_name: Name of the collection

    Returns:
        Tuple of (client, database, collection)
    """
    client = get_mongodb_client()
    database = client.get_database(database_name)
    collection = database.get_collection(collection_name)

    return client, database, collection

def close_mongodb_client(client: MongoClient) -> None:
    """Safely close MongoDB client connection"""
    try:
        client.close()
    except Exception as e:
        print(f"Warning: Error closing MongoDB connection: {e}")
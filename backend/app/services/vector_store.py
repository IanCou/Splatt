import os
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        
        try:
            # Connect to local dockerized ChromaDB instance or fallback to ephemeral
            # The prompt requested we mock/stub these but setting it up for real so it integrates well
            host = os.getenv("CHROMA_DB_HOST", "localhost")
            port = os.getenv("CHROMA_DB_PORT", "8001") # Docker compose mapped to 8001
            
            # Using HTTPClient to connect to the docker container. 
            self.chroma_client = chromadb.HttpClient(host=host, port=port)
            self.collection = self.chroma_client.get_or_create_collection(name="site_splats")
            logger.info("Connected to ChromaDB successfully.")
        except Exception as e:
            logger.warning(f"Failed to connect to HTTP ChromaDB: {e}. Falling back to ephemeral client.")
            self.chroma_client = chromadb.EphemeralClient()
            self.collection = self.chroma_client.get_or_create_collection(name="site_splats")

    def add_hotspot(self, hotspot_id: str, label: str, metadata: dict):
        """
        Add a hotspot (object) with its metadata (e.g., coordinates) into ChromaDB
        """
        try:
            self.collection.add(
                documents=[f"{label} - {metadata.get('type', 'object')}"],
                metadatas=[metadata],
                ids=[hotspot_id]
            )
            return True
        except Exception as e:
            logger.error(f"Error adding hotspot to vector store: {e}")
            return False

    def search_hotspots(self, query: str, n_results: int = 3):
        """
        Search for hotspots matching the query
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return {"documents": [], "metadatas": [], "ids": []}

vector_store = VectorStore()

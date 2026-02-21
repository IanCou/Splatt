import os
import logging
from typing import Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

supabase: Optional[Client] = None

def get_supabase_client() -> Client:
    global supabase
    if supabase is None:
        url: str = os.environ.get("SUPABASE_URL", "mock_url")
        key: str = os.environ.get("SUPABASE_KEY", "mock_key")
        
        # If running without real credentials, return a dummy or handle gracefully
        if url == "mock_url" or key == "mock_key":
            logger.warning("Using mock Supabase credentials. Database operations will not persist remotely.")
            # For hackathon/demo, we can just return None and handle it in the services,
            # or use an in-memory fallback. Since the prompt requested Supabase for metadata,
            # we initialize a dummy client if real ones aren't provided to prevent crashes.
            try:
                supabase = create_client(url, key)
            except Exception as e:
                logger.error(f"Failed to init Supabase fallback: {e}")
        else:
            supabase = create_client(url, key)
    
    return supabase

def save_project_metadata(project_data: dict):
    """
    Save project metadata to Supabase.
    """
    client = get_supabase_client()
    if client:
        try:
            # mock behavior, assuming a "projects" table exists
            response = client.table("projects").insert(project_data).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error saving to Supabase: {e}")
    return None

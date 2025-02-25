import pytest
import asyncio
import pytest_asyncio
import os
from vector_store.milvus_ops import connect_milvus, init_collection, ensure_connection
from config.config import CONFIG
from dotenv import load_dotenv

# Load the real environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env.local"))

@pytest_asyncio.fixture(scope="session")
async def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_milvus():
    """Set up Milvus connection and collection for all tests."""
    try:
        # Ensure connection is established
        ensure_connection()
        
        # Initialize collection with recreate=True to ensure clean state
        init_collection(recreate=True)
        
        yield
    except Exception as e:
        pytest.fail(f"Failed to set up Milvus: {str(e)}")
    
    # No teardown needed as collection will be cleaned up on recreate 